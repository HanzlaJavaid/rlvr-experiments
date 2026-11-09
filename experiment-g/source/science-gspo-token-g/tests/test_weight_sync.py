"""CPU-only contract tests with a NumPy finite-precision tensor stand-in.

These test nonmutation, arithmetic order, parameter naming, and guards. They do
not claim to execute PyTorch, PEFT, vLLM, or CUDA in this development environment.
"""

from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from science_gspo_g.weight_sync import WEIGHT_SYNC_MODE, install_nonmutating_lora_sync


class Tensor:
    def __init__(self, values, dtype=np.float16, *, requires_grad=False):
        self.array = np.asarray(values, dtype=dtype)
        self.requires_grad = requires_grad
        self.is_quantized = False
        self.device = "cpu"

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    def is_floating_point(self):
        return np.issubdtype(self.dtype, np.floating)

    def detach(self):
        result = Tensor(self.array, self.dtype)
        assert np.shares_memory(result.array, self.array)
        return result

    def clone(self):
        return Tensor(self.array.copy(), self.dtype)

    def to(self, *, dtype):
        return Tensor(self.array.astype(dtype), dtype)

    def add_(self, other):
        self.array += other.array
        return self


class BaseLinear:
    def __init__(self, weight):
        self.weight = weight


class LoraLayer:
    pass


class Linear(LoraLayer):
    def __init__(self, weight, delta):
        self.base = BaseLinear(weight)
        self.delta = delta
        self.active_adapters = ["default"]
        self.lora_A = {"default": object()}
        self.lora_B = {"default": object()}
        self.lora_variant = {}
        self.use_dora = {"default": False}
        self.lora_bias = {"default": False}
        self.merged = False
        self.disable_adapters = False
        self.fan_in_fan_out = False

    def get_base_layer(self):
        return self.base

    def get_delta_weight(self, adapter):
        assert adapter == "default"
        return self.delta


class PeftModel:
    prefix = "lora_"
    modules_to_save = None
    active_adapters = ["default"]

    def __init__(self, layer, shared):
        self.layer = layer
        self.shared = shared
        self.peft_config = {"default": SimpleNamespace(peft_type="LORA", bias="none", lora_bias=False, use_dora=False)}
        self.adapter = Tensor([[0.25]], np.float32, requires_grad=True)

    def modules(self):
        return iter([self, self.layer, self.layer.base])

    def named_parameters(self, remove_duplicate=True):
        # A tied embedding/head weight is yielded only once, like PyTorch's
        # default remove_duplicate=True behavior.
        yield "base_model.model.model.embed_tokens.weight", self.shared
        if not remove_duplicate:
            yield "base_model.model.lm_head.weight", self.shared
        yield "base_model.model.model.layers.0.q_proj.base_layer.weight", self.layer.base.weight
        if getattr(self, "adapted_alias", False) and not remove_duplicate:
            yield "base_model.model.model.other_proj.weight", self.layer.base.weight
        yield "base_model.model.model.layers.0.q_proj.lora_A.default.weight", self.adapter

    def merge_adapter(self):
        raise AssertionError("Live model must never be merged")

    def unmerge_adapter(self):
        raise AssertionError("Live model must never be unmerged")


@pytest.fixture
def backend(monkeypatch):
    torch = ModuleType("torch")
    torch.nn = SimpleNamespace(Linear=BaseLinear)
    torch.no_grad = nullcontext
    torch.isfinite = lambda tensor: np.isfinite(tensor.array)
    peft = ModuleType("peft")
    peft.PeftModel = PeftModel
    layer_module = ModuleType("peft.tuners.lora.layer")
    layer_module.Linear = Linear
    layer_module.LoraLayer = LoraLayer
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "peft.tuners", ModuleType("peft.tuners"))
    monkeypatch.setitem(sys.modules, "peft.tuners.lora", ModuleType("peft.tuners.lora"))
    monkeypatch.setitem(sys.modules, "peft.tuners.lora.layer", layer_module)
    layer = Linear(Tensor([[0.9995, 0.125]], np.float16), Tensor([[0.0003, 0.12507]], np.float32))
    model = PeftModel(layer, Tensor([[2.0]], np.float16))
    return SimpleNamespace(
        mode="colocate", tensor_parallel_size=1, model=model,
        _dist=SimpleNamespace(is_fsdp=False, is_zero3=False),
        _fix_param_name_to_vllm=lambda name, extra_prefixes: name.replace("_checkpoint_wrapped_module.", ""),
    )


def test_repeated_sync_never_mutates_base_or_adapter(backend):
    before = backend.model.layer.base.weight.array.copy()
    adapter_before = backend.model.adapter.array.copy()
    metadata = install_nonmutating_lora_sync(backend)
    assert metadata == {"mode": WEIGHT_SYNC_MODE, "adapter": "default", "adapted_linear_layers": 1}
    expected = before.copy()
    expected += backend.model.layer.delta.array.astype(expected.dtype)
    for _ in range(30):
        result = dict(backend._iter_named_params())
        value = result["model.layers.0.q_proj.weight"]
        np.testing.assert_array_equal(value.array, expected)
        assert not np.shares_memory(value.array, backend.model.layer.base.weight.array)
        np.testing.assert_array_equal(backend.model.layer.base.weight.array, before)
        np.testing.assert_array_equal(backend.model.adapter.array, adapter_before)


def test_safe_merge_casts_delta_before_adding(backend):
    # Find a small representable pair where cast-before-add and add-before-cast
    # differ, so this test would catch an accidental promotion of the addition.
    base = np.array([[1.0]], dtype=np.float16)
    delta = np.array([[0.1001]], dtype=np.float32)
    safe = (base + delta.astype(np.float16)).astype(np.float16)
    promoted = (base.astype(np.float32) + delta).astype(np.float16)
    assert not np.array_equal(safe, promoted)
    backend.model.layer.base.weight = Tensor(base)
    backend.model.layer.delta = Tensor(delta, np.float32)
    install_nonmutating_lora_sync(backend)
    result = dict(backend._iter_named_params())
    np.testing.assert_array_equal(result["model.layers.0.q_proj.weight"].array, safe)


def test_unadapted_tied_weight_stream_is_deduplicated(backend):
    install_nonmutating_lora_sync(backend)
    result = list(backend._iter_named_params())
    assert [name for name, _ in result] == ["model.embed_tokens.weight", "model.layers.0.q_proj.weight"]
    assert np.shares_memory(result[0][1].array, backend.model.shared.array)


@pytest.mark.parametrize("attribute,value", [("merged", True), ("disable_adapters", True), ("fan_in_fan_out", True)])
def test_rejects_unsupported_layer_state(backend, attribute, value):
    setattr(backend.model.layer, attribute, value)
    with pytest.raises(ValueError):
        install_nonmutating_lora_sync(backend)


@pytest.mark.parametrize("field", ["use_dora", "lora_bias", "lora_variant"])
def test_rejects_variants(backend, field):
    setattr(backend.model.layer, field, {"default": True})
    with pytest.raises(ValueError):
        install_nonmutating_lora_sync(backend)


@pytest.mark.parametrize("mode,tp,fsdp,zero3", [("server", 1, False, False), ("colocate", 2, False, False), ("colocate", 1, True, False), ("colocate", 1, False, True)])
def test_rejects_unsupported_backends(backend, mode, tp, fsdp, zero3):
    backend.mode, backend.tensor_parallel_size = mode, tp
    backend._dist.is_fsdp, backend._dist.is_zero3 = fsdp, zero3
    with pytest.raises(ValueError):
        install_nonmutating_lora_sync(backend)


def test_revalidates_adapter_state_each_sync(backend):
    install_nonmutating_lora_sync(backend)
    backend.model.layer.merged = True
    with pytest.raises(ValueError):
        list(backend._iter_named_params())


def test_rejects_nonfinite_merged_weights_without_mutation(backend):
    before = backend.model.layer.base.weight.array.copy()
    backend.model.layer.delta = Tensor([[np.nan, 0.0]], np.float32)
    install_nonmutating_lora_sync(backend)
    with pytest.raises(ValueError, match="Non-finite"):
        list(backend._iter_named_params())
    np.testing.assert_array_equal(backend.model.layer.base.weight.array, before)


def test_rejects_trainable_base_weights(backend):
    backend.model.layer.base.weight.requires_grad = True
    with pytest.raises(ValueError, match="frozen"):
        install_nonmutating_lora_sync(backend)


def test_rejects_adapter_bias_and_extra_adapters(backend):
    backend.model.peft_config["default"].bias = "all"
    with pytest.raises(ValueError, match="bias"):
        install_nonmutating_lora_sync(backend)
    backend.model.peft_config["default"].bias = "none"
    backend.model.peft_config["second"] = backend.model.peft_config["default"]
    with pytest.raises(ValueError, match="Exactly one"):
        install_nonmutating_lora_sync(backend)


def test_rejects_adapted_parameter_alias(backend):
    backend.model.adapted_alias = True
    with pytest.raises(ValueError, match="aliased"):
        install_nonmutating_lora_sync(backend)


def test_fixture_exposes_live_merge_unmerge_rounding_drift(backend):
    original = backend.model.layer.base.weight.array.copy()
    delta = backend.model.layer.delta.array
    round_trip = (original.astype(np.float32) + delta).astype(original.dtype)
    round_trip -= delta.astype(original.dtype)
    assert not np.array_equal(round_trip, original)
    install_nonmutating_lora_sync(backend)
    list(backend._iter_named_params())
    np.testing.assert_array_equal(backend.model.layer.base.weight.array, original)


def test_rejects_quantized_model(backend):
    backend.model.is_loaded_in_4bit = True
    with pytest.raises(ValueError, match="Quantized"):
        install_nonmutating_lora_sync(backend)


def test_rejects_modules_to_save(backend):
    backend.model.peft_config["default"].modules_to_save = ["lm_head"]
    with pytest.raises(ValueError, match="modules_to_save"):
        install_nonmutating_lora_sync(backend)


def test_inactive_frozen_trl_reference_is_excluded_from_policy_stream(backend):
    backend.model.peft_config["ref"] = backend.model.peft_config["default"]
    original_parameters = backend.model.named_parameters
    reference = Tensor([[19.0]], np.float32, requires_grad=False)
    def parameters(remove_duplicate=True):
        yield from original_parameters(remove_duplicate=remove_duplicate)
        yield "base_model.model.model.layers.0.q_proj.lora_A.ref.weight", reference
    backend.model.named_parameters = parameters
    install_nonmutating_lora_sync(backend)
    values = dict(backend._iter_named_params())
    assert set(values) == {"model.embed_tokens.weight", "model.layers.0.q_proj.weight"}
    reference.requires_grad = True
    with pytest.raises(ValueError, match="reference adapter must be frozen"):
        list(backend._iter_named_params())


def test_reference_adapter_cannot_be_active_policy(backend):
    backend.model.peft_config["ref"] = backend.model.peft_config["default"]
    backend.model.active_adapters = ["ref"]
    with pytest.raises(ValueError, match="Exactly one"):
        install_nonmutating_lora_sync(backend)


def test_rejects_delta_shape_mismatch(backend):
    backend.model.layer.delta = Tensor([[1.0]], np.float32)
    install_nonmutating_lora_sync(backend)
    with pytest.raises(ValueError, match="shape or device"):
        list(backend._iter_named_params())
