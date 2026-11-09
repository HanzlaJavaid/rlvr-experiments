import ast
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from science_gspo_g.config import batch_plan, load_config, protocol_config, validate_config
from science_gspo_g.io import atomic_json, digest
from science_gspo_g.vllm_tuning import effective_engine_settings, tuned_engine


ROOT = Path(__file__).parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs/experiment_g_4x5090.yaml")


def engine_kwargs():
    return dict(model="local-snapshot", tensor_parallel_size=1,
                distributed_executor_backend="external_launcher", max_num_batched_tokens=4096,
                max_num_seqs=32, max_model_len=16384, enable_sleep_mode=True,
                gpu_memory_utilization=0.55, seed=7, logprobs_mode="processed_logprobs",
                return_sampling_mask=True, trust_remote_code=False, quantization=None)


@pytest.fixture
def trl_module(monkeypatch):
    module = SimpleNamespace(LLM=Mock(return_value=SimpleNamespace(sleep=Mock())))
    monkeypatch.setitem(sys.modules, "trl.generation.vllm_generation", module)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: {"trl": "1.17.0", "vllm": "0.30.0"}[name])
    return module


def resolved_backend(requested):
    return SimpleNamespace(enable_sleep_mode=True, llm=SimpleNamespace(llm_engine=SimpleNamespace(
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(**{k: requested[k] for k in
                ("max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill")}),
            cache_config=SimpleNamespace(**{k: requested[k] for k in
                ("enable_prefix_caching", "gpu_memory_utilization", "mamba_cache_mode")}),
            model_config=SimpleNamespace(**{k: requested[k] for k in ("max_model_len", "enforce_eager", "model_impl")},
                multimodal_config=SimpleNamespace(language_model_only=requested["language_model_only"]),
                logprobs_mode="processed_logprobs", return_sampling_mask=True),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            compilation_config=SimpleNamespace(cudagraph_mode="FULL_AND_PIECEWISE"),
        ))))


def test_constructor_overrides_restore_and_preserve_sampling_contract(cfg, trl_module):
    original = trl_module.LLM
    with tuned_engine(cfg) as audit:
        trl_module.LLM(**engine_kwargs(), hf_token="private-do-not-record")
    assert trl_module.LLM is original
    passed = original.call_args.kwargs
    for key, value in engine_kwargs().items():
        if key not in audit["overrides"]:
            assert passed[key] == value
    assert passed["max_num_batched_tokens"] == 8192
    assert passed["enable_prefix_caching"] is False
    assert passed["language_model_only"] is True
    assert passed["model_impl"] == "vllm"
    assert passed["enable_chunked_prefill"] is True
    assert passed["enforce_eager"] is False
    assert "private-do-not-record" not in str(audit)
    assert "local-snapshot" not in str(audit)
    report = effective_engine_settings(resolved_backend(audit["requested"]), audit)
    assert report["effective"]["max_num_batched_tokens"] == 8192


def test_restores_factory_after_engine_failure(cfg, trl_module):
    original = trl_module.LLM
    original.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        with tuned_engine(cfg):
            trl_module.LLM(**engine_kwargs())
    assert trl_module.LLM is original
    original.side_effect = None
    with tuned_engine(cfg):
        trl_module.LLM(**engine_kwargs())  # lock was released too


def test_missing_or_duplicate_constructor_calls_are_rejected(cfg, trl_module):
    original = trl_module.LLM
    with pytest.raises(RuntimeError, match="not applied"):
        with tuned_engine(cfg):
            pass
    assert trl_module.LLM is original
    with pytest.raises(RuntimeError, match="exactly one"):
        with tuned_engine(cfg):
            trl_module.LLM(**engine_kwargs())
            trl_module.LLM(**engine_kwargs())
    assert original.call_count == 1
    assert trl_module.LLM is original


def test_nested_construction_rejected_without_losing_outer_patch(cfg, trl_module):
    original = trl_module.LLM
    with tuned_engine(cfg):
        with pytest.raises(RuntimeError, match="Concurrent/nested"):
            with tuned_engine(cfg):
                pass
        assert trl_module.LLM is not original
        trl_module.LLM(**engine_kwargs())
    assert trl_module.LLM is original


def test_unrecognized_runtime_fails_before_factory_change(cfg, trl_module, monkeypatch):
    original = trl_module.LLM
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "99.0.0")
    with pytest.raises(RuntimeError, match="requires trl"):
        with tuned_engine(cfg):
            pass
    assert trl_module.LLM is original


@pytest.mark.parametrize("change", [
    {"tensor_parallel_size": 2}, {"distributed_executor_backend": "ray"},
    {"max_num_seqs": 20000},
])
def test_unsupported_topology_or_capacity_rejected(cfg, trl_module, change):
    original = trl_module.LLM
    # Leave sequence capacity under the caller's control in this case.
    cfg["generation"]["vllm_engine"].pop("max_num_seqs")
    with pytest.raises((RuntimeError, ValueError)):
        with tuned_engine(cfg):
            trl_module.LLM(**(engine_kwargs() | change))
    assert original.call_count == 0


@pytest.mark.parametrize("options", [
    {"max_num_batched_tokens": 0}, {"max_num_batched_tokens": True},
    {"max_num_seqs": 3.5}, {"enable_prefix_caching": "true"},
    {"enforce_eager": 0}, {"quantization": "fp8"},
    {"max_num_seqs": 64, "max_num_batched_tokens": 32},
    {"enable_chunked_prefill": False, "max_num_batched_tokens": 8192},
    {"enable_prefix_caching": False, "mamba_cache_mode": "align"},
    [16384],
])
def test_invalid_options_rejected_before_gpu_initialization(cfg, options):
    cfg["generation"]["vllm_engine"] = options
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_ignored_setting_is_never_reported_as_applied(cfg, trl_module):
    with tuned_engine(cfg) as audit:
        trl_module.LLM(**engine_kwargs())
    backend = resolved_backend(audit["requested"])
    backend.llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens = 4096
    with pytest.raises(RuntimeError, match="ignored max_num_batched_tokens"):
        effective_engine_settings(backend, audit)


def test_empty_user_overrides_still_require_text_only_native_model(cfg, trl_module):
    cfg["generation"]["vllm_engine"] = {}
    with tuned_engine(cfg) as audit:
        trl_module.LLM(**engine_kwargs())
    assert audit["overrides"] == {"language_model_only": True, "model_impl": "vllm"}


def test_engine_configuration_is_part_of_protocol(cfg):
    changed = deepcopy(cfg)
    changed["generation"]["vllm_engine"]["max_num_batched_tokens"] = 16384
    assert digest(protocol_config(cfg)) != digest(protocol_config(changed))
    assert batch_plan(cfg, 4) == batch_plan(changed, 4)


def test_full_g_experiment_supports_pinned_engine_tuning(cfg):
    assert not cfg.get("pilot", {}).get("enabled", False)
    validate_config(cfg)


def test_pinned_upstream_initializer_uses_scoped_factory(cfg, trl_module):
    """Optional source-contract test; set SCIENCE_G_TRL_VLLM_SOURCE to the pinned source."""
    path = os.environ.get("SCIENCE_G_TRL_VLLM_SOURCE", str(ROOT / "tests/fixtures/trl_1_17_vllm_generation.py"))
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLLMGeneration")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_init_vllm")
    scope = trl_module.__dict__
    scope.update(is_vllm_available=lambda: True, is_bitsandbytes_available=lambda: False,
                 os=SimpleNamespace(environ={}), ensure_master_addr_port=lambda: None)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    backend = SimpleNamespace(mode="colocate", tensor_parallel_size=1,
        model=SimpleNamespace(name_or_path="local-snapshot"),
        accelerator=SimpleNamespace(num_processes=2, process_index=0, local_process_index=0, wait_for_everyone=Mock()),
        gpu_memory_utilization=0.55, max_model_length=16384, max_num_seqs=32,
        enable_sleep_mode=True, model_impl="vllm", trust_remote_code=False, return_sampling_mask=True)
    original = trl_module.LLM
    with tuned_engine(cfg):
        scope["_init_vllm"](backend)
    assert original.call_args.kwargs["max_num_batched_tokens"] == 8192
    assert original.call_args.kwargs["return_sampling_mask"] is True
    backend.llm.sleep.assert_called_once_with(level=2)
    backend.accelerator.wait_for_everyone.assert_called_once()
