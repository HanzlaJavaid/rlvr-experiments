"""Nonmutating LoRA-to-vLLM weight materialization for the pinned runtime.

TRL 1.17's default PEFT synchronization merges and then unmerges the live base
weights. In finite precision, that round trip need not restore the base exactly.
Here each adapted weight is merged in a temporary tensor, matching PEFT 0.21.2's
``safe_merge=True`` arithmetic. The frozen training base is never modified.

This is a deliberately narrow private-API integration: ordinary DDP, colocated
vLLM, TP=1, and one vanilla ``default`` LoRA adapter on unquantized Linear layers.
"""

from __future__ import annotations

from types import MethodType


WEIGHT_SYNC_MODE = "nonmutating_lora_safe_merge_qwen35_text_v1"


def qwen35_checkpoint_name(name):
    """Return the *checkpoint* name expected by vLLM's conditional wrapper.

    HF's causal policy has clean model.layers.* keys. vLLM loads the original
    conditional checkpoint and maps model.language_model.* to its internal
    language_model.model.*. Its native mapper performs QKV, gate/up and GDN
    fusion; the training iterator must not pre-fuse any tensors.
    """
    if name.startswith("model.") and not name.startswith(("model.visual.", "model.language_model.")):
        return "model.language_model." + name[len("model."):]
    if name.startswith("lm_head."):
        return name
    raise ValueError("Unexpected Qwen3.5 causal policy parameter name")


def validate_qwen35_sync_target(backend):
    """Audit actual initialized rollout architecture before any policy sync."""
    config = getattr(backend.model, "config", None)
    if getattr(config, "model_type", None) != "qwen3_5_text":
        return {}
    target = backend.llm.llm_engine.model_executor.driver_worker.model_runner.model
    if type(target).__name__ != "Qwen3_5ForConditionalGeneration":
        raise RuntimeError("Qwen3.5 sync expected the native conditional rollout wrapper")
    if any(name.startswith(("visual.", "model.visual.", "mtp.")) for name, _ in target.named_parameters()):
        raise RuntimeError("Qwen3.5 rollout unexpectedly loaded vision or MTP weights")
    decoder = target.language_model.model
    if getattr(decoder.config, "model_type", None) != "qwen3_5_text":
        raise RuntimeError("Qwen3.5 rollout decoder has the wrong configuration")
    for key in ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads", "num_key_value_heads"):
        if getattr(decoder.config, key) != getattr(config, key):
            raise RuntimeError("Qwen3.5 training and rollout text architectures differ")
    packed = target.packed_modules_mapping
    required = {"qkv_proj": ["q_proj", "k_proj", "v_proj"],
                "gate_up_proj": ["gate_proj", "up_proj"],
                "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
                "in_proj_ba": ["in_proj_b", "in_proj_a"]}
    if any(packed.get(key) != value for key, value in required.items()):
        raise RuntimeError("Qwen3.5 native vLLM projection packing contract changed")
    return {"rollout_class": type(target).__name__, "vision_loaded": False,
            "policy_to_checkpoint_prefix": "model. -> model.language_model.",
            "projection_fusion": "native vLLM QKV/gate-up/GDN loader"}


def _adapted_weights(backend) -> dict:
    """Validate the supported adapter state and map base parameter IDs to layers."""
    import torch
    from peft import PeftModel
    from peft.tuners.lora.layer import Linear, LoraLayer

    if backend.mode != "colocate" or backend.tensor_parallel_size != 1:
        raise ValueError("Nonmutating LoRA sync requires colocated vLLM with TP=1")
    if backend._dist.is_fsdp or backend._dist.is_zero3:
        raise ValueError("Nonmutating LoRA sync supports ordinary DDP only")
    model = backend.model
    if not isinstance(model, PeftModel):
        raise TypeError("Nonmutating LoRA sync requires a PEFT model")
    # TRL 1.17 adds an inactive, frozen "ref" adapter when an already-wrapped
    # PEFT model is passed in. It supplies the KL reference and must never be
    # merged into the inference policy. Only "default" may be active/trainable.
    if not {"default"} <= set(model.peft_config) <= {"default", "ref"} or list(model.active_adapters) != ["default"]:
        raise ValueError("Exactly one active LoRA adapter named 'default' is supported")
    if "ref" in model.peft_config:
        for name, parameter in model.named_parameters():
            if ".ref." in name and parameter.requires_grad:
                raise ValueError("The inactive TRL reference adapter must be frozen")
    config = model.peft_config["default"]
    if getattr(config, "peft_type", None) != "LORA":
        raise ValueError("Only vanilla LoRA is supported by nonmutating weight sync")
    if getattr(config, "bias", "none") != "none" or getattr(config, "lora_bias", False):
        raise ValueError("LoRA bias updates are not supported by nonmutating weight sync")
    if getattr(config, "modules_to_save", None) or getattr(model, "modules_to_save", None):
        raise ValueError("modules_to_save is not supported by nonmutating weight sync")
    if getattr(config, "use_dora", False):
        raise ValueError("DoRA is not supported by nonmutating weight sync")
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise ValueError("Quantized models are not supported by nonmutating weight sync")

    adapted = {}
    for layer in model.modules():
        if not isinstance(layer, LoraLayer):
            continue
        if type(layer) is not Linear or type(layer.get_base_layer()) is not torch.nn.Linear:
            raise TypeError("Only plain PEFT LoRA Linear layers wrapping torch.nn.Linear are supported")
        if layer.merged or layer.disable_adapters:
            raise ValueError("LoRA layers must be enabled and unmerged during weight sync")
        if list(layer.active_adapters) != ["default"]:
            raise ValueError("Every LoRA layer must use only the default adapter")
        if layer.lora_variant or any(layer.use_dora.values()) or any(layer.lora_bias.values()):
            raise ValueError("LoRA variants, DoRA, and adapter bias are unsupported")
        if getattr(layer, "fan_in_fan_out", False):
            raise ValueError("fan_in_fan_out is unsupported for this Linear-only sync")
        if "default" not in layer.lora_A or "default" not in layer.lora_B:
            raise ValueError("Every adapted layer must have both default LoRA matrices")
        weight = layer.get_base_layer().weight
        if weight.requires_grad or not weight.is_floating_point() or weight.is_quantized:
            raise ValueError("Base weights must be frozen, floating-point and unquantized")
        if id(weight) in adapted:
            raise ValueError("Different adapted layers sharing a base parameter are unsupported")
        adapted[id(weight)] = layer
    if not adapted:
        raise ValueError("No supported LoRA Linear layers were found")
    # A delta applied to a tied parameter would also change another module's
    # behavior. The only tie allowed here is among unadapted parameters, such as
    # Qwen's input embeddings and output head.
    adapted_aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in adapted:
            if id(parameter) in adapted_aliases:
                raise ValueError(
                    "An adapted base parameter is aliased by multiple names: "
                    f"{adapted_aliases[id(parameter)]}, {name}"
                )
            adapted_aliases[id(parameter)] = name
    return adapted


def _iter_nonmutating_named_params(backend):
    import torch

    # Revalidate on every sync; a changed adapter state must not silently change
    # the meaning of the weight stream after installation.
    adapted = _adapted_weights(backend)
    model = backend.model
    emitted_adapted = set()
    for name, parameter in model.named_parameters():
        # Match TRL 1.17's PEFT name translation exactly. named_parameters uses
        # its default duplicate removal, preserving the treatment of tied,
        # unadapted embedding/head weights in TRL's original iterator.
        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
        if model.prefix in name:
            continue
        if "original_module" in name:
            raise ValueError("Unexpected modules_to_save/original_module parameter")
        name = backend._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])
        if getattr(getattr(model, "config", None), "model_type", None) == "qwen3_5_text":
            name = qwen35_checkpoint_name(name)
        if parameter.requires_grad:
            raise ValueError(f"Unexpected trainable base parameter: {name}")
        layer = adapted.get(id(parameter))
        with torch.no_grad():
            if layer is None:
                value = parameter.detach()
            else:
                value = parameter.detach().clone()
                delta = layer.get_delta_weight("default")
                if delta.shape != value.shape or delta.device != value.device:
                    raise ValueError(f"LoRA delta shape or device differs from base weight: {name}")
                # Exact arithmetic order in PEFT 0.21.2 Linear.merge(safe_merge=True).
                value.add_(delta.to(dtype=value.dtype))
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Non-finite merged inference weight: {name}")
                emitted_adapted.add(id(parameter))
        yield name, value
    if emitted_adapted != set(adapted):
        raise ValueError("An adapted base parameter was missing from the vLLM weight stream")


def install_nonmutating_lora_sync(backend) -> dict:
    """Replace only TRL's weight iterator, retaining its wake/load/reset lifecycle."""
    adapted = _adapted_weights(backend)
    target_audit = validate_qwen35_sync_target(backend)
    backend._iter_named_params = MethodType(_iter_nonmutating_named_params, backend)
    backend._science_gspo_g_weight_sync_mode = WEIGHT_SYNC_MODE
    return {"mode": WEIGHT_SYNC_MODE, "adapter": "default", "adapted_linear_layers": len(adapted), **target_audit}
