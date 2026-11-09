"""Qwen3.5 text-only loading and explicit generation contracts.

No model or network access happens on import. The official repository contains
vision and MTP weights; the training policy uses only its causal text backbone.
"""
from __future__ import annotations

import json
from pathlib import Path


THINKING_GENERAL_SAMPLING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repetition_penalty": 1.0,
}
TEXT_LOADER = "qwen3_5_text"


def training_kernel_audit():
    """Fail before model loading if optional CUDA/Triton imports cannot load.

    Installing kernels from a registry is an explicit setup step. No Hub kernel
    downloads or alternative implementations are selected automatically here.
    The pilot still must verify actual GPU forward/backward execution.
    """
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    functions = dict(causal_conv1d_fn=causal_conv1d_fn, causal_conv1d_update=causal_conv1d_update,
                     chunk_gated_delta_rule=chunk_gated_delta_rule,
                     fused_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule)
    if not all(callable(fn) for fn in functions.values()):
        raise RuntimeError("Pinned DeltaNet training kernels are unavailable")
    return {name: fn.__module__ for name, fn in functions.items()}


def generation_settings(cfg):
    """A single allowlisted sampler configuration shared by arguments/audits."""
    generation = cfg["generation"]
    actual = {name: generation[name] for name in THINKING_GENERAL_SAMPLING}
    if actual != THINKING_GENERAL_SAMPLING or generation.get("enable_thinking") is not True:
        raise ValueError("Experiment G requires Qwen3.5 general-task thinking sampling exactly")
    return actual


def generation_kwargs(cfg, eos_ids):
    """Extra kwargs reach TRL's vLLM SamplingParams, including presence penalty."""
    return {**generation_settings(cfg), "stop_token_ids": list(eos_ids)}


def validate_snapshot_config(raw, cfg):
    if cfg["model"].get("loader") != TEXT_LOADER or cfg["model"].get("language_model_only") is not True:
        raise ValueError("Experiment G requires its explicit Qwen3.5 text-only loader")
    if raw.get("model_type") != "qwen3_5" or raw.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("Snapshot is not the reviewed Qwen3.5 conditional-generation checkpoint")
    text = raw.get("text_config", {})
    expected = {"model_type": TEXT_LOADER, "hidden_size": 2560,
                "num_hidden_layers": 32, "vocab_size": 248320,
                "num_attention_heads": 16, "num_key_value_heads": 4}
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("Snapshot text architecture differs from the reviewed Qwen3.5-4B configuration")
    layout = ["linear_attention"] * 3 + ["full_attention"]
    if text.get("layer_types") != layout * 8:
        raise ValueError("Unexpected Qwen3.5 hybrid layer layout")
    maximum = cfg["model"]["max_prompt_tokens"] + cfg["generation"]["max_new_tokens"]
    if text.get("max_position_embeddings", 0) < maximum:
        raise ValueError("Configured prompt plus completion exceeds model context")
    return dict(text)


def validate_loading_info(info):
    """Missing policy weights are fatal, even when HF would only warn."""
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError("Qwen3.5 text policy did not load every expected weight exactly")
    allowed = ("model.visual.", "visual.", "mtp.", "model.mtp.")
    unexpected = list(info.get("unexpected_keys", []))
    if any(not key.startswith(allowed) for key in unexpected):
        raise RuntimeError("Unexpected non-vision/non-MTP weights in Qwen3.5 text policy checkpoint")
    return {"missing_policy_weights": 0, "mismatched_policy_weights": 0,
            "ignored_vision_or_mtp_weights": len(unexpected)}


def validate_text_policy(model):
    if getattr(model.config, "model_type", None) != TEXT_LOADER:
        raise RuntimeError("Training policy must be the Qwen3.5 causal text class")
    names = [name for name, _ in model.named_parameters()]
    if any(name.startswith(("model.visual.", "visual.", "mtp.", "model.mtp.")) for name in names):
        raise RuntimeError("Vision or MTP parameters were loaded into the text-only policy")
    if not names or any(not name.startswith(("model.", "lm_head.")) for name in names):
        raise RuntimeError("Unexpected training policy parameter namespace")
    required = {"model.embed_tokens.weight", "model.norm.weight"}
    if not required <= set(names):
        raise RuntimeError("Qwen3.5 text backbone is incomplete")
    return {"class": type(model).__name__, "model_type": TEXT_LOADER,
            "vision_loaded": False, "mtp_loaded": False,
            "parameter_tensors": len(names),
            "training_kernels": "Transformers pinned implementation; GPU forward/backward unverified"}


def load_text_policy(snapshot, cfg, local_rank, *, torch_module=None, transformers_module=None):
    """Load a causal text policy from a local, immutable official snapshot.

    Explicit key_mapping is idempotent with Transformers 5.20.2's built-in
    qwen3_5_text PrefixChange conversion. Never instantiate a vision tower.
    """
    if torch_module is None:
        import torch as torch_module
    if transformers_module is None:
        import transformers as transformers_module
    raw = json.loads((Path(snapshot) / "config.json").read_text())
    values = validate_snapshot_config(raw, cfg)
    text_config = transformers_module.Qwen3_5TextConfig.from_dict(values)
    text_config.architectures = ["Qwen3_5ForCausalLM"]
    text_config.tie_word_embeddings = bool(raw.get("tie_word_embeddings", values.get("tie_word_embeddings", False)))
    model, loading = transformers_module.Qwen3_5ForCausalLM.from_pretrained(
        str(snapshot), config=text_config, local_files_only=True,
        key_mapping={r"^model\.language_model\.": "model."},
        output_loading_info=True, dtype=torch_module.bfloat16,
        device_map={"": local_rank}, attn_implementation=cfg["model"]["attention_implementation"],
    )
    audit = {**validate_loading_info(loading), **validate_text_policy(model),
             "source_architecture": raw["architectures"][0],
             "source_to_policy_prefix": "model.language_model. -> model.",
             "gpu_forward_backward_tested": False}
    model.config.use_cache = False
    return model, audit


def effective_sampling_settings(backend, cfg):
    """Reject silently dropped settings before the first rollout."""
    expected = generation_settings(cfg)
    extras = getattr(backend, "generation_kwargs", {})
    actual = {key: extras.get(key, getattr(backend, key, None)) for key in expected}
    if actual != expected:
        raise RuntimeError("vLLM generation backend did not retain all requested sampling settings")
    return {"requested": expected, "effective": actual, "thinking_enabled": True,
            "sampler_logprobs": "processed_logprobs",
            "importance_correction": "preserved; penalties and support filtering affect sampler probabilities"}
