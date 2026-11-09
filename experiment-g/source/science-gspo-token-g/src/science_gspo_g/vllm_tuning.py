"""Scoped engine tuning for the pinned TRL 1.17.0 / vLLM 0.30.0 experiment.

TRL does not expose engine kwargs and fixes its scheduler token budget at 4096.
Intercept only its LLM constructor during trainer initialization, restoring the
original in finally. Each DDP worker constructs one trainer in its own process.
Sampling, weight synchronization and sleep/wake remain owned by TRL.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.metadata
from threading import Lock


_CONSTRUCTION_LOCK = Lock()
_INTEGER_OPTIONS = {"max_num_batched_tokens", "max_num_seqs"}
_BOOLEAN_OPTIONS = {"enable_prefix_caching", "enable_chunked_prefill", "enforce_eager", "language_model_only"}
_STRING_OPTIONS = {"mamba_cache_mode"}


def engine_options(cfg: dict) -> dict:
    options = cfg["generation"].get("vllm_engine", {})
    if not isinstance(options, dict):
        raise ValueError("generation.vllm_engine must be a mapping")
    unknown = set(options) - _INTEGER_OPTIONS - _BOOLEAN_OPTIONS - _STRING_OPTIONS
    if unknown:
        raise ValueError(f"Unsupported vLLM engine options: {sorted(unknown)}")
    for key, value in options.items():
        if key in _INTEGER_OPTIONS and (type(value) is not int or value < 1):
            raise ValueError(f"vllm_engine.{key} must be a positive integer")
        if key in _BOOLEAN_OPTIONS and type(value) is not bool:
            raise ValueError(f"vllm_engine.{key} must be a boolean")
        if key == "mamba_cache_mode" and value not in {"none", "align"}:
            raise ValueError("Qwen3.5 supports only none/align mamba cache modes")
    tokens, seqs = options.get("max_num_batched_tokens"), options.get("max_num_seqs")
    if tokens is not None and seqs is not None and tokens < seqs:
        raise ValueError("vLLM token budget must be at least max_num_seqs")
    if options.get("enable_chunked_prefill") is False and tokens is not None:
        context = cfg["model"]["max_prompt_tokens"] + cfg["generation"]["max_new_tokens"]
        if tokens < context:
            raise ValueError("Without chunked prefill, vLLM token budget must cover the full context")
    if options.get("enable_prefix_caching") is False and options.get("mamba_cache_mode") == "align":
        raise ValueError("With prefix caching disabled, vLLM resolves mamba_cache_mode to none; configure none explicitly")
    if cfg["model"].get("language_model_only") is not True or options.get("language_model_only", True) is not True:
        raise ValueError("Qwen3.5 rollout must omit the vision tower")
    # The architecture is fixed by the immutable upstream checkpoint. Keep its
    # conditional wrapper, but skip its vision tower and multimodal profiling.
    return {**options, "language_model_only": True, "model_impl": "vllm"}


@contextmanager
def tuned_engine(cfg: dict):
    """Yield a JSON-safe constructor audit; perform no GPU imports for baseline."""
    options = engine_options(cfg)
    audit = {"overrides": options, "constructor_called": False}
    if not options:
        yield audit
        return
    for name, expected in (("trl", "1.17.0"), ("vllm", "0.30.0")):
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Experiment G engine tuning requires {name}=={expected}")
    module = importlib.import_module("trl.generation.vllm_generation")
    original = module.LLM
    if not _CONSTRUCTION_LOCK.acquire(blocking=False):
        raise RuntimeError("Concurrent/nested vLLM engine construction is unsupported")

    def construct(*args, **kwargs):
        if audit["constructor_called"]:
            raise RuntimeError("Expected exactly one vLLM engine per training worker")
        if args or kwargs.get("tensor_parallel_size") != 1 or kwargs.get("distributed_executor_backend") != "external_launcher":
            raise RuntimeError("Engine tuning requires the pinned TRL colocated TP=1 constructor")
        if kwargs.get("logprobs_mode") != "processed_logprobs" or kwargs.get("return_sampling_mask") is not True:
            raise RuntimeError("Qwen3.5 sampling requires processed sampler logprobs and candidate support")
        merged = {**kwargs, **options}
        if merged["max_num_batched_tokens"] < merged["max_num_seqs"]:
            raise ValueError("vLLM token budget must cover the effective sequence capacity")
        audit["constructor_called"] = True
        # Only record an allowlist: never serialize model paths or credentials.
        keys = set(options) | {"gpu_memory_utilization", "enable_sleep_mode", "max_model_len", "max_num_seqs",
                               "tensor_parallel_size", "max_num_batched_tokens"}
        audit["requested"] = {key: merged[key] for key in sorted(keys) if key in merged}
        return original(**merged)

    try:
        module.LLM = construct
        yield audit
        if not audit["constructor_called"]:
            raise RuntimeError("TRL did not use the expected vLLM constructor; tuning was not applied")
    finally:
        module.LLM = original
        _CONSTRUCTION_LOCK.release()


def effective_engine_settings(backend, audit: dict) -> dict:
    """Record resolved settings and reject ignored overrides before training."""
    config = backend.llm.llm_engine.vllm_config
    fields = {
        "max_num_batched_tokens": ("scheduler_config", "max_num_batched_tokens"),
        "max_num_seqs": ("scheduler_config", "max_num_seqs"),
        "enable_chunked_prefill": ("scheduler_config", "enable_chunked_prefill"),
        "enable_prefix_caching": ("cache_config", "enable_prefix_caching"),
        "gpu_memory_utilization": ("cache_config", "gpu_memory_utilization"),
        "enforce_eager": ("model_config", "enforce_eager"),
        "max_model_len": ("model_config", "max_model_len"),
        "tensor_parallel_size": ("parallel_config", "tensor_parallel_size"),
        "mamba_cache_mode": ("cache_config", "mamba_cache_mode"),
        "model_impl": ("model_config", "model_impl"),
        "logprobs_mode": ("model_config", "logprobs_mode"),
        "return_sampling_mask": ("model_config", "return_sampling_mask"),
    }
    effective = {key: getattr(getattr(config, section), field) for key, (section, field) in fields.items()}
    effective["enable_sleep_mode"] = backend.enable_sleep_mode
    effective["language_model_only"] = config.model_config.multimodal_config.language_model_only
    compilation = getattr(config, "compilation_config", None)
    graph_mode = getattr(compilation, "cudagraph_mode", None)
    effective["cudagraph_mode"] = str(graph_mode) if graph_mode is not None else None
    for key, expected in audit["overrides"].items():
        if effective[key] != expected:
            raise RuntimeError(f"vLLM ignored {key}: requested {expected!r}, resolved {effective[key]!r}")
    if effective["logprobs_mode"] != "processed_logprobs" or effective["return_sampling_mask"] is not True:
        raise RuntimeError("vLLM did not retain processed logprobs and candidate support")
    return {**audit, "effective": effective}
