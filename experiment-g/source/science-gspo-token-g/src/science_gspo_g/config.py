"""Configuration and world-size-independent batch planning (no GPU imports)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import math


def load_config(path: str | Path, *, run_name: str | None = None, hub_repo: str | None = None) -> dict:
    import yaml

    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if run_name is not None:
        cfg["run_name"] = run_name
    if hub_repo is not None:
        cfg["hub"]["repo_id"] = hub_repo
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    required = ("run_name", "paths", "model", "data", "training", "generation", "judge", "evaluation", "wandb", "hub")
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing configuration field: {key}")
    if not isinstance(cfg["run_name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cfg["run_name"]):
        raise ValueError("run_name must contain only letters, numbers, underscores, dots or hyphens")
    train, gen = cfg["training"], cfg["generation"]
    if cfg.get("experiment") != {"id": "G", "method": "gspo_token", "calibration": False, "comparison_arms": False}:
        raise ValueError("This package runs experiment G only, without calibration or comparison arms")
    if cfg["wandb"]["mode"] != "disabled" or cfg["hub"]["publish_after_training"]:
        raise ValueError("Experiment G uses local logging and does not automatically publish")
    if cfg["evaluation"].get("enabled", False) or cfg["evaluation"]["dev_every_steps"] != 0:
        raise ValueError("Automatic evaluation is disabled for this G-only training build")
    for key in ("global_batch_size", "microbatch_per_gpu", "num_generations", "save_steps", "logging_steps"):
        if not isinstance(train[key], int) or isinstance(train[key], bool) or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if train["num_generations"] < 2:
        raise ValueError("GRPO requires at least two generations per question")
    if train["global_batch_size"] % train["num_generations"]:
        raise ValueError("global_batch_size must be divisible by num_generations")
    if train["epochs"] != 1.0:
        raise ValueError("Experiment G uses one full epoch")
    if len(train["reward_weights"]) != 4 or any(w != 1.0 for w in train["reward_weights"]):
        raise ValueError("Four unit reporting weights required; G uses scientific correctness plus three whole-trace process rewards")
    if train["loss_type"] != "grpo" or train["importance_sampling_level"] != "sequence":
        raise ValueError("G requires sequence-mean GSPO-token loss normalization")
    for key in ("learning_rate", "advantage_std_floor", "process_z_clip", "epsilon", "epsilon_high"):
        value = train[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    for key in ("beta", "process_weight"):
        value = train[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative")
    if "outcome_weights" in train:
        raise ValueError("outcome_weights is obsolete: G uses scientific correctness as its sole outcome reward")
    if train.get("lr_scheduler_type") != "constant_with_warmup":
        raise ValueError("G requires constant_with_warmup; cosine decay is not part of this experiment")
    if (isinstance(train.get("warmup_ratio"), bool)
            or not isinstance(train.get("warmup_ratio"), (int, float))
            or not math.isfinite(train["warmup_ratio"])
            or not 0 <= train["warmup_ratio"] < 1):
        raise ValueError("training.warmup_ratio must be finite and in [0, 1)")
    model = cfg["model"]
    if (model.get("name") != "Qwen/Qwen3.5-4B" or model.get("loader") != "qwen3_5_text"
            or model.get("language_model_only") is not True):
        raise ValueError("G requires the text-only Qwen/Qwen3.5-4B loader")
    if not isinstance(model.get("revision"), str) or not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("Pin model.revision to an immutable 40-character commit SHA")
    sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
    if gen.get("enable_thinking") is not True:
        raise ValueError("G explicitly enables native thinking")
    for name, expected in sampling.items():
        value = gen.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != expected:
            raise ValueError(f"generation.{name} must match the requested general-thinking preset ({expected})")
    if gen.get("vllm_importance_sampling_correction") is not True:
        raise ValueError("G requires processed-sampler importance correction")
    if gen.get("vllm_importance_sampling_mode") not in {"sequence_mask", "sequence_truncate", "token_mask", "token_truncate"}:
        raise ValueError("Unknown vLLM inference correction mode")
    clip_max = gen.get("vllm_importance_sampling_clip_max")
    if isinstance(clip_max, bool) or not isinstance(clip_max, (int,float)) or not math.isfinite(clip_max) or clip_max <= 0:
        raise ValueError("generation.vllm_importance_sampling_clip_max must be finite and positive")
    max_steps = train.get("max_steps", -1)
    if type(max_steps) is not int or (max_steps != -1 and max_steps < 1):
        raise ValueError("training.max_steps must be -1 for one epoch or a positive update limit")
    from .prompts import SUBJECTS
    if set(cfg["data"]["categories"]) != set(SUBJECTS):
        raise ValueError("Experiment G must declare all eight subject categories")
    timeout = cfg.get("launch", {}).get("timeout_minutes")
    if type(timeout) is not int or timeout < 1:
        raise ValueError("launch.timeout_minutes must be a positive integer")
    pilot = cfg.get("pilot", {})
    if not isinstance(pilot, dict) or type(pilot.get("enabled", False)) is not bool:
        raise ValueError("pilot must be a mapping with a boolean enabled flag")
    if pilot.get("enabled", False):
        for key in ("max_updates", "warmup_updates", "min_measured_updates", "preflight_timeout_seconds", "startup_timeout_minutes"):
            if type(pilot.get(key)) is not int or pilot[key] < 1:
                raise ValueError(f"pilot.{key} must be a positive integer")
        if train.get("max_steps") != pilot["max_updates"]:
            raise ValueError("Pilot training.max_steps must equal pilot.max_updates")
        if pilot["min_measured_updates"] < 3 or pilot["warmup_updates"] + pilot["min_measured_updates"] > pilot["max_updates"]:
            raise ValueError("Pilot needs at least three measured updates after warmup")
        if pilot["max_updates"] % train["save_steps"]:
            raise ValueError("Pilot must save a resumable checkpoint on its last update")
        if train["logging_steps"] != 1:
            raise ValueError("Pilot requires per-update loss/gradient logging")
        if pilot["startup_timeout_minutes"] > timeout or pilot["preflight_timeout_seconds"] >= timeout*60:
            raise ValueError("Pilot startup/preflight timeouts must fit inside the launch timeout")
    if gen["tensor_parallel_size"] != 1:
        raise ValueError("This launcher uses one independent colocated vLLM engine per GPU; tensor_parallel_size must be 1")
    if not 0 < gen["vllm_gpu_memory_utilization"] < 1:
        raise ValueError("vllm_gpu_memory_utilization must be between 0 and 1")
    from .vllm_tuning import engine_options
    engine_options(cfg)
    if not 0 < gen["top_p"] <= 1 or gen["temperature"] <= 0:
        raise ValueError("Require positive sampling temperature and 0 < top_p <= 1")
    if cfg["evaluation"]["batch_size_per_rank"] < 1 or cfg["evaluation"]["samples_per_prompt"] < 1:
        raise ValueError("Evaluation sample and batch counts must be positive")
    if cfg["judge"]["global_max_concurrency"] < 1:
        raise ValueError("judge.global_max_concurrency must be positive")
    if not 0 < cfg["data"]["train_fraction"] < 1 or not 0 < cfg["data"]["dev_fraction"] < 1:
        raise ValueError("Split fractions must be positive and below one")
    if cfg["data"]["train_fraction"] + cfg["data"]["dev_fraction"] >= 1:
        raise ValueError("Leave a nonempty test fraction")
    for value in (cfg["model"]["max_prompt_tokens"], cfg["model"]["max_reference_tokens"], gen["max_new_tokens"]):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("Token limits must be positive integers")
    repo = cfg["hub"].get("repo_id")
    if repo is not None and (not isinstance(repo, str) or len(repo.split("/")) != 2 or any(not x for x in repo.split("/"))):
        raise ValueError("hub.repo_id must be NAMESPACE/REPOSITORY")


def batch_plan(cfg: dict, world_size: int) -> dict:
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    train = cfg["training"]
    per_microstep = train["microbatch_per_gpu"] * world_size
    total = train["global_batch_size"]
    if total % per_microstep:
        raise ValueError(f"Global batch {total} must be divisible by microbatch {train['microbatch_per_gpu']} × {world_size} GPUs")
    if cfg["judge"]["global_max_concurrency"] < world_size:
        raise ValueError("global_max_concurrency must be at least world_size (one judge worker per GPU)")
    return {
        "world_size": world_size,
        "gradient_accumulation_steps": total // per_microstep,
        "global_batch_size": total,
        "generation_batch_size": total,
        "completions_per_rank": total // world_size,
        "questions_per_update": total // train["num_generations"],
        "judge_workers_per_rank": cfg["judge"]["global_max_concurrency"] // world_size,
    }


def runtime_config(cfg: dict, world_size: int) -> dict:
    result = deepcopy(cfg)
    result["runtime"] = batch_plan(cfg, world_size)
    result["judge"]["concurrency_per_rank"] = result["runtime"]["judge_workers_per_rank"]
    return result


def run_directory(cfg: dict) -> Path:
    return Path(cfg["paths"]["output_dir"]).expanduser().resolve() / cfg["run_name"]


def protocol_config(cfg: dict) -> dict:
    """Exclude deployment paths and publication destinations from the scientific protocol."""
    result = deepcopy(cfg)
    for key in ("paths", "hub", "wandb", "runtime"):
        result.pop(key, None)
    result["judge"].pop("concurrency_per_rank", None)
    return result

