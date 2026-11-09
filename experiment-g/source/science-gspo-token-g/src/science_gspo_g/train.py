"""Experiment G only: fresh BF16 LoRA, four-worker DDP and colocated vLLM.

No calibration, comparison arms, automatic evaluation, export, or publication.
A launch is explicit; importing this module has no network/model side effects.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import uuid

from .config import load_config, protocol_config, run_directory, runtime_config
from .execution import call_on_main, choose_checkpoint, require_same_training_world_size
from .io import atomic_json, append_jsonl, digest
from .preflight import StartupAudit, check_bf16_device, check_collective

EXPECTED_VERSIONS = {
    "torch": "2.13.0", "vllm": "0.30.0", "transformers": "5.20.2",
    "trl": "1.17.0", "peft": "0.21.2", "accelerate": "1.15.0",
    "typesafe-sdk": "0.7.5", "datasets": "5.1.0", "huggingface-hub": "2.4.0",
    "numpy": "2.3.5", "pandas": "3.1.0", "PyYAML": "6.0.3",
    "flash-linear-attention": "0.6.0", "fla-core": "0.6.0", "causal-conv1d": "1.7.0",
}


def package_versions():
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    for name, expected in EXPECTED_VERSIONS.items():
        if versions[name].split("+", 1)[0] != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {versions[name]}; use the pinned runtime")
    return versions


def source_digest():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))})


def installed_distribution_manifest():
    return dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                       if d.metadata.get("Name")))


def write_protocol(cfg, run_dir, assets, manifest, versions, upstream_contract):
    from .judge import RUBRIC_SPEC, PROCESS_RUBRICS, rubric_hash
    from .prompts import prompt_registry
    from .parsing import COMPLETION_FORMAT_VERSION
    from .weight_sync import WEIGHT_SYNC_MODE
    payload = {
        "format": "science-gspo-token-G-v1", "arm": "G", "config": protocol_config(cfg),
        "completion_format_version": COMPLETION_FORMAT_VERSION,
        "model_revision": assets["model_revision"], "data_manifest": manifest,
        "rubric": RUBRIC_SPEC, "process_rubrics": PROCESS_RUBRICS,
        "rubric_hash": rubric_hash(cfg["judge"]["model"]),
        "prompts": prompt_registry(), "code_hash": source_digest(), "package_versions": versions,
        "upstream_trainer_contract": upstream_contract, "weight_sync": WEIGHT_SYNC_MODE,
        "outcome_formula": "scientific_correctness",
        "advantage_formula": "group_z(scientific_correctness) + I[reasoning_start<=token<reasoning_stop] * process_weight/3 * sum(clip(group_z(process_j), -process_z_clip, process_z_clip))",
        "native_thinking_credit": "correctness advantage across completion; each process advantage across the same original-token thinking span only, excluding native delimiters, final answer, EOS and padding",
        "judge_visibility": {
            "scientific_correctness": "question, conversation context, plain final answer, fallible reference answer; no thinking trace",
            "model_quality": "question, conversation context, whole thinking trace; no target reference or final answer",
            "application_quality": "question, conversation context, whole thinking trace; no target reference or final answer",
            "verification_quality": "question, conversation context, whole thinking trace; no target reference or final answer",
        },
        "normalization": "population standard deviation; configured floor and process clipping",
        "invalid_structure": "ambiguous native boundaries are unavailable; a valid closed thinking trace can retain process feedback without a complete final answer; an empty trace does not erase a valid final answer; missing components are None, never fake scientific zeros",
        "training_censoring": "per-component availability; at least two comparable members per reward dimension",
        "evaluation_policy": "No automatic evaluation, calibration, or comparison arms",
    }
    payload["protocol_hash"] = digest(payload)
    path = run_dir / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Scientific protocol, source or runtime changed; choose a new run name")
    atomic_json(path, payload)
    atomic_json(run_dir / "resolved_config.json", cfg)
    atomic_json(run_dir / "rubric.json", {"scores": RUBRIC_SPEC, "process_rubrics": PROCESS_RUBRICS})
    atomic_json(run_dir / "prompts.json", prompt_registry())
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", default="auto")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, run_name=args.run_name)
    rank, local_rank, world_size = [int(os.environ.get(k, default)) for k, default in
                                  (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"))]
    effective = runtime_config(cfg, world_size)
    plan, run_dir = effective["runtime"], run_directory(cfg)
    pilot_enabled = bool(cfg.get("pilot", {}).get("enabled", False))
    session_id = uuid.uuid4().hex[:12]
    startup = StartupAudit(run_dir, rank, enabled=pilot_enabled, session_id=session_id)
    startup.event("worker_started", world_size=world_size)
    versions = package_versions()
    startup.event("packages_verified")
    if world_size != 4:
        raise ValueError("Experiment G's recorded deployment requires exactly four GPU workers")
    if any(os.environ.get(k, "").lower() == "true" for k in ("ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED")):
        raise ValueError("G uses ordinary DDP; sharded training requires a new protocol")
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise ValueError("Set TYPESAFE_API_KEY at launch; credentials are never included in artifacts")
    os.environ["WANDB_MODE"] = "disabled"

    import torch
    import torch.distributed as dist
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError("One visible CUDA GPU is required per worker")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support is required")
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank),
        timeout=timedelta(seconds=cfg["training"]["distributed_timeout_seconds"]))
    startup.event("distributed_initialized")
    def barrier():
        dist.barrier(device_ids=[local_rank])
    def broadcast(values):
        dist.broadcast_object_list(values, src=0)

    judge, run_lock = None, None
    try:
        if pilot_enabled:
            device_check = check_bf16_device(torch, local_rank)
            collective_check = check_collective(torch, dist, local_rank, world_size)
            startup.event("cuda_nccl_ready", device=device_check, collective=collective_check)
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoTokenizer, TrainerCallback, set_seed
        from trl import GRPOConfig, GRPOTrainer
        from .data import load_prepared
        from .judge import JevEngine, make_reward_functions, install_expected_censoring_filter
        from .preparation import prepare_assets
        from .profiling import StepProfiler
        from .trainer import make_trainer_class, verify_upstream_contract
        from .vllm_tuning import tuned_engine, effective_engine_settings
        from .weight_sync import install_nonmutating_lora_sync
        from .model import load_text_policy, generation_kwargs, effective_sampling_settings, training_kernel_audit
        kernel_audit = training_kernel_audit()
        upstream_contract = verify_upstream_contract(GRPOTrainer)
        GSPOTokenTrainer = make_trainer_class(GRPOTrainer)

        def initialize_run():
            nonlocal run_lock
            import fcntl
            run_dir.mkdir(parents=True, exist_ok=True)
            run_lock = (run_dir / ".training.lock").open("a")
            try:
                fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another process owns this experiment directory") from None
            assets, manifest = prepare_assets(cfg)
            write_protocol(cfg, run_dir, assets, manifest, versions, upstream_contract)
        call_on_main(initialize_run, rank=rank, broadcast=broadcast)
        barrier()
        assets = json.loads((run_dir / "assets.json").read_text())
        protocol = json.loads((run_dir / "protocol.json").read_text())
        protocol_hash = protocol["protocol_hash"]
        if protocol["code_hash"] != source_digest() or protocol["package_versions"] != versions:
            raise RuntimeError("Every rank must use identical code and packages")
        if protocol["upstream_trainer_contract"] != upstream_contract:
            raise RuntimeError("Every rank must use identical TRL hook source")
        datasets = load_prepared(Path(assets["data_dir"]))
        assets_ready_seconds = startup.event("assets_ready", training_rows=len(datasets["train"]))
        finished = (run_dir / "training_completed.json").is_file()
        execution_path = run_dir / "execution.json"
        if rank == 0 and not execution_path.exists():
            atomic_json(execution_path, {"protocol_hash": protocol_hash, "training_world_size": world_size,
                "batch_plan": plan, "created_at": datetime.now(timezone.utc).isoformat()})
        barrier()
        execution = json.loads(execution_path.read_text())
        if execution["protocol_hash"] != protocol_hash:
            raise ValueError("Execution metadata belongs to a different protocol")
        require_same_training_world_size(execution, world_size, training_finished=finished)
        if finished:
            if rank == 0:
                print("Experiment G already finished; no additional training performed", flush=True)
            return
        checkpoint_dir = run_dir / "checkpoints"
        resume = choose_checkpoint(checkpoint_dir, args.resume, protocol_hash=protocol_hash, world_size=world_size)
        rank_dir = run_dir / "ranks" / f"rank_{rank:05d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        profiler = StepProfiler(rank_dir / "update_timings.jsonl", rank=rank)
        session_record = {
            "protocol_hash": protocol_hash, "rank": rank, "local_rank": local_rank,
            "world_size": world_size, "batch_plan": plan, "resume_checkpoint": resume,
            "gpu": torch.cuda.get_device_name(local_rank),
            "gpu_memory_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
            "packages": versions, "all_installed_packages": installed_distribution_manifest(),
        }
        if pilot_enabled:
            session_record.update(startup_session_id=session_id,
                                  assets_ready_seconds=assets_ready_seconds)
        session_path = rank_dir / f"session_{session_id}.json"
        atomic_json(session_path, session_record)
        set_seed(cfg["training"]["seed"])
        tokenizer = AutoTokenizer.from_pretrained(assets["model_snapshot"], local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base, model_audit = load_text_policy(assets["model_snapshot"], cfg, local_rank)
        model_audit["training_kernels"] = kernel_audit
        atomic_json(rank_dir / f"model_loader_{session_id}.json", model_audit)
        initial_adapter, final_adapter = run_dir / "initial_adapter", run_dir / "final_adapter"
        ready = [(initial_adapter / "initial_adapter_complete.json").is_file() if rank == 0 else None]
        broadcast(ready)
        if ready[0]:
            model = PeftModel.from_pretrained(base, str(initial_adapter), is_trainable=True)
        else:
            model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=cfg["model"]["lora_rank"],
                lora_alpha=cfg["model"]["lora_alpha"], lora_dropout=0.0, target_modules="all-linear", bias="none"))
            if rank == 0:
                model.save_pretrained(str(initial_adapter), safe_serialization=True)
                tokenizer.save_pretrained(str(initial_adapter))
                atomic_json(initial_adapter / "initial_adapter_complete.json", {"protocol_hash": protocol_hash})
        barrier()
        startup.event("model_loaded")
        if cfg["training"]["gradient_checkpointing"]:
            model.enable_input_require_grads()
        eos = model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        if tokenizer.eos_token_id not in eos_ids:
            eos_ids.append(tokenizer.eos_token_id)
        judge = JevEngine(effective, rank_dir / "jev_cache.jsonl")
        rewards = make_reward_functions(judge, tokenizer, effective, rank, run_dir, protocol_hash, eos_token_ids=eos_ids)
        install_expected_censoring_filter()

        class InstrumentedTrainer(GSPOTokenTrainer):
            def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
                result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
                from .resume import restore_checkpoint_policy
                self._g_resume_adapter_state = restore_checkpoint_policy(
                    self.model if model is None else model, resume_from_checkpoint)
                return result
            def _generate_single_turn(self, *a, **kw):
                started = time.perf_counter()
                try:
                    result = super()._generate_single_turn(*a, **kw)
                    profiler.add_tokens(sum(len(ids) for ids in result[0]), len(result[0]))
                    return result
                finally:
                    profiler.add_stage("generation_and_sync", time.perf_counter()-started)
            def _calculate_rewards(self, *a, **kw):
                started = time.perf_counter()
                try:
                    return super()._calculate_rewards(*a, **kw)
                finally:
                    profiler.add_stage("judging_and_rank_wait", time.perf_counter()-started)
            def log(self, logs, start_time=None):
                # GRPO.log adds aggregated reward/KL/quality metrics to this
                # dictionary before clearing its internal metric buffers.
                result = super().log(logs, start_time=start_time)
                # Append small scalar metrics, never credentials or model state.
                if self.accelerator.is_main_process:
                    clean = {k: v for k, v in logs.items() if isinstance(v, (str, int, float, bool))
                             and not (isinstance(v, float) and not math.isfinite(v))}
                    nonfinite = [k for k, v in logs.items() if isinstance(v, float) and not math.isfinite(v)]
                    append_jsonl(run_dir / "metrics.jsonl", [{"global_step": self.state.global_step, **clean,
                                                            "nonfinite_metric_keys": nonfinite}])
                return result
            def _save_checkpoint(self, model, trial):
                super()._save_checkpoint(model, trial)
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    atomic_json(checkpoint_dir / f"checkpoint-{self.state.global_step}" / "checkpoint_complete.json",
                        {"protocol_hash": protocol_hash, "world_size": world_size, "global_step": self.state.global_step})
                self.accelerator.wait_for_everyone()

        max_steps = cfg["training"].get("max_steps", -1)
        total_steps = max_steps if max_steps > 0 else math.ceil(len(datasets["train"])/plan["questions_per_update"])
        warmup_steps = math.ceil(total_steps * cfg["training"]["warmup_ratio"])
        trainer_args = GRPOConfig(
            output_dir=str(checkpoint_dir), num_train_epochs=cfg["training"]["epochs"], max_steps=max_steps,
            per_device_train_batch_size=cfg["training"]["microbatch_per_gpu"],
            gradient_accumulation_steps=plan["gradient_accumulation_steps"],
            generation_batch_size=plan["generation_batch_size"], num_generations=cfg["training"]["num_generations"],
            max_completion_length=cfg["generation"]["max_new_tokens"],
            temperature=cfg["generation"]["temperature"], top_p=cfg["generation"]["top_p"], top_k=cfg["generation"]["top_k"],
            min_p=cfg["generation"]["min_p"], repetition_penalty=cfg["generation"]["repetition_penalty"],
            generation_kwargs=generation_kwargs(cfg, eos_ids),
            chat_template_kwargs={"enable_thinking": cfg["generation"]["enable_thinking"]},
            learning_rate=cfg["training"]["learning_rate"], warmup_steps=warmup_steps,
            lr_scheduler_type=cfg["training"]["lr_scheduler_type"], max_grad_norm=1.0, optim="adamw_torch", weight_decay=0.0,
            beta=cfg["training"]["beta"], num_iterations=1, loss_type="grpo", importance_sampling_level="sequence",
            epsilon=cfg["training"]["epsilon"], epsilon_high=cfg["training"]["epsilon_high"],
            scale_rewards="group", multi_objective_aggregation="sum_then_normalize", reward_weights=[1.0]*4,
            use_bias_correction_kl=False,
            bf16=True, fp16=False, gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False}, ddp_find_unused_parameters=False,
            ddp_timeout=cfg["training"]["distributed_timeout_seconds"],
            use_vllm=True, vllm_mode="colocate", vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=cfg["generation"]["vllm_gpu_memory_utilization"],
            vllm_enable_sleep_mode=cfg["generation"]["vllm_enable_sleep_mode"],
            vllm_max_model_length=cfg["model"]["max_prompt_tokens"]+cfg["generation"]["max_new_tokens"],
            vllm_importance_sampling_correction=cfg["generation"]["vllm_importance_sampling_correction"],
            vllm_importance_sampling_mode=cfg["generation"]["vllm_importance_sampling_mode"],
            vllm_importance_sampling_clip_max=cfg["generation"]["vllm_importance_sampling_clip_max"],
            disable_dropout=True, remove_unused_columns=False, mask_truncated_completions=False,
            logging_steps=cfg["training"]["logging_steps"], logging_first_step=True, report_to=[],
            run_name=cfg["run_name"], log_completions=False, num_completions_to_print=0,
            eval_strategy="no", save_strategy="steps", save_steps=cfg["training"]["save_steps"],
            save_total_limit=cfg["training"]["save_total_limit"], save_only_model=False,
            seed=cfg["training"]["seed"], data_seed=cfg["training"]["seed"],
        )
        with tuned_engine(cfg) as audit:
            trainer = InstrumentedTrainer(model=model, args=trainer_args, processing_class=tokenizer,
                reward_funcs=rewards, train_dataset=datasets["train"], experiment_config=effective,
                advantage_audit_path=rank_dir / "token_advantages.jsonl", eos_token_ids=eos_ids)
        engine_audit = effective_engine_settings(trainer.vllm_generation, audit)
        atomic_json(rank_dir / f"sampling_{session_id}.json", effective_sampling_settings(trainer.vllm_generation, cfg))
        atomic_json(rank_dir / f"vllm_engine_{session_id}.json", engine_audit)
        sync_metadata = install_nonmutating_lora_sync(trainer.vllm_generation)
        atomic_json(rank_dir / f"weight_sync_{session_id}.json", sync_metadata)
        if trainer.accelerator.num_processes != world_size or trainer.accelerator.distributed_type.name != "MULTI_GPU":
            raise RuntimeError("Unexpected distributed execution mode")
        startup_seconds = startup.event("trainer_vllm_ready")
        if pilot_enabled:
            session_record["startup_seconds"] = startup_seconds
            atomic_json(session_path, session_record)
        if rank == 0:
            atomic_json(run_dir / "training_args.json", json.loads(trainer_args.to_json_string()))
            print(f"Experiment G: {len(datasets['train'])} training presentations, {total_steps} planned updates", flush=True)
            model.print_trainable_parameters()

        class TimingCallback(TrainerCallback):
            first_update_started = False
            first_update_completed = False

            def on_step_begin(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                if not self.first_update_started:
                    elapsed = startup.event("first_update_started", global_step=state.global_step + 1)
                    if pilot_enabled:
                        session_record["startup_seconds_to_first_update"] = elapsed
                        atomic_json(session_path, session_record)
                    self.first_update_started = True
                profiler.begin(state.global_step+1)
                return control
            def on_step_end(self, args, state, control, **kwargs):
                torch.cuda.synchronize()
                profiler.end(state.global_step, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                if not self.first_update_completed:
                    startup.event("first_update_completed", global_step=state.global_step)
                    self.first_update_completed = True
                return control
        trainer.add_callback(TimingCallback())
        started = time.perf_counter()
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(final_adapter))
        barrier()
        if rank == 0:
            tokenizer.save_pretrained(str(final_adapter))
            adapter_path = final_adapter / "adapter_config.json"
            adapter = json.loads(adapter_path.read_text())
            adapter.update(base_model_name_or_path=cfg["model"]["name"], revision=assets["model_revision"])
            atomic_json(adapter_path, adapter)
            trainer.save_state()
            atomic_json(run_dir / "training_completed.json", {"protocol_hash": protocol_hash,
                "global_step": int(trainer.state.global_step), "epoch": float(trainer.state.epoch),
                "session_training_seconds": time.perf_counter()-started, "training_metrics": result.metrics,
                "run_scope": "full_epoch" if max_steps < 0 else "bounded_updates",
                "evaluation_performed": False, "calibration_performed": False})
            print("Experiment G training finished. Adapter:", final_adapter, flush=True)
        barrier()
        startup.event("training_finished", global_step=int(trainer.state.global_step))
    finally:
        if judge is not None:
            judge.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        if run_lock is not None:
            run_lock.close()


if __name__ == "__main__":
    main()
