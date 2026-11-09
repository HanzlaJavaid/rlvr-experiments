"""Explicit preparation and launch commands for the independent G experiment."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
from .config import batch_plan, load_config, run_directory
from .execution import torchrun_command


def main(argv=None):
    parser = argparse.ArgumentParser(prog="science-gspo-g")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect-config", "prepare", "launch", "show-prompts", "pilot"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/pilot_g_4x5090.yaml" if name == "pilot" else "configs/experiment_g_4x5090.yaml")
        p.add_argument("--run-name")
        if name in ("inspect-config", "launch", "pilot"):
            p.add_argument("--gpus", type=int, default=4)
        if name in ("launch", "pilot"):
            if name == "launch":
                p.add_argument("--resume", default="auto")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--hourly-usd", type=float, help="Optional actual cluster rate for indicative cost projections")
        if name == "show-prompts":
            p.add_argument("--subject")
    p = sub.add_parser("pilot-report")
    p.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "pilot-report":
        from .pilot_report import write_pilot_report
        print(json.dumps(write_pilot_report(Path(args.run_dir), world_size=4), indent=2))
        return
    cfg = load_config(args.config, run_name=args.run_name)
    if args.command == "show-prompts":
        from .prompts import build_system_prompt, prompt_registry
        print(build_system_prompt(args.subject) if args.subject else json.dumps(prompt_registry(), indent=2))
        return
    if args.command == "prepare":
        from .preparation import prepare_assets
        assets, manifest = prepare_assets(cfg)
        print(json.dumps({"assets": assets, "data": manifest}, indent=2))
        return
    plan = batch_plan(cfg, args.gpus)
    if args.command == "inspect-config":
        from .prompts import prompt_registry
        print(json.dumps({"experiment": cfg["experiment"], "run_dir": str(run_directory(cfg)),
                          "batch_plan": plan, "model": cfg["model"], "generation": cfg["generation"],
                          "training": cfg["training"], "judge_requests_per_update": plan["global_batch_size"]*4,
                          "prompts": prompt_registry(), "launch": cfg["launch"], "pilot": cfg.get("pilot")}, indent=2))
        return
    if args.command == "pilot" or cfg.get("pilot", {}).get("enabled", False):
        if getattr(args, "resume", "auto") not in ("auto", "none"):
            parser.error("Timing pilots require a fresh run; use a new --run-name")
        from .pilot import run_pilot
        run_pilot(args.config, cfg, gpus=args.gpus, dry_run=args.dry_run, hourly_usd=args.hourly_usd)
        return
    if args.hourly_usd is not None:
        parser.error("--hourly-usd is available only for pilot reports")
    command = torchrun_command(args.config, gpus_per_node=args.gpus, resume=args.resume, run_name=args.run_name)
    print("Batch plan:", json.dumps(plan), flush=True)
    print("Command:", shlex.join(command), flush=True)
    if args.dry_run:
        return
    child_env = os.environ.copy()
    child_env.update(WANDB_MODE="disabled", WANDB_DISABLED="true")
    child_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    child_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    child_env.setdefault("OMP_NUM_THREADS", "4")
    process = subprocess.Popen(command, env=child_env, start_new_session=True)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        code = process.wait(timeout=cfg["launch"]["timeout_minutes"]*60)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise RuntimeError("Training workers stopped; stop/delete the Vast instance separately to end rental billing") from None
    finally:
        signal.signal(signal.SIGTERM, previous)
    if code:
        raise subprocess.CalledProcessError(code, command)


if __name__ == "__main__":
    main()
