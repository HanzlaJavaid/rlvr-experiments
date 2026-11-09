"""Deployment and checkpoint rules, independent of torch and GPU availability."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def torchrun_command(
    config_path: str | Path, *, gpus_per_node: int, nodes: int = 1,
    node_rank: int = 0, master_addr: str | None = None, master_port: int = 29500,
    resume: str = "auto", run_name: str | None = None,
) -> list[str]:
    if gpus_per_node < 1 or nodes < 1 or not 0 <= node_rank < nodes:
        raise ValueError("Require positive GPUs/nodes and 0 <= node_rank < nodes")
    if not 1 <= master_port <= 65535:
        raise ValueError("master_port must be between 1 and 65535")
    command = [sys.executable, "-m", "torch.distributed.run", "--max_restarts=0",
               f"--nproc_per_node={gpus_per_node}"]
    if nodes == 1:
        command.append("--standalone")
    else:
        if not master_addr or master_addr in {"localhost", "127.0.0.1", "0.0.0.0"}:
            raise ValueError("Multiple nodes need node 0's reachable private address via --master-addr")
        command += [f"--nnodes={nodes}", f"--node_rank={node_rank}",
                    f"--master_addr={master_addr}", f"--master_port={master_port}"]
    command += ["--module", "science_gspo_g.train", "--config", str(Path(config_path).resolve()),
                "--resume", resume]
    if run_name:
        command += ["--run-name", run_name]
    return command


def checkpoint_is_complete(path: Path, *, protocol_hash: str, world_size: int) -> bool:
    marker = path / "checkpoint_complete.json"
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text())
        if record["protocol_hash"] != protocol_hash or record["world_size"] != world_size:
            return False
        required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json"]
        required += ["rng_state.pth"] if world_size == 1 else [f"rng_state_{r}.pth" for r in range(world_size)]
        if not all((path / name).is_file() for name in required):
            return False
        return (path / "adapter_model.safetensors").is_file()
    except (OSError, ValueError, KeyError):
        return False


def choose_checkpoint(directory: Path, mode: str, *, protocol_hash: str, world_size: int) -> str | None:
    checkpoints = list(directory.glob("checkpoint-*")) if directory.exists() else []
    if mode == "none":
        if checkpoints:
            raise ValueError("This run already has checkpoints. Use --resume auto or a new --run-name")
        return None
    if mode != "auto":
        candidate = Path(mode).expanduser().resolve()
        if not checkpoint_is_complete(candidate, protocol_hash=protocol_hash, world_size=world_size):
            raise ValueError(f"Checkpoint is incomplete or belongs to another protocol/world size: {candidate}")
        return str(candidate)
    completed = [p for p in checkpoints if checkpoint_is_complete(p, protocol_hash=protocol_hash, world_size=world_size)]
    if not completed:
        if checkpoints:
            raise ValueError("No complete checkpoint was found. Preserve this run and choose a new run name to restart")
        return None
    return str(max(completed, key=lambda p: int(p.name.rsplit("-", 1)[1])))


def require_same_training_world_size(execution: dict, world_size: int, *, training_finished: bool) -> None:
    if not training_finished and execution["training_world_size"] != world_size:
        raise ValueError(
            f"This run trained with {execution['training_world_size']} GPU workers. "
            f"Resume it with that count, or use a new run name for {world_size} workers. "
            "Changing checkpoint world size is not an exact training resume."
        )


def call_on_main(function, *, rank: int, broadcast):
    """Broadcast main-process startup failures so peers do not wait at a barrier."""
    error, result, caught = [None], None, None
    if rank == 0:
        try:
            result = function()
        except Exception as exc:
            caught = exc
            error[0] = f"{type(exc).__name__}: {exc}"
    broadcast(error)
    if error[0] is not None:
        raise RuntimeError(f"Main-process operation failed: {error[0]}") from caught
    return result

