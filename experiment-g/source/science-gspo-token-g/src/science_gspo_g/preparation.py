"""CPU preparation may run before renting GPUs; writes only requested run paths."""

from __future__ import annotations

import json
from pathlib import Path

from .config import run_directory
from .data import prepare_data
from .io import atomic_json


def prepare_assets(cfg: dict) -> tuple[dict, dict]:
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    run_dir = run_directory(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    asset_path = run_dir / "assets.json"
    old = json.loads(asset_path.read_text()) if asset_path.exists() else None
    if old:
        if old["model"] != cfg["model"]["name"] or old["requested_revision"] != cfg["model"].get("revision"):
            raise ValueError("This run's model identity changed. Choose a new run name")
        revision = old["model_revision"]
    else:
        revision = HfApi().model_info(cfg["model"]["name"], revision=cfg["model"].get("revision")).sha
    model_dir = run_dir / "base_model_snapshot"
    # An immutable SHA and explicit local path also pin the model loaded internally by vLLM.
    snapshot_download(
        cfg["model"]["name"], revision=revision, local_dir=str(model_dir),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken", "LICENSE*"],
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain natural selection."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    if not probe.rstrip().endswith("<think>"):
        raise ValueError("Unexpected chat template: this experiment expects Qwen's prefilled <think> opening")
    manifest = prepare_data(cfg, Path(cfg["paths"]["data_dir"]).expanduser().resolve(), tokenizer, revision)
    assets = {"model": cfg["model"]["name"], "requested_revision": cfg["model"].get("revision"),
              "model_revision": revision, "model_snapshot": str(model_dir),
              "data_dir": str(Path(cfg["paths"]["data_dir"]).expanduser().resolve()),
              "model_loader": cfg["model"]["loader"], "language_model_only": True,
              "chat_template_kwargs": {"enable_thinking": True}}
    atomic_json(asset_path, assets)
    return assets, manifest

