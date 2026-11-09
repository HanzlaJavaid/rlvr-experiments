"""Restore the root policy adapter after pinned Transformers' PEFT reload.

PEFT saves ``default`` at the checkpoint root and inactive ``ref`` in a
subdirectory. Transformers 5.20.2 reloads only adapter subdirectories when any
exist, so it can leave ``default`` at initialization while restoring optimizer
state. Explicitly reload the root policy without replacing the frozen reference.
"""
from __future__ import annotations

from pathlib import Path


def restore_checkpoint_policy(model, checkpoint):
    """Reload only default weights, then enforce the training adapter contract."""
    root = Path(checkpoint)
    if not (root / "adapter_config.json").is_file() or not (root / "adapter_model.safetensors").is_file():
        raise ValueError("Resume requires the checkpoint-root default adapter config and safetensors")
    if not {"default"} <= set(model.peft_config) <= {"default", "ref"}:
        raise ValueError("Resume supports only the default policy and optional frozen ref adapter")
    # load_adapter updates existing parameters in place. Optimizer parameter
    # identities stay intact; inactive reference tensors are never reloaded here.
    model.load_adapter(str(root), "default", is_trainable=True)
    model.set_adapter("default")
    if list(model.active_adapters) != ["default"]:
        raise RuntimeError("Resume did not activate exactly the default policy adapter")
    policy_parameters = 0
    for name, parameter in model.named_parameters():
        if ".default." in name and "lora_" in name:
            policy_parameters += 1
            if not parameter.requires_grad:
                raise RuntimeError("Resume left a default policy adapter parameter frozen")
        elif parameter.requires_grad:
            if ".ref." in name:
                raise RuntimeError("Resume made the reference adapter trainable")
            raise RuntimeError("Resume made a non-policy parameter trainable")
    if not policy_parameters:
        raise RuntimeError("Resume found no trainable default LoRA parameters")
    return {"policy_adapter": "default", "reference_adapter_present": "ref" in model.peft_config,
            "trainable_policy_parameter_tensors": policy_parameters}
