"""Small, explicit GSPO-token extension to the pinned TRL generation machinery.

Pure Python imports keep configuration and CPU tests independent of CUDA/TRL.
The factory permits contract tests with a lightweight stand-in base class.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from collections import Counter
from pathlib import Path

from .advantages import build_advantages
from .io import append_jsonl, atomic_json
from .judge import METRICS
from .loss import gspo_token_loss
from .parsing import parse_completion


class PilotSignalError(RuntimeError):
    """A pilot's first rollout cannot supply a policy-learning signal."""

EXPECTED_TRL_TRAINER_SHA256 = "28f815d86f6416842debd444dcbd815b4373fe86ad2917e180b4d1780a220605"


def verify_upstream_contract(base_class):
    """Fail on incompatible private hooks and record exact installed source."""
    source_path = Path(inspect.getsourcefile(base_class))
    actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual != EXPECTED_TRL_TRAINER_SHA256:
        raise RuntimeError("Installed TRL trainer source differs from the reviewed 1.17.0 wheel; review the integration before changing the pin")
    requirements = {
        "_calculate_rewards": ("completion_ids_list", "gather", "rewards_per_func"),
        "_generate_and_score_completions": ("unscorable_mask", "old_per_token_logps", "sampling_mask", "completion_mask"),
        "_get_per_token_logps_and_entropies": ("sampling_mask",),
        "compute_loss": ("_compute_loss",),
    }
    hashes = {"module_sha256": actual}
    for name, markers in requirements.items():
        source = inspect.getsource(getattr(base_class, name))
        if any(marker not in source for marker in markers):
            raise RuntimeError(f"Installed TRL hook {name} is incompatible with Experiment G")
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    return hashes


def _compact_segments(advantages, length):
    if not length:
        return []
    output, start, value = [], 0, advantages[0]
    for i in range(1, length):
        if advantages[i] != value:
            output.append({"start": start, "stop": i, "advantage": value})
            start, value = i, advantages[i]
    output.append({"start": start, "stop": length, "advantage": value})
    return output


# A diagnostic/first-rollout floor only: the objective retains the exact supplied
# correction. v2 had positive ~1e-12 weights and effectively no learning signal.
MIN_USABLE_INFERENCE_CORRECTION = 1e-6


def effective_policy_signal(inputs, *, require_inference_correction=False):
    """Count possible policy-gradient rows after correction and sampling support.

    TRL's sampling_mask contains allowed vocabulary IDs (B,T,K), padded with -1.
    It is not a boolean token mask. A singleton support has constant log-probability
    zero and cannot supply a policy gradient, even with nonzero advantages.
    Correction below the numerical-readiness floor is not certified as usable.
    These counts are diagnostics, not a guarantee of improvement. The floor never
    clips or replaces correction weights in the actual loss.
    """
    import torch

    ids, advantages, mask = (inputs[name] for name in ("completion_ids", "advantages", "completion_mask"))
    if ids.ndim != 2 or advantages.shape != ids.shape or mask.shape != ids.shape:
        raise ValueError("Effective policy signal requires aligned completion IDs, advantages and mask")
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Completion mask must be finite and binary")
    active = mask.bool()
    if not torch.isfinite(advantages[active]).all():
        raise ValueError("Nonfinite active policy advantages")
    raw_tokens = active & advantages.ne(0)
    correction = inputs.get("importance_sampling_ratio")
    if correction is None:
        if require_inference_correction:
            raise RuntimeError("Missing vLLM inference/training importance correction")
        weighted_tokens = raw_tokens
    else:
        if correction.shape not in (ids.shape, (ids.shape[0], 1)):
            raise ValueError("Inference correction must have shape (B,T) or (B,1)")
        correction = torch.broadcast_to(correction, ids.shape)
        if not torch.isfinite(correction[active]).all() or (correction[active] < 0).any():
            raise ValueError("Inference correction must be finite and nonnegative")
        weighted_tokens = raw_tokens & correction.gt(MIN_USABLE_INFERENCE_CORRECTION)
    support = inputs.get("sampling_mask")
    effective_tokens = weighted_tokens
    if support is not None:
        if (support.ndim != 3 or support.shape[:2] != ids.shape or support.shape[-1] < 1
                or support.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)):
            raise ValueError("Sampling support must contain integer IDs with shape (B,T,K)")
        if ((support < -1) & active.unsqueeze(-1)).any():
            raise ValueError("Sampling support IDs must be nonnegative or the -1 padding sentinel")
        if not ((support == ids.unsqueeze(-1)).any(-1) | ~active).all():
            raise ValueError("Sampled completion token is absent from its sampling support")
        effective_tokens = weighted_tokens & (support.ge(0).sum(-1) > 1)
    return torch.stack([raw_tokens.any(-1).sum(), weighted_tokens.any(-1).sum(),
                        effective_tokens.any(-1).sum()])


def make_trainer_class(base_class):
    class GSPOTokenTrainer(base_class):
        def __init__(self, *args, experiment_config, advantage_audit_path=None,
                     eos_token_ids=None, **kwargs):
            self.experiment_config = experiment_config
            self.advantage_audit_path = Path(advantage_audit_path) if advantage_audit_path else None
            self.experiment_eos_ids = eos_token_ids
            self._g_reward_capture = None
            self._g_first_rollout_checked = False
            super().__init__(*args, **kwargs)
            if self.loss_type != "grpo" or self.importance_sampling_level != "sequence":
                raise ValueError("G uses sequence-mean GSPO-token; configure loss_type=grpo and sequence importance")
            if (getattr(self, "off_policy_mask_threshold", None) is not None
                    or getattr(self, "top_entropy_quantile", 1.0) != 1.0
                    or getattr(self, "_entropy_bonus_enabled", False)
                    or getattr(self, "aux_loss_enabled", False)
                    or getattr(self.args, "use_bias_correction_kl", False)
                    or getattr(self.args, "delta", None) is not None):
                raise ValueError("G does not support extra entropy/off-policy/router/KL-bias objective variants")

        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            import torch.distributed as dist

            rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
            local = []
            for row, ids in zip(inputs, completion_ids_list, strict=True):
                parsed = parse_completion(ids, self.processing_class,
                    self.experiment_config["generation"]["max_new_tokens"], self.experiment_eos_ids)
                if parsed["generated_tokens"] != len(ids):
                    raise RuntimeError("Upstream completion contains tokens after EOS; refuse misaligned token advantages")
                local.append({"example_id": row["example_id"], "status": parsed["status"],
                    "reasoning_token_span": parsed.get("reasoning_token_span"),
                    "process_start_token": parsed.get("process_start_token"),
                    "visible_start_token": parsed.get("visible_start_token"),
                    "process_token_spans": parsed.get("process_token_spans"),
                    "length": len(ids), "parsed_length": parsed["generated_tokens"]})
            # Same rank-concatenated ordering as Accelerate's reward gather.
            if dist.is_initialized():
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, local)
            else:
                gathered = [local]
            if any(len(part) != len(local) for part in gathered):
                raise RuntimeError("G requires equally sized rank-local generation batches")
            metadata = [record for part in gathered for record in part]
            if len(metadata) != rewards.shape[0] or rewards.shape[1] != len(METRICS):
                raise RuntimeError("Global judge scores and generation metadata are misaligned")
            self._g_reward_capture = (rewards.detach().cpu().tolist(), metadata, len(local))
            return rewards

        def _generate_and_score_completions(self, inputs):
            import torch

            self._g_reward_capture = None
            mode = "train" if self.model.training else "eval"
            # Retain earlier rollout contributions in the current logging window.
            # Only replace primary metrics appended by this generation call.
            primary_lengths = {name: len(self._metrics[mode].get(name, []))
                               for name in ("reward", "reward_std")}
            output = super()._generate_and_score_completions(inputs)
            if self._g_reward_capture is None:
                raise RuntimeError("Upstream generation did not invoke the required reward hook")
            scores, metadata, local_count = self._g_reward_capture
            cfg = self.experiment_config["training"]
            built = build_advantages(scores, metadata, group_size=self.num_generations,
                process_weight=cfg["process_weight"], std_floor=cfg["advantage_std_floor"],
                process_z_clip=cfg["process_z_clip"])
            start = self.accelerator.process_index * local_count
            stop = start + local_count
            local_width = output["completion_ids"].shape[1]
            dense = [row[:local_width] for row in built["advantages"][start:stop]]
            if any(len(row) != local_width for row in dense):
                # Upstream pad_to_multiple_of may add right padding beyond global max.
                dense = [row + [0.0] * (local_width - len(row)) for row in dense]
            output["advantages"] = torch.tensor(dense, dtype=torch.float32,
                                                  device=output["completion_ids"].device)
            active = torch.tensor(built["active_rows"][start:stop],
                                  device=output["completion_ids"].device)
            output["completion_mask"] = output["completion_mask"] * active[:, None]
            # The native sampling mask, old/reference probabilities and inference
            # correction stay intact; only the policy advantage and censor mask change.
            output["advantages"] *= output["completion_mask"]
            mode = "train" if self.model.training else "eval"
            local_signal = effective_policy_signal(output, require_inference_correction=(
                self.use_vllm and self.vllm_importance_sampling_correction))
            global_signal = self.accelerator.reduce(local_signal, reduction="sum").tolist()
            self._metrics[mode]["g/effective_policy_signal_fraction"].append(global_signal[2] / len(metadata))
            self._metrics[mode]["g/corrected_policy_signal_fraction"].append(global_signal[1] / len(metadata))
            # Reporting uses every available raw correctness score, including
            # singleton cohorts that cannot supply a relative policy advantage.
            # Never let a process-only batch change "reward" into a process sum.
            outcome = [float(row[0]) for row, available in zip(
                scores, built["reward_availability"], strict=True) if available[0]]
            self._metrics[mode]["g/scientific_correctness_availability_fraction"].append(
                len(outcome) / len(scores))
            for name, before in primary_lengths.items():
                values = self._metrics[mode].get(name)
                if values is not None:
                    del values[before:]
                    if not values:
                        self._metrics[mode].pop(name, None)
            if outcome:
                mean = sum(outcome) / len(outcome)
                self._metrics[mode]["g/outcome_reward"].append(mean)
                self._metrics[mode]["reward"].append(mean)
                self._metrics[mode]["reward_std"].append(
                    math.sqrt(sum((v-mean)**2 for v in outcome)/len(outcome)))
            self._metrics[mode]["g/skipped_groups"].append(built["skipped_groups"])
            self._metrics[mode]["g/active_fraction"].append(sum(built["active_rows"]) / len(metadata))
            if self.advantage_audit_path:
                records = []
                for i in range(start, stop):
                    records.append({"global_step": self.state.global_step, **metadata[i],
                        "global_batch_row": i, "outcome_reward": built["outcome_rewards"][i],
                        "outcome_advantage": built["outcome_advantages"][i],
                        "process_z": built["process_z"][i], "active": built["active_rows"][i],
                        "reward_availability": built["reward_availability"][i],
                        "outcome_cohort": built["outcome_cohorts"][i // self.num_generations],
                        "advantage_segments": _compact_segments(built["advantages"][i], metadata[i]["length"])})
                append_jsonl(self.advantage_audit_path, records)
            # All ranks decide from the same globally gathered rewards and
            # token advantages, so one rank cannot continue while another
            # rejects a locally uninformative batch. Check before any backward
            # pass rather than spending the complete pilot on zero-signal data.
            if (self.experiment_config.get("pilot", {}).get("enabled", False)
                    and mode == "train" and not self._g_first_rollout_checked):
                self._g_first_rollout_checked = True
                complete = sum(row["status"] == "complete" for row in metadata)
                scored = sum(any(mask) for mask in built["reward_availability"])
                signal = sum(built["active_rows"][i] and any(value != 0.0 for value in row)
                             for i, row in enumerate(built["advantages"]))
                reason = ("no_available_rewards" if scored == 0 else
                          "no_nonzero_policy_advantages" if signal == 0 else
                          "no_effective_policy_signal" if global_signal[2] == 0 else None)
                diagnostic = {
                    "format": "science-g-first-rollout-signal-v1",
                    "passed": reason is None, "reason": reason,
                    "global_step": self.state.global_step,
                    "scope": "globally_gathered_first_training_rollout",
                    "rank": self.accelerator.process_index,
                    "completion_records": len(metadata),
                    "completion_status_counts": dict(Counter(row["status"] for row in metadata)),
                    "complete_completions": complete, "scored_complete_completions": scored,
                    "scoreable_completions": scored,
                    "nonzero_policy_signal_sequences": signal,
                    "corrected_policy_signal_sequences": int(global_signal[1]),
                    "effective_policy_signal_sequences": int(global_signal[2]),
                    "sampling_support_semantics": "allowed vocabulary IDs; singleton support has zero policy gradient",
                    "inference_correction_checked": True,
                    "inference_correction_readiness_floor": MIN_USABLE_INFERENCE_CORRECTION,
                    "checked_before_backward": True,
                }
                if self.advantage_audit_path:
                    atomic_json(self.advantage_audit_path.parent / "first_rollout_signal.json", diagnostic)
                if reason is not None:
                    raise PilotSignalError(
                        f"Pilot stopped before backward: {reason}; first rollout has "
                        f"{scored}/{len(metadata)} scoreable completions and {signal} sequences "
                        "with nonzero policy advantages. Inspect first_rollout_signal.json.")
            self._g_reward_capture = None
            return output

        def _compute_loss(self, model, inputs):
            import torch

            if "tool_mask" in inputs or "pixel_values" in inputs:
                raise ValueError("Experiment G supports text-only, tool-free completions")
            completion_mask = inputs["completion_mask"]
            input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
            attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
            logps, entropies, _ = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, inputs["completion_ids"].size(1),
                compute_entropy=True, compute_aux_loss=False,
                sampling_mask=inputs.get("sampling_mask"))
            old = inputs.get("old_per_token_logps", logps.detach())
            mode = "train" if self.model.training else "eval"
            accumulation = self.current_gradient_accumulation_steps if mode == "train" else 1
            correction = inputs.get("importance_sampling_ratio")
            if self.use_vllm and self.vllm_importance_sampling_correction and correction is None:
                raise RuntimeError("Missing vLLM inference/training importance correction")
            loss, diagnostics = gspo_token_loss(logps, old, inputs["advantages"], completion_mask,
                epsilon_low=self.epsilon_low, epsilon_high=self.epsilon_high,
                ref_logps=inputs.get("ref_per_token_logps"), beta=self.beta,
                inference_correction=correction, accumulation_steps=accumulation)
            def mean_active(value):
                value = torch.where(completion_mask.bool(), value.float(), 0.0)
                totals = self.accelerator.reduce(torch.stack([value.sum(), completion_mask.sum().float()]), reduction="sum")
                return (totals[0] / totals[1].clamp_min(1)).item()
            self._metrics[mode]["entropy"].append(mean_active(entropies.detach()))
            self._metrics[mode]["kl"].append(mean_active(diagnostics["kl"]))
            self._metrics[mode]["clip_ratio/low_mean"].append(mean_active(diagnostics["low_clipped"]))
            self._metrics[mode]["clip_ratio/high_mean"].append(mean_active(diagnostics["high_clipped"]))
            self._metrics[mode]["clip_ratio/region_mean"].append(mean_active(diagnostics["low_clipped"] | diagnostics["high_clipped"]))
            return loss

    GSPOTokenTrainer.__name__ = "GSPOTokenTrainer"
    return GSPOTokenTrainer
