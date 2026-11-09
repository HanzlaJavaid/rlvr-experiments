"""CPU-only reports from completed, rank-aligned Experiment G pilot journals.

Reports never launch work, repair another process's journal, infer missing
measurements, or treat a short projection as a full-run performance guarantee.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import statistics

from .execution import checkpoint_is_complete
from .io import atomic_json, read_jsonl


def _finite(value, *, positive=False):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and (not positive or value > 0)


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _statistics(values):
    if not values:
        return None
    return {"min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


def summarize_pilot(run_dir, cfg=None, world_size=4):
    """Summarize existing journals, including failed or interrupted startups.

    Duplicate optimizer steps are ambiguous resumed attempts and are excluded
    from every rank. Their presence suppresses projections and success claims.
    A response journal's global_step is the *preceding* optimizer step, so its
    records belong to update global_step+1.
    """
    root = Path(run_dir)
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    errors = []

    def read_object(path):
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected JSON object")
            return value
        except (OSError, ValueError) as exc:
            errors.append({"file": str(path.relative_to(root)), "error": str(exc)})
            return {}

    def read_rows(path):
        try:
            rows = read_jsonl(path, repair_tail=False)
            if any(not isinstance(row, dict) for row in rows):
                raise ValueError("expected JSONL objects")
            return rows
        except (OSError, ValueError) as exc:
            errors.append({"file": str(path.relative_to(root)), "error": str(exc)})
            return []

    if cfg is None:
        cfg = read_object(root / "resolved_config.json") or read_object(root / "launch_config.json")
    settings = cfg.get("pilot", {})
    expected = settings.get("max_updates", cfg.get("training", {}).get("max_steps", 12))
    warmup = settings.get("warmup_updates", 2)
    minimum = settings.get("min_measured_updates", 3)
    if type(expected) is not int or expected < 1:
        raise ValueError("A pilot requires a positive planned update count")
    if type(warmup) is not int or warmup < 0 or type(minimum) is not int or minimum < 1:
        raise ValueError("Invalid pilot measurement settings")
    protocol = read_object(root / "protocol.json")
    launcher = read_object(root / "launcher_metrics.json")
    preflight = read_object(root / "preflight.json")
    completed = read_object(root / "training_completed.json")
    epoch_steps = protocol.get("data_manifest", {}).get("optimizer_steps_per_epoch")
    if type(epoch_steps) is not int or epoch_steps < 1:
        epoch_steps = None

    rank_rows, duplicate_steps, invalid_steps = [], set(), set()
    startup, missing_ranks, first_rollouts = [], [], []
    for rank in range(world_size):
        rank_dir = root / "ranks" / f"rank_{rank:05d}"
        signal_check = read_object(rank_dir / "first_rollout_signal.json")
        if signal_check:
            first_rollouts.append({"rank": rank, **signal_check})
        rows = read_rows(rank_dir / "update_timings.jsonl")
        by_step = {}
        if not rows:
            missing_ranks.append(rank)
        for row in rows:
            step = row.get("global_step")
            if type(step) is not int or step < 1:
                errors.append({"file": str((rank_dir / 'update_timings.jsonl').relative_to(root)),
                               "error": "invalid optimizer step"})
                continue
            if step in by_step:
                duplicate_steps.add(step)
            by_step[step] = row
            if row.get("rank", rank) != rank or not _finite(row.get("update_seconds"), positive=True):
                invalid_steps.add(step)
            for key in ("generated_tokens", "generated_completions"):
                value = row.get(key)
                if type(value) is not int or value < 0:
                    invalid_steps.add(step)
        rank_rows.append(by_step)
        events = read_rows(rank_dir / "startup_events.jsonl")
        session_ids = sorted({row.get("session_id") for row in events if isinstance(row.get("session_id"), str)})
        row = {"rank": rank, "sessions": len(session_ids), "events": [e.get("event") for e in events],
               "last_event": events[-1].get("event") if events else None,
               "last_event_elapsed_seconds": events[-1].get("elapsed_seconds") if events else None,
               "seconds_to_trainer_ready": None, "seconds_to_first_update": None,
               "seconds_to_first_update_completed": None}
        # Multiple startup attempts must not be silently merged into one timing.
        if len(session_ids) == 1:
            for event, field in (("trainer_vllm_ready", "seconds_to_trainer_ready"),
                                 ("first_update_started", "seconds_to_first_update"),
                                 ("first_update_completed", "seconds_to_first_update_completed")):
                matches = [e.get("elapsed_seconds") for e in events if e.get("event") == event]
                if len(matches) == 1 and _finite(matches[0]) and matches[0] >= 0:
                    row[field] = matches[0]
        startup.append(row)

    shared = set.intersection(*(set(rows) for rows in rank_rows))
    all_seen = set.union(*(set(rows) for rows in rank_rows))
    excluded_steps = duplicate_steps | invalid_steps
    steps = sorted(shared - excluded_steps)
    warm_steps, measured_steps = steps[:warmup], steps[warmup:]
    updates = []
    for step in steps:
        rows = [by_step[step] for by_step in rank_rows]
        stages = sorted(set().union(*(row.get("stages_seconds", {}).keys() for row in rows)))
        stage_times = {}
        for name in stages:
            vals = [row.get("stages_seconds", {}).get(name) for row in rows]
            if all(_finite(v) and v >= 0 for v in vals):
                stage_times[name] = max(vals)
        updates.append({"step": step, "wall_seconds": max(row["update_seconds"] for row in rows),
                        "generated_tokens": sum(row["generated_tokens"] for row in rows),
                        "generated_completions": sum(row["generated_completions"] for row in rows),
                        "slowest_rank_stage_seconds": stage_times})
    measured = [row for row in updates if row["step"] in measured_steps]
    walls = [row["wall_seconds"] for row in measured]
    total_seconds = sum(walls)
    total_tokens = sum(row["generated_tokens"] for row in measured)
    stage_names = sorted(set().union(*(row["slowest_rank_stage_seconds"].keys() for row in measured))) if measured else []
    stages = {name: _statistics([row["slowest_rank_stage_seconds"][name] for row in measured
                                if name in row["slowest_rank_stage_seconds"]]) for name in stage_names}
    memory = []
    for rank, rows in enumerate(rank_rows):
        record = {"rank": rank}
        for name in ("peak_allocated_gib", "peak_reserved_gib"):
            values = [rows[step].get(name) for step in steps]
            values = [value for value in values if _finite(value) and value >= 0]
            record[name] = max(values) if values else None
        memory.append(record)

    status_counts, signal_counts = Counter(), Counter()
    judge_scored, complete_judge_scored, ambiguous_audit_steps = 0, 0, set()
    reward_available, reward_unavailable = Counter(), Counter()
    judge_failure_counts = Counter()
    audit_counts = []
    for rank in range(world_size):
        rank_dir = root / "ranks" / f"rank_{rank:05d}"
        generations = read_rows(rank_dir / "training_generations.jsonl")
        advantages = read_rows(rank_dir / "token_advantages.jsonl")
        generation_counts = Counter(row["global_step"]+1 for row in generations
                                    if type(row.get("global_step")) is int and row["global_step"]+1 in steps)
        advantage_counts = Counter(row["global_step"]+1 for row in advantages
                                   if type(row.get("global_step")) is int and row["global_step"]+1 in steps)
        for step in steps:
            expected_count = rank_rows[rank][step]["generated_completions"]
            audit_counts.append({"rank": rank, "step": step, "expected_completions": expected_count,
                                 "generation_records": generation_counts[step],
                                 "advantage_records": advantage_counts[step],
                                 "complete": generation_counts[step] == advantage_counts[step] == expected_count})
        batches = {}
        for row in generations:
            prior = row.get("global_step")
            if type(prior) is int and prior+1 in steps:
                batch = row.get("batch_id")
                if batch is not None:
                    batches.setdefault(prior+1, set()).add(batch)
        ambiguous = {step for step, batch_ids in batches.items() if len(batch_ids) > 1}
        ambiguous_audit_steps.update(ambiguous)
        for row in generations:
            prior = row.get("global_step")
            if type(prior) is int and prior+1 in steps and prior+1 not in ambiguous:
                status_counts[row.get("status", "unknown")] += 1
                availability = row.get("reward_availability")
                if isinstance(availability, dict):
                    for metric, available in availability.items():
                        (reward_available if available is True else reward_unavailable)[metric] += 1
                    scored = any(value is True for value in availability.values())
                else:
                    # Historical v1/v2 journals only wrote keys after successful grading.
                    scored = row.get("status") == "complete" and len(row.get("judge_cache_keys", [])) == 4 and all(row["judge_cache_keys"])
                if scored:
                    judge_scored += 1
                    complete_judge_scored += int(row.get("status") == "complete")
                for failure in row.get("judge_failures", []):
                    judge_failure_counts[failure.get("error_type", "unknown")] += 1
        for row in advantages:
            prior = row.get("global_step")
            if type(prior) is not int or prior+1 not in steps or prior+1 in ambiguous:
                continue
            signal_counts["records"] += 1
            signal_counts["active_records"] += int(row.get("active") is True)
            signal_counts["nonzero_policy_signal_records"] += int(row.get("active") is True and any(
                _finite(segment.get("advantage")) and segment["advantage"] != 0
                for segment in row.get("advantage_segments", [])))
    metrics = read_rows(root / "metrics.jsonl")
    completed_metrics = [row for row in metrics if type(row.get("global_step")) is int
                         and row["global_step"] in steps]
    losses = [row["loss"] for row in completed_metrics if "loss" in row]
    finite_losses = [value for value in losses if _finite(value)]
    loss_steps = {row["global_step"] for row in completed_metrics if "loss" in row and _finite(row["loss"])}
    nonfinite_losses = sum(("loss" in row and not _finite(row["loss"]))
                          or "loss" in row.get("nonfinite_metric_keys", []) for row in completed_metrics)
    gradients = [row.get("grad_norm") for row in completed_metrics if "grad_norm" in row]
    finite_gradients = [value for value in gradients if _finite(value)]
    gradient_steps = {row["global_step"] for row in completed_metrics
                      if "grad_norm" in row and _finite(row["grad_norm"])}
    nonfinite_gradients = sum(("grad_norm" in row and not _finite(row["grad_norm"]))
                             or "grad_norm" in row.get("nonfinite_metric_keys", []) for row in completed_metrics)

    effective_signal_steps = {row["global_step"] for row in completed_metrics
                              if _finite(row.get("g/effective_policy_signal_fraction"))
                              and 0 <= row["g/effective_policy_signal_fraction"] <= 1}
    effective_signal_observed = any(row.get("g/effective_policy_signal_fraction", 0) > 0
                                    for row in completed_metrics
                                    if _finite(row.get("g/effective_policy_signal_fraction")))
    first_effective_signal_ok = (len(first_rollouts) == world_size and all(
        row.get("passed") is True and type(row.get("effective_policy_signal_sequences")) is int
        and row["effective_policy_signal_sequences"] > 0 for row in first_rollouts))

    marker_ok = (completed.get("run_scope") == "bounded_updates"
                 and completed.get("global_step") == expected
                 and completed.get("protocol_hash") == protocol.get("protocol_hash")
                 and bool(protocol.get("protocol_hash")))
    checkpoint = root / "checkpoints" / f"checkpoint-{expected}"
    checkpoint_marker = read_object(checkpoint / "checkpoint_complete.json")
    checkpoint_ok = bool(protocol.get("protocol_hash")) and checkpoint_is_complete(
        checkpoint, protocol_hash=protocol["protocol_hash"], world_size=world_size)
    checkpoint_ok = checkpoint_ok and checkpoint_marker.get("global_step") == expected
    contiguous_complete = steps == list(range(1, expected+1))
    checks = {
        "first_update_completed_on_all_ranks": 1 in steps,
        "all_planned_updates_completed": contiguous_complete,
        "training_completion_marker_valid": bool(marker_ok),
        "final_resumable_checkpoint_verified": checkpoint_ok,
        "launcher_exit_zero": launcher.get("returncode") == 0,
        "finite_loss_observed": bool(steps) and loss_steps == set(steps) and not nonfinite_losses,
        "nonzero_gradient_observed": any(value > 0 for value in finite_gradients)
                                      and gradient_steps == set(steps) and not nonfinite_gradients,
        "jev_scoring_observed": judge_scored > 0,
        "nonzero_policy_advantages_observed": signal_counts["nonzero_policy_signal_records"] > 0,
        "first_rollout_effective_signal_verified": first_effective_signal_ok,
        "effective_policy_signal_observed": bool(steps) and effective_signal_steps == set(steps)
                                            and effective_signal_observed,
        "complete_generation_and_advantage_audits": bool(audit_counts) and all(row["complete"] for row in audit_counts),
        "unambiguous_journals": not (duplicate_steps or invalid_steps or ambiguous_audit_steps or errors),
    }
    success = all(checks.values())
    if success:
        state = "completed"
    elif contiguous_complete:
        state = "completed_with_failed_checks"
    elif steps:
        state = "partial"
    elif launcher or errors or any(row["events"] for row in startup):
        state = "startup_failed_or_incomplete"
    else:
        state = "not_started"
    projection = None
    withheld = []
    if len(measured) < minimum:
        withheld.append("insufficient_warmed_updates")
    if epoch_steps is None:
        withheld.append("missing_full_epoch_update_count")
    if not checks["unambiguous_journals"]:
        withheld.append("ambiguous_or_invalid_journals")
    for check in ("finite_loss_observed", "nonzero_policy_advantages_observed",
                  "nonzero_gradient_observed", "jev_scoring_observed",
                  "complete_generation_and_advantage_audits"):
        if not checks[check]:
            withheld.append(check+"_not_verified")
    projection_ready = not withheld
    if projection_ready:
        mean = statistics.mean(walls)
        projection = {
            "provisional": True, "basis": "measured optimizer updates; excludes setup, checkpoints, evaluation and export",
            "reward_coverage_matches_observed_pilot_only": True,
            "epoch_updates": epoch_steps, "training_seconds": mean*epoch_steps,
            "training_hours": mean*epoch_steps/3600,
            "observed_update_range_projection_hours": [min(walls)*epoch_steps/3600, max(walls)*epoch_steps/3600],
            "range_is_confidence_interval": False,
        }
        rate = launcher.get("hourly_usd", settings.get("hourly_rate_usd"))
        if _finite(rate, positive=True):
            projection["assumed_cluster_hourly_usd"] = rate
            projection["training_rental_usd"] = projection["training_hours"]*rate
    report = {
        "format": "science-g-pilot-report-v1", "status": state, "success": success,
        "world_size": world_size, "planned_updates": expected,
        "completed_updates_all_ranks": steps, "warmup_updates_excluded": warm_steps,
        "measured_updates": measured_steps, "minimum_measured_updates_for_projection": minimum,
        "missing_timing_ranks": missing_ranks, "partial_rank_updates_excluded": sorted(all_seen-shared),
        "duplicate_steps_excluded": sorted(duplicate_steps), "invalid_timing_steps_excluded": sorted(invalid_steps),
        "ambiguous_generation_steps_excluded_from_quality": sorted(ambiguous_audit_steps),
        "update_seconds": _statistics(walls), "completed_update_details": updates,
        "measured_generated_tokens": total_tokens,
        "end_to_end_generated_tokens_per_second": total_tokens/total_seconds if total_seconds else None,
        "stage_seconds": stages, "gpu_peak_memory": memory,
        "startup_by_rank": startup, "completion_status_counts": dict(status_counts),
        "first_rollout_signal_checks": first_rollouts,
        "startup_seconds_to_training": max(row["seconds_to_first_update"] for row in startup)
            if all(row["seconds_to_first_update"] is not None for row in startup) else None,
        "startup_seconds_to_first_completed_update": max(row["seconds_to_first_update_completed"] for row in startup)
            if all(row["seconds_to_first_update_completed"] is not None for row in startup) else None,
        "judge_scored_completions": judge_scored, "judge_scored_complete_answers": complete_judge_scored,
        "reward_availability_counts": dict(reward_available), "reward_unavailability_counts": dict(reward_unavailable),
        "judge_failure_counts": dict(judge_failure_counts), "advantage_diagnostics": dict(signal_counts),
        "rank_update_audit_counts": audit_counts,
        "loss_diagnostics": {"finite_observations": len(finite_losses), "nonfinite_observations": nonfinite_losses,
                             "observed_steps": sorted(loss_steps), "missing_steps": sorted(set(steps)-loss_steps),
                             "observed_loss": _statistics(finite_losses)},
        "gradient_diagnostics": {"finite_observations": len(finite_gradients),
                                 "nonfinite_observations": nonfinite_gradients,
                                 "observed_steps": sorted(gradient_steps), "missing_steps": sorted(set(steps)-gradient_steps),
                                 "observed_grad_norm": _statistics(finite_gradients)},
        "checks": checks, "projection": projection, "projection_withheld_reasons": withheld,
        "launcher": launcher or None,
        "preflight": {key: preflight[key] for key in ("success", "status", "stage", "http_status_code", "error_type", "error", "elapsed_seconds")
                      if key in preflight} or None,
        "journal_errors": errors,
        "limitations": [
            "This checks execution and estimates time; it does not evaluate scientific improvement or calibrate rewards.",
            "Per-stage rank maxima are diagnostic and are not additive across stages.",
            "PyTorch peak allocation may omit vLLM/device allocations.",
            "Generation length and judge latency can change full-run throughput; observed ranges are not confidence intervals.",
            "Partial reward coverage can reduce judge workload; timing estimates apply to observed coverage only.",
            "Workload exit does not stop Vast compute or storage billing.",
        ],
    }
    return _safe(report)


def write_pilot_report(run_dir, cfg=None, world_size=4):
    """Write compact machine-readable and human-readable reports, even on failure."""
    root = Path(run_dir)
    report = summarize_pilot(root, cfg, world_size)
    atomic_json(root / "pilot_report.json", report)
    seconds = report["update_seconds"]
    lines = ["# Experiment G pilot report", "", f"Status: **{report['status']}**", "",
             f"Completed on every rank: {len(report['completed_updates_all_ranks'])}/{report['planned_updates']} updates.",
             f"Warmup excluded: {report['warmup_updates_excluded']}. Measured updates: {report['measured_updates']}.", ""]
    if report["startup_seconds_to_first_completed_update"] is not None:
        lines += [f"Slowest-rank time from worker start to its first completed update: {report['startup_seconds_to_first_completed_update']:.2f} seconds.",
                  "This includes worker startup/model preparation; launcher preflight and package installation are separate.", ""]
    launcher = report["launcher"] or {}
    if launcher:
        lines += [f"Launcher status: `{launcher.get('status', 'unknown')}`; phase: `{launcher.get('phase', 'unknown')}`; exit code: `{launcher.get('returncode')}`."]
        if launcher.get("timeout_kind"):
            lines += [f"Timeout: `{launcher['timeout_kind']}`."]
        if launcher.get("error_type"):
            lines += [f"Launcher error type: `{launcher['error_type']}`."]
        lines += [""]
    preflight = report["preflight"] or {}
    if preflight:
        lines += [f"Preflight: `{preflight.get('status', 'unknown')}` at stage `{preflight.get('stage', 'unknown')}`."]
        if preflight.get("http_status_code") is not None:
            lines += [f"Service HTTP status: `{preflight['http_status_code']}`."]
        if preflight.get("error_type"):
            lines += [f"Preflight error type: `{preflight['error_type']}`."]
        if preflight.get("error"):
            lines += [f"Preflight detail: {preflight['error']}"]
        lines += [""]
    for diagnostic in report["first_rollout_signal_checks"]:
        if diagnostic.get("passed") is False:
            lines += [f"Rank {diagnostic['rank']} stopped before backward: `{diagnostic.get('reason')}`; "
                      f"{diagnostic.get('complete_completions', 0)}/{diagnostic.get('completion_records', 0)} complete thinking/final outputs, "
                      f"{diagnostic.get('scoreable_completions', diagnostic.get('scored_complete_completions', 0))} with available rewards, "
                      f"{diagnostic.get('nonzero_policy_signal_sequences', 0)} sequences with nonzero policy advantages.", ""]
    lines += ["| Rank | Last startup event | Worker elapsed seconds |", "|---|---|---|"]
    for rank in report["startup_by_rank"]:
        lines += [f"| {rank['rank']} | {rank['last_event'] or 'not recorded'} | {rank['last_event_elapsed_seconds'] if rank['last_event_elapsed_seconds'] is not None else 'not recorded'} |"]
    lines += [""]
    if _finite(launcher.get("wall_seconds")):
        lines += [f"Observed launcher wall time: {launcher['wall_seconds']:.2f} seconds.", ""]
    if seconds:
        lines += [f"Update seconds: mean {seconds['mean']:.2f}; median {seconds['median']:.2f}; observed range {seconds['min']:.2f}–{seconds['max']:.2f}.",
                  f"End-to-end generated throughput: {report['end_to_end_generated_tokens_per_second']:.2f} tokens/second across all GPUs.", ""]
    if report["projection"]:
        p = report["projection"]
        lines += [f"Provisional full-epoch training estimate: **{p['training_hours']:.2f} hours** for {p['epoch_updates']} updates.",
                  "This excludes setup, checkpoints, evaluation and export. The estimate is not a runtime guarantee.", ""]
        if "training_rental_usd" in p:
            lines += [f"At the supplied ${p['assumed_cluster_hourly_usd']:.3f}/hour cluster rate: about ${p['training_rental_usd']:.2f} for projected training only.", ""]
    else:
        lines += ["Full-run projection withheld: " + "; ".join(reason.replace('_', ' ') for reason in report["projection_withheld_reasons"]) + ".", ""]
    lines += ["| Execution check | Result |", "|---|---|"]
    lines += [f"| {key.replace('_',' ')} | {'PASS' if value else 'NOT VERIFIED'} |" for key,value in report["checks"].items()]
    lines += ["", "Completion statuses: `"+json.dumps(report["completion_status_counts"],sort_keys=True)+"`.",
              "", "Available rewards: `"+json.dumps(report["reward_availability_counts"],sort_keys=True)+"`.",
              "Unavailable rewards: `"+json.dumps(report["reward_unavailability_counts"],sort_keys=True)+"`.",
              "Judge failures: `"+json.dumps(report["judge_failure_counts"],sort_keys=True)+"`.",
              "", "Stop/delete the Vast instance separately to end rental charges.", ""]
    (root / "pilot_report.md").write_text("\n".join(lines))
    return report
