"""Collect the final G adapter, four-question evaluation, and training diagnostics."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def reward_rows(run):
    by_step = collections.defaultdict(lambda: [0, collections.defaultdict(list), 0])
    ranks = sorted(run.glob("ranks/rank_*/training_generations.jsonl"))
    assert len(ranks) == 4
    for rank in ranks:
        rank_counts = collections.Counter()
        with rank.open() as handle:
            for line in handle:
                record = json.loads(line)
                step = record["global_step"] + 1
                rank_counts[step] += 1
                stats = by_step[step]
                stats[0] += 1
                for name, value in record["scores"].items():
                    scores = stats[1][name]
                    if value is not None:
                        scores.append(value)
                stats[2] += len(record["judge_failures"])
        assert rank_counts == {step: 16 for step in range(1, 172)}
    return [
        {
            "step": step, "completions": total, "judge_failures": failures,
            "rewards": {
                name: {"scored": len(values), "mean": sum(values) / len(values) if values else None}
                for name, values in sorted(rewards.items())
            },
        }
        for step, (total, rewards, failures) in sorted(by_step.items())
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    run, workspace = args.run_dir, args.workspace
    evaluation = run / "paired_eval_4_final171"
    adapter = run / "final_adapter"
    assert (workspace / "science_g_full.exit").read_text().strip() == "0"
    assert (workspace / "science_g_eval_generate.exit").read_text().strip() == "0"
    assert (workspace / "science_g_eval_score.exit").read_text().strip() == "0"
    assert json.loads((run / "training_completed.json").read_text())["global_step"] == 171
    assert json.loads((evaluation / "eval_completed.json").read_text())["success"]
    settings = json.loads((evaluation / "eval_protocol.json").read_text())
    assert digest(adapter / "adapter_model.safetensors") == settings["checkpoint_policy_sha256"]
    rewards = evaluation / "training_rewards_by_step.json"
    rewards.write_text(json.dumps(reward_rows(run), indent=2) + "\n")

    files = {f"adapter/{path.name}": path for path in sorted(adapter.iterdir()) if path.is_file()}
    files.update({f"evaluation/{path.name}": path for path in sorted(evaluation.iterdir()) if path.is_file()})
    for name in ("assets.json", "execution.json", "metrics.jsonl", "prompts.json", "protocol.json",
                 "resolved_config.json", "rubric.json", "training_completed.json", "training_args.json",
                 "checkpoints/trainer_state.json"):
        if (run / name).is_file():
            files[f"training/{name}"] = run / name
    for pattern in ("update_timings.jsonl", "session_*.json", "model_loader_*.json", "sampling_*.json",
                    "weight_sync_*.json", "vllm_engine_*.json"):
        for path in sorted(run.glob(f"ranks/rank_*/{pattern}")):
            files[f"training/{path.relative_to(run)}"] = path
    for name in ("science_g_full_launch.log", "science_g_full.exit", "science_g_kernels.json",
                 "science_g_eval_generate.log", "science_g_eval_generate.exit",
                 "science_g_eval_score.log", "science_g_eval_score.exit",
                 "paired_eval_g4.py", "collect_eval_g4.py"):
        files[f"host/{name}"] = workspace / name
    if (evaluation / "supplemental_correctness.json").is_file():
        for name in ("score_g4_visible_answers.py", "science_g_eval_supplemental.log"):
            files[f"host/{name}"] = workspace / name
    for path in files.values():
        assert path.is_file() and path.stat().st_size > 0, path.name

    manifest = {
        "run_name": run.name,
        "files": [
            {"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
            for name, path in sorted(files.items())
        ],
    }
    archive = workspace / "g_eval4_59881279_results.zip"
    partial = archive.with_suffix(".zip.part")
    with ZipFile(partial, "w", compression=ZIP_DEFLATED, compresslevel=3) as package:
        for name, path in sorted(files.items()):
            package.write(path, name)
        package.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
    with ZipFile(partial) as package:
        assert package.testzip() is None
    partial.replace(archive)
    print(f"{archive.name} bytes={archive.stat().st_size} entries={len(files)} sha256={digest(archive)}")


if __name__ == "__main__":
    main()
