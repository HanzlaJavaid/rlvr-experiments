"""Supplemental correctness scoring for responses with repeated closing delimiters."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

from science_gspo_g.io import atomic_json
from science_gspo_g.judge import JevEngine, score_parsed_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--credential-file", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir
    out = run / "paired_eval_4_final171"
    # Consume the temporary credential without sending its value to the terminal.
    try:
        name, _, secret = args.credential_file.read_text().strip().partition("=")
        assert name == "TYPESAFE_API_KEY" and secret
        os.environ[name] = secret
    finally:
        args.credential_file.unlink(missing_ok=True)

    from transformers import AutoTokenizer

    assets = json.loads((run / "assets.json").read_text())
    cfg = json.loads((run / "resolved_config.json").read_text())
    cfg["judge"]["concurrency_per_rank"] = cfg["judge"]["global_max_concurrency"]
    protocol = json.loads((run / "protocol.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model_snapshot"], local_files_only=True)
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    rows = json.loads((out / "selected_questions.json").read_text())
    strict = json.loads((out / "eval_records.json").read_text())
    candidates = []
    scores = {}
    metric = "scientific_correctness"
    for record in strict:
        index, arm = record["selection_index"], record["arm"]
        value = record["scores"][metric]
        if value is not None:
            scores[index, arm] = {"score": value, "source": "strict"}
            continue
        parsed = record["parsed"]
        assert parsed["parse_error"] == "native_thinking_boundary"
        assert parsed["ended_with_eos"] and parsed["native_boundary_count"] > 1
        ids = parsed["token_ids"]
        boundary = max(i for i, token in enumerate(ids) if token == close_id)
        answer_ids = ids[boundary + 1:-1]
        assert answer_ids and not set(answer_ids).intersection(tokenizer.all_special_ids)
        answer = tokenizer.decode(answer_ids, clean_up_tokenization_spaces=False).strip()
        assert answer
        candidates.append((index, arm, answer, parsed["native_boundary_count"]))

    random.Random(456).shuffle(candidates)
    with JevEngine(cfg, out / "jev_cache.jsonl") as engine:
        evaluated, stats = score_parsed_rows(
            [rows[index] for index, _, _, _ in candidates],
            [
                {"status": "complete", "final_answer": answer,
                 "section_availability": {"answer": True, "model": False, "apply": False, "check": False}}
                for _, _, answer, _ in candidates
            ],
            engine,
            protocol["protocol_hash"],
        )
    assert stats["failed_requests"] == 0
    for (index, arm, answer, count), record in zip(candidates, evaluated, strict=True):
        value = record["scores"][metric]
        assert value is not None
        scores[index, arm] = {
            "score": value,
            "source": "after_last_native_closing_delimiter",
            "native_closing_delimiter_count": count,
            "final_answer": answer,
        }

    pairs = []
    for index, row in enumerate(rows):
        base, trained = scores[index, "base"], scores[index, "checkpoint_171"]
        pairs.append({
            "example_id": row["example_id"], "category": row["category"],
            "base": base, "trained": trained, "delta": trained["score"] - base["score"],
        })
    result = {
        "format": "science-g-supplemental-visible-answer-correctness-v1",
        "evaluator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "Reused strict scores where available. For duplicate closing delimiters, "
                  "scored the final text after the last native closing token with the same rubric. "
                  "Excluded the terminating EOS. Generated no additional responses.",
        "process_scores": "Strict parser results only",
        "mean_delta": sum(pair["delta"] for pair in pairs) / len(pairs),
        "pairs": pairs,
    }
    atomic_json(out / "supplemental_correctness.json", result)
    for pair in pairs:
        print(f"{pair['category']}: {pair['base']['score']:.6f} -> {pair['trained']['score']:.6f}")
    print(f"Mean paired correctness change: {result['mean_delta']:+.6f}")


if __name__ == "__main__":
    main()
