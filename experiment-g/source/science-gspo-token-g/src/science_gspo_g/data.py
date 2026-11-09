"""Prepare the complete science subset without conversation leakage.

Only ``prepare_data`` and ``load_prepared`` import Hugging Face dependencies.
The transformation helpers intentionally work on ordinary Python dictionaries.
Call preparation on the main distributed process, then synchronize workers before
loading the saved DatasetDict. Worker count does not affect split membership,
eligibility, or the number of repeated examples used for sampler alignment.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import tempfile
import unicodedata
from typing import Any, Iterable, Mapping

from .io import append_jsonl, atomic_json, digest


DATA_FORMAT_VERSION = "science-g-qwen35-native-trace-v1"
RESPONSE_LAYOUT = "native_thinking_then_plain_final_answer_v1"
SPLIT_NAMES = ("train", "dev", "test")


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving scientifically meaningful case."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def prompt_digest(prompt: list[dict[str, str]]) -> str:
    return digest([
        {"role": message["role"], "content": normalize_text(message["content"])}
        for message in prompt
    ])


def extract_final_answer(text: str) -> str:
    """Strip a leading reference thinking block, rejecting ambiguous formats."""
    if not isinstance(text, str):
        raise ValueError("Assistant content is not text")
    text = text.strip()
    if text.startswith("<think>"):
        if "</think>" not in text:
            raise ValueError("Unclosed reference thinking block")
        text = text.split("</think>", 1)[1].strip()
    if not text or "<think>" in text or "</think>" in text:
        raise ValueError("Empty reference answer or unexpected thinking tags")
    return text


def conversation_to_samples(conversation: Mapping[str, Any], source_index: int) -> list[dict]:
    """Expand assistant targets using preceding messages only.

    A malformed turn rejects the entire conversation at the caller, so partially
    parsed conversations cannot silently contribute an incomplete target set.
    Earlier assistant messages contain their final answers, never their traces.
    """
    history: list[dict[str, str]] = []
    output: list[dict] = []
    conversation_id = f"conv_{source_index}"
    for message_index, message in enumerate(conversation["messages"]):
        role, content = message["role"], message["content"]
        if not isinstance(content, str):
            raise ValueError("Non-text conversation message")
        if role in {"system", "user"}:
            history.append({"role": role, "content": content})
        elif role == "assistant":
            if not history or history[-1]["role"] != "user" or not history[-1]["content"].strip():
                raise ValueError("Assistant target must follow a nonempty user message")
            reference = extract_final_answer(content)
            prompt = deepcopy(history)
            output.append({
                "conversation_id": conversation_id,
                "example_id": f"{conversation_id}_message_{message_index}",
                "category": conversation["category"],
                "prompt": prompt,
                "prompt_hash": prompt_digest(prompt),
                "user_question": history[-1]["content"],
                "conversation_context": deepcopy(history[:-1]),
                "reference_answer": reference,
            })
            history.append({"role": "assistant", "content": reference})
        else:
            raise ValueError(f"Unsupported message role: {role}")
    return output


def group_and_deduplicate(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Connect conversations sharing any cleaned prompt, then keep first targets.

    Grouping occurs before deduplication. This preserves transitive connections
    even when the shared target itself is removed as a duplicate.
    """
    rows = [deepcopy(row) for row in rows]
    parent = {row["conversation_id"]: row["conversation_id"] for row in rows}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    owners: dict[str, str] = {}
    for row in rows:
        previous = owners.setdefault(row["prompt_hash"], row["conversation_id"])
        left, right = find(previous), find(row["conversation_id"])
        if left != right:
            parent[max(left, right)] = min(left, right)

    unique, duplicates, seen = [], [], {}
    for row in rows:
        row["leakage_group"] = find(row["conversation_id"])
        previous = seen.get(row["prompt_hash"])
        if previous is not None:
            duplicates.append({
                "dropped": row["example_id"],
                "kept": previous["example_id"],
                "reference_conflict": (
                    normalize_text(row["reference_answer"])
                    != normalize_text(previous["reference_answer"])
                ),
            })
        else:
            seen[row["prompt_hash"]] = row
            unique.append(row)
    return unique, duplicates


def split_grouped_rows(
    rows: list[dict], seed: int = 42, train_fraction: float = 0.75,
    dev_fraction: float = 0.10,
) -> dict[str, list[dict]]:
    """Match the notebook's seeded 75/10/15 conversation-group assignments."""
    if not (0 < train_fraction < 1 and 0 < dev_fraction < 1
            and train_fraction + dev_fraction < 1):
        raise ValueError("Train/dev fractions must be positive and leave a positive test fraction")
    groups = sorted({row["leakage_group"] for row in rows})
    if len(groups) < 20:
        raise ValueError("Too few independent conversation groups: require at least 20")
    random.Random(seed).shuffle(groups)
    train_end = round(train_fraction * len(groups))
    dev_end = round((train_fraction + dev_fraction) * len(groups))
    assignments = {
        group: ("train" if index < train_end else "dev" if index < dev_end else "test")
        for index, group in enumerate(groups)
    }
    splits = {
        name: [row for row in rows if assignments[row["leakage_group"]] == name]
        for name in SPLIT_NAMES
    }
    validate_splits(splits)
    return splits


def validate_splits(splits: Mapping[str, Iterable[dict]], *, allow_padding: bool = False) -> None:
    """Check original examples for leakage; verify explicit training repeats."""
    seen_groups: set[str] = set()
    seen_prompts: set[str] = set()
    seen_examples: set[str] = set()
    for name in SPLIT_NAMES:
        rows = list(splits[name])
        padding = [row for row in rows if row.get("epoch_padding", False)]
        if padding and (not allow_padding or name != "train"):
            raise ValueError("Padding is allowed only in the prepared training split")
        originals = [row for row in rows if not row.get("epoch_padding", False)]
        if not originals:
            raise ValueError(f"Empty {name} split")
        groups = {row["leakage_group"] for row in originals}
        prompts = {row["prompt_hash"] for row in originals}
        examples = {row["example_id"] for row in originals}
        if len(examples) != len(originals) or len(prompts) != len(originals):
            raise ValueError("Duplicate original examples or prompts within a split")
        if not seen_groups.isdisjoint(groups):
            raise ValueError("Conversation group leakage")
        if not seen_prompts.isdisjoint(prompts):
            raise ValueError("Prompt leakage")
        if not seen_examples.isdisjoint(examples):
            raise ValueError("Example leakage")
        seen_groups.update(groups)
        seen_prompts.update(prompts)
        seen_examples.update(examples)
        original_by_id = {row["example_id"]: row for row in originals}
        for row in rows:
            if row["prompt"][-1]["role"] != "user" or row["prompt"][-1]["content"] != row["user_question"]:
                raise ValueError("Prompt must end with the target user question")
            if not row["reference_answer"].strip() or any(
                token in row["reference_answer"] for token in ("<think>", "</think>")
            ):
                raise ValueError("Reference contains an empty answer or thinking tags")
            for message in row["prompt"]:
                if message["role"] == "assistant" and any(
                    token in message["content"] for token in ("<think>", "</think>")
                ):
                    raise ValueError("Historical assistant thinking leaked into a prompt")
        for row in padding:
            original = original_by_id.get(row["example_id"])
            candidate = {**row, "epoch_padding": False}
            if original is None or candidate != {**original, "epoch_padding": False}:
                raise ValueError("A padded row must be an exact repeat of an original training row")


def measure_lengths(row: Mapping[str, Any], tokenizer: Any) -> dict[str, int]:
    # Transformers 5.20 defaults to a BatchEncoding, whose len counts fields.
    prompt_ids = tokenizer.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False,
        enable_thinking=True,
    )
    if not isinstance(prompt_ids, list) or not all(isinstance(token, int) for token in prompt_ids):
        raise TypeError("The tokenizer must return a flat list of prompt token IDs")
    reference_ids = tokenizer(row["reference_answer"], add_special_tokens=False)["input_ids"]
    return {"prompt_tokens": len(prompt_ids), "reference_tokens": len(reference_ids)}


def apply_subject_scaffolds(splits: Mapping[str, Iterable[dict]]) -> dict[str, list[dict]]:
    """Attach policy instructions after source-prompt deduplication and splitting.

    References never enter the policy prompt. Source system messages remain in
    the original judge context as fallible data but do not override G's policy.
    """
    from .prompts import build_system_prompt, PROMPT_VERSION
    result = {}
    for name in SPLIT_NAMES:
        result[name] = []
        for original in splits[name]:
            row = deepcopy(original)
            row["source_prompt_hash"] = row["prompt_hash"]
            row["prompt"] = [{"role": "system", "content": build_system_prompt(row["category"])}] + [
                message for message in row["prompt"] if message["role"] != "system"
            ]
            row["prompt_hash"] = prompt_digest(row["prompt"])
            row["scaffold_version"] = PROMPT_VERSION
            result[name].append(row)
    validate_splits(result)
    return result


def filter_by_lengths(
    splits: Mapping[str, Iterable[dict]], tokenizer: Any,
    max_prompt_tokens: int, max_reference_tokens: int,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Exclude entire overlong examples; never truncate scientific context."""
    eligible: dict[str, list[dict]] = {}
    excluded: list[dict] = []
    for name in SPLIT_NAMES:
        eligible[name] = []
        for original in splits[name]:
            row = {**original, **measure_lengths(original, tokenizer)}
            reasons = []
            if row["prompt_tokens"] > max_prompt_tokens:
                reasons.append("prompt_too_long")
            if row["reference_tokens"] > max_reference_tokens:
                reasons.append("reference_too_long")
            if reasons:
                excluded.append({
                    "split": name, "example_id": row["example_id"],
                    "prompt_tokens": row["prompt_tokens"],
                    "reference_tokens": row["reference_tokens"], "reasons": reasons,
                })
            else:
                eligible[name].append(row)
    validate_splits(eligible)
    return eligible, excluded


def pad_training_rows(
    rows: list[dict], global_batch_size: int, num_generations: int,
) -> tuple[list[dict], list[str]]:
    """Align one full epoch to complete global GRPO batches.

    ``global_batch_size`` counts completions, not questions. This calculation
    deliberately has no world-size parameter: 2, 4, 8, and 16 workers see the
    same examples when the global batch is held fixed.
    """
    if global_batch_size <= 0 or num_generations <= 0 or global_batch_size % num_generations:
        raise ValueError("Global completion batch must be a positive multiple of num_generations")
    if not rows:
        raise ValueError("Cannot pad an empty training split")
    questions_per_update = global_batch_size // num_generations
    padding_count = (-len(rows)) % questions_per_update
    output = [{**deepcopy(row), "epoch_padding": False} for row in rows]
    padding_ids = []
    for index in range(padding_count):
        original = rows[index % len(rows)]
        output.append({**deepcopy(original), "epoch_padding": True})
        padding_ids.append(original["example_id"])
    return output, padding_ids


def _policy(cfg: dict, tokenizer: Any, model_revision: str) -> dict:
    from .prompts import prompt_registry
    data, model, training = cfg["data"], cfg["model"], cfg["training"]
    return {
        "format": DATA_FORMAT_VERSION,
        "dataset": data["dataset"], "file": data.get("file", "full_train.jsonl"),
        "requested_dataset_revision": data.get("revision"),
        "categories": sorted(data["categories"]),
        "split_seed": int(data.get("split_seed", 42)),
        "train_fraction": float(data.get("train_fraction", 0.75)),
        "dev_fraction": float(data.get("dev_fraction", 0.10)),
        "model": model["name"], "model_revision": model_revision,
        "chat_template_hash": digest(getattr(tokenizer, "chat_template", None)),
        "chat_template_kwargs": {"enable_thinking": True},
        "model_loader": model.get("loader", "qwen3_5_text"),
        "language_model_only": model.get("language_model_only", True),
        "max_prompt_tokens": int(model["max_prompt_tokens"]),
        "max_reference_tokens": int(model["max_reference_tokens"]),
        "global_batch_size": int(training["global_batch_size"]),
        "num_generations": int(training["num_generations"]),
        "prompt_registry_hash": digest(prompt_registry()),
        "response_layout": RESPONSE_LAYOUT,
        "source_system_policy": "replace in policy input; preserve as fallible judge context",
    }


def load_prepared(output_dir: str | Path):
    """Load and verify a completed preparation; returns a Hugging Face DatasetDict."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != DATA_FORMAT_VERSION:
        raise ValueError("Unsupported prepared-data format. Prepare the current scaffold in a new data directory.")
    from .prompts import PROMPT_VERSION, prompt_registry
    policy = manifest.get("policy", {})
    if (policy.get("format") != DATA_FORMAT_VERSION
            or policy.get("response_layout") != RESPONSE_LAYOUT
            or policy.get("prompt_registry_hash") != digest(prompt_registry())):
        raise ValueError("Prepared-data scaffold policy changed. Prepare the current instructions in a new data directory.")
    from datasets import load_from_disk
    splits = load_from_disk(str(output_dir / "datasets"))
    validate_splits(splits, allow_padding=True)
    for name in SPLIT_NAMES:
        if any(row.get("scaffold_version") != PROMPT_VERSION for row in splits[name]):
            raise ValueError(f"Prepared {name} rows belong to a different scaffold version")
        if digest(list(splits[name])) != manifest["split_digests"][name]:
            raise ValueError(f"Prepared {name} data changed on disk")
    return splits


def prepare_data(
    cfg: dict, output_dir: str | Path, tokenizer: Any, model_revision: str,
) -> dict:
    """Pin, prepare, audit, and atomically publish the full eligible dataset.

    Call once on rank zero. A completed matching directory is reused and checked.
    Interrupted preparation creates only a temporary sibling directory and never
    publishes a half-written dataset. A new data policy requires a new directory.
    """
    from datasets import Dataset, DatasetDict, load_dataset
    from huggingface_hub import HfApi

    output_dir = Path(output_dir).expanduser().resolve()
    policy = _policy(cfg, tokenizer, model_revision)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["policy"] != policy:
            raise ValueError("Prepared-data policy changed. Choose a new run/data directory.")
        load_prepared(output_dir)
        return manifest
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Incomplete or unrelated data directory: {output_dir}")

    requested_revision = cfg["data"].get("revision")
    source_revision = HfApi().dataset_info(
        policy["dataset"], revision=requested_revision,
    ).sha
    if not source_revision:
        raise RuntimeError("Could not resolve an immutable dataset revision")
    raw = load_dataset(
        policy["dataset"], data_files=policy["file"], split="train", revision=source_revision,
    )
    rows, rejected = [], []
    selected_conversations = 0
    for index, conversation in enumerate(raw):
        if conversation.get("category") not in policy["categories"]:
            continue
        selected_conversations += 1
        try:
            samples = conversation_to_samples(conversation, index)
            if not samples:
                raise ValueError("Conversation contains no assistant target")
            rows.extend(samples)
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"conversation_id": f"conv_{index}", "reason": str(error)})

    unique, duplicates = group_and_deduplicate(rows)
    original_splits = split_grouped_rows(
        unique, policy["split_seed"], policy["train_fraction"], policy["dev_fraction"],
    )
    scaffolded_splits = apply_subject_scaffolds(original_splits)
    eligible, excluded = filter_by_lengths(
        scaffolded_splits, tokenizer, policy["max_prompt_tokens"], policy["max_reference_tokens"],
    )
    eligible_counts = {name: len(split) for name, split in eligible.items()}
    category_counts = {name: {category: sum(row["category"] == category for row in split)
                             for category in policy["categories"]} for name, split in eligible.items()}
    eligible_ids = {name: digest([row["example_id"] for row in split]) for name, split in eligible.items()}
    eligible["train"], padding_ids = pad_training_rows(
        eligible["train"], policy["global_batch_size"], policy["num_generations"],
    )
    for name in ("dev", "test"):
        eligible[name] = [{**row, "epoch_padding": False} for row in eligible[name]]
    validate_splits(eligible, allow_padding=True)
    splits = DatasetDict({name: Dataset.from_list(eligible[name]) for name in SPLIT_NAMES})
    # Hash the Arrow round-trip representation, which can normalize scalar types.
    manifest = {
        "format": DATA_FORMAT_VERSION, "policy": policy,
        "source_revision": source_revision, "source_rows": len(raw),
        "selected_conversations": selected_conversations,
        "target_rows_before_dedup": len(rows),
        "rejected_conversations": len(rejected),
        "duplicate_targets_removed": len(duplicates),
        "conflicting_duplicate_references": sum(row["reference_conflict"] for row in duplicates),
        "split_sizes_before_eligibility": {name: len(split) for name, split in original_splits.items()},
        "eligible_split_sizes": eligible_counts, "eligible_split_ids": eligible_ids,
        "eligible_category_counts": category_counts,
        "excluded_examples": len(excluded),
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "split_digests": {name: digest(list(split)) for name, split in splits.items()},
        "training_padding_ids": padding_ids,
        "questions_per_update": policy["global_batch_size"] // policy["num_generations"],
        "optimizer_steps_per_epoch": len(splits["train"]) * policy["num_generations"] // policy["global_batch_size"],
        "training_completions_per_epoch": len(splits["train"]) * policy["num_generations"],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-preparing-", dir=output_dir.parent))
    try:
        splits.save_to_disk(str(staging / "datasets"))
        for filename, records in (
            ("exclusions.jsonl", excluded),
            ("duplicate_references.jsonl", duplicates),
            ("rejected_conversations.jsonl", rejected),
        ):
            (staging / filename).touch()
            append_jsonl(staging / filename, records)
        atomic_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            output_dir.rmdir()  # Only an empty directory is permitted above.
        staging.replace(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest
