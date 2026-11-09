"""Offline dataset contract checks; no model weights or network access."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from science_gspo_g.data import (
    DATA_FORMAT_VERSION,
    RESPONSE_LAYOUT,
    apply_subject_scaffolds,
    conversation_to_samples,
    extract_final_answer,
    filter_by_lengths,
    group_and_deduplicate,
    measure_lengths,
    load_prepared,
    pad_training_rows,
    prepare_data,
    prompt_digest,
    split_grouped_rows,
    validate_splits,
)
from science_gspo_g.io import digest
from science_gspo_g.prompts import prompt_registry


def conversation(question="Question", answer="Answer", category="biology"):
    return {"category": category, "messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}


def row(index, question=None):
    sample = conversation_to_samples(conversation(question or f"Question {index}"), index)[0]
    sample["leakage_group"] = sample["conversation_id"]
    return sample


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, prompt, *, tokenize, add_generation_prompt, return_dict=True,
                            enable_thinking=True):
        assert enable_thinking is True
        count = 90 if "long" in prompt[-1]["content"] else 12
        ids = list(range(count))
        return {"input_ids": ids, "attention_mask": [1] * count} if return_dict else ids

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": list(range(len(text.split())))}


class FakeColumn:
    """Iterable intentionally unsupported by json.dumps, like a datasets Column."""
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)


class FakeDataset(list):
    @classmethod
    def from_list(cls, rows):
        return cls(deepcopy(rows))

    def __getitem__(self, key):
        if isinstance(key, str):
            return FakeColumn([row[key] for row in self])
        return super().__getitem__(key)


class FakeDatasetDict(dict):
    def save_to_disk(self, path):
        path = Path(path)
        path.mkdir(parents=True)
        (path / "rows.json").write_text(json.dumps(self), encoding="utf-8")


def fake_load_from_disk(path):
    value = json.loads((Path(path) / "rows.json").read_text(encoding="utf-8"))
    return FakeDatasetDict({name: FakeDataset(rows) for name, rows in value.items()})


class DataTests(unittest.TestCase):
    def test_obsolete_preparation_is_rejected_before_dataset_loading(self):
        # An old cached dataset must not be loaded merely because its files are
        # intact: G uses native thinking followed by a plain final answer.
        datasets = types.ModuleType("datasets")
        datasets.load_from_disk = lambda *args: self.fail("Do not load obsolete cached rows")
        current_policy = {"format": DATA_FORMAT_VERSION, "response_layout": RESPONSE_LAYOUT,
                          "prompt_registry_hash": digest(prompt_registry())}
        cases = [
            ({"format": "science-e-subject-scaffold-v1", "policy": current_policy}, "Unsupported prepared-data format"),
            ({"format": "science-f-qwen35-subject-scaffold-v1", "policy": current_policy}, "Unsupported prepared-data format"),
            ({"format": DATA_FORMAT_VERSION, "policy": {**current_policy, "response_layout": "native_thinking_then_visible_model_apply_check_answer_v2"}}, "scaffold policy changed"),
            ({"format": DATA_FORMAT_VERSION, "policy": {**current_policy, "response_layout": "tags_inside_thinking_v1"}}, "scaffold policy changed"),
            ({"format": DATA_FORMAT_VERSION, "policy": {**current_policy, "prompt_registry_hash": "old-instructions"}}, "scaffold policy changed"),
        ]
        with tempfile.TemporaryDirectory() as temporary, patch.dict("sys.modules", {"datasets": datasets}):
            destination = Path(temporary)
            for manifest, reason in cases:
                with self.subTest(reason=reason, manifest=manifest):
                    (destination / "manifest.json").write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, reason):
                        load_prepared(destination)

    def test_prepared_rows_must_have_current_scaffold_version(self):
        splits = apply_subject_scaffolds({"train": [row(0)], "dev": [row(1)], "test": [row(2)]})
        splits["train"][0]["scaffold_version"] = "science-gspo-g-scaffolds-v1-obsolete"
        datasets = types.ModuleType("datasets")
        datasets.load_from_disk = lambda *args: FakeDatasetDict(splits)
        # Even a self-consistent row digest cannot relabel old scaffold rows as
        # the current preparation policy.
        manifest = {"format": DATA_FORMAT_VERSION,
                    "policy": {"format": DATA_FORMAT_VERSION, "response_layout": RESPONSE_LAYOUT,
                               "prompt_registry_hash": digest(prompt_registry())},
                    "split_digests": {name: digest(rows) for name, rows in splits.items()}}
        with tempfile.TemporaryDirectory() as temporary, patch.dict("sys.modules", {"datasets": datasets}):
            destination = Path(temporary)
            (destination / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "different scaffold version"):
                load_prepared(destination)

    def test_targets_do_not_see_future_or_reference_trace(self):
        value = conversation("first", "<think>private reference thought</think>first answer")
        value["messages"].extend([
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "<think>second private thought</think>second answer"},
        ])
        first, second = conversation_to_samples(value, 11)
        self.assertEqual(first["prompt"], [{"role": "user", "content": "first"}])
        self.assertEqual(first["reference_answer"], "first answer")
        self.assertEqual(second["prompt"][1]["content"], "first answer")
        self.assertEqual(second["conversation_context"], second["prompt"][:-1])
        self.assertNotIn("private", str(second["prompt"]))
        self.assertNotIn("second answer", str(second["prompt"]))

    def test_malformed_reference_is_rejected(self):
        for text in ["<think>unfinished", "<think>x</think>", "answer </think> bad", ""]:
            with self.assertRaises(ValueError):
                extract_final_answer(text)

    def test_transitive_groups_survive_deduplication(self):
        # c0 -- shared A -- c1 -- shared B -- c2, including unique targets.
        a = row(0, "shared A")
        b = row(1, "shared A")
        c = row(1, "shared B"); c["example_id"] += "_second"
        d = row(2, "shared B")
        e = row(2, "unique C"); e["example_id"] += "_second"
        b["reference_answer"] = "conflicting answer"
        unique, duplicates = group_and_deduplicate([a, b, c, d, e])
        self.assertEqual(len(unique), 3)
        self.assertEqual(len(duplicates), 2)
        self.assertEqual({item["leakage_group"] for item in unique}, {"conv_0"})
        self.assertEqual(sum(item["reference_conflict"] for item in duplicates), 1)

    def test_split_is_reproducible_and_disjoint(self):
        rows = [row(index) for index in range(40)]
        first = split_grouped_rows(rows)
        self.assertEqual(first, split_grouped_rows(rows))
        self.assertEqual({name: len(values) for name, values in first.items()},
                         {"train": 30, "dev": 4, "test": 6})
        validate_splits(first)
        bad = deepcopy(first)
        bad["dev"][0]["leakage_group"] = bad["train"][0]["leakage_group"]
        with self.assertRaisesRegex(ValueError, "group leakage"):
            validate_splits(bad)

    def test_prompt_hash_preserves_scientific_case(self):
        self.assertEqual(prompt_digest([{"role": "user", "content": "A  B"}]),
                         prompt_digest([{"role": "user", "content": "A\nB"}]))
        self.assertNotEqual(prompt_digest([{"role": "user", "content": "N"}]),
                            prompt_digest([{"role": "user", "content": "n"}]))

    def test_token_counter_requests_ids_not_batch_encoding_fields(self):
        measured = measure_lengths(row(0), FakeTokenizer())
        self.assertEqual(measured, {"prompt_tokens": 12, "reference_tokens": 1})
        self.assertNotEqual(measured["prompt_tokens"], 2)

    def test_filter_excludes_whole_examples_and_audits_reason(self):
        splits = {"train": [row(0), row(3, "long question")],
                  "dev": [row(1)], "test": [row(2)]}
        eligible, exclusions = filter_by_lengths(splits, FakeTokenizer(), 20, 20)
        self.assertEqual(len(eligible["train"]), 1)
        self.assertEqual(exclusions[0]["reasons"], ["prompt_too_long"])
        self.assertEqual(splits["train"][1]["user_question"], "long question")

    def test_padding_depends_on_global_batch_only(self):
        rows = [row(index) for index in range(19)]
        expected, ids = pad_training_rows(rows, global_batch_size=64, num_generations=4)
        self.assertEqual(len(expected), 32)
        self.assertEqual(ids, [item["example_id"] for item in rows[:13]])
        self.assertEqual(sum(item["epoch_padding"] for item in expected), 13)
        for workers in (2, 4, 8, 16):
            microbatch, accumulation = 1, 64 // workers
            actual, actual_ids = pad_training_rows(rows, workers * microbatch * accumulation, 4)
            self.assertEqual((actual, actual_ids), (expected, ids))
        self.assertNotIn("epoch_padding", rows[0])

    def test_padding_handles_a_small_nonempty_training_split(self):
        padded, ids = pad_training_rows([row(0)], 64, 4)
        self.assertEqual(len(padded), 16)
        self.assertEqual(len(ids), 15)
        splits = {"train": padded, "dev": [row(1)], "test": [row(2)]}
        validate_splits(splits, allow_padding=True)
        bad = deepcopy(splits)
        bad["train"][-1]["reference_answer"] = "changed"
        with self.assertRaisesRegex(ValueError, "exact repeat"):
            validate_splits(bad, allow_padding=True)

    def test_invalid_global_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            pad_training_rows([row(0)], 15, 4)

    def test_prepare_cache_revision_serialization_and_tamper_detection(self):
        # Exercise the entire preparation path without downloading data or models.
        calls = []
        datasets = types.ModuleType("datasets")
        datasets.Dataset, datasets.DatasetDict = FakeDataset, FakeDatasetDict
        datasets.load_from_disk = fake_load_from_disk

        def fake_load_dataset(name, *, data_files, split, revision):
            calls.append((name, data_files, split, revision))
            return FakeDataset([
                conversation(f"Question {index}") for index in range(40)
            ])

        datasets.load_dataset = fake_load_dataset
        hub = types.ModuleType("huggingface_hub")

        class FakeApi:
            def dataset_info(self, name, *, revision):
                return types.SimpleNamespace(sha="immutable-dataset-revision")

        hub.HfApi = FakeApi
        cfg = {
            "data": {"dataset": "example/science", "categories": ["biology"], "revision": None},
            "model": {"name": "example/model", "max_prompt_tokens": 100, "max_reference_tokens": 100},
            "training": {"global_batch_size": 64, "num_generations": 4},
            "distributed": {"world_size": 2},
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            "sys.modules", {"datasets": datasets, "huggingface_hub": hub},
        ):
            destination = Path(temporary) / "prepared"
            manifest = prepare_data(cfg, destination, FakeTokenizer(), "immutable-model-revision")
            self.assertEqual(calls, [("example/science", "full_train.jsonl", "train", "immutable-dataset-revision")])
            self.assertEqual(manifest["eligible_split_sizes"], {"train": 30, "dev": 4, "test": 6})
            self.assertEqual(manifest["split_sizes"], {"train": 32, "dev": 4, "test": 6})
            self.assertEqual(manifest["optimizer_steps_per_epoch"], 2)
            self.assertEqual(manifest["training_completions_per_epoch"], 128)
            self.assertEqual(manifest["format"], DATA_FORMAT_VERSION)
            self.assertEqual(manifest["policy"]["response_layout"], RESPONSE_LAYOUT)
            self.assertEqual(len(manifest["training_padding_ids"]), 2)
            json.dumps(manifest)  # No non-serializable Dataset columns in the manifest.
            prepared = load_prepared(destination)
            with self.assertRaises(TypeError):
                json.dumps(prepared["train"]["example_id"])

            cfg["distributed"]["world_size"] = 16
            self.assertEqual(prepare_data(cfg, destination, FakeTokenizer(), "immutable-model-revision"), manifest)
            self.assertEqual(len(calls), 1, "Changing workers must not redownload or resplit data")
            with patch("science_gspo_g.prompts.prompt_registry", return_value={"version": "future-scaffold"}):
                with self.assertRaisesRegex(ValueError, "policy changed"):
                    prepare_data(cfg, destination, FakeTokenizer(), "immutable-model-revision")
            self.assertEqual(len(calls), 1, "A stale cache must fail before another download")
            changed = deepcopy(cfg)
            changed["training"]["global_batch_size"] = 32
            with self.assertRaisesRegex(ValueError, "policy changed"):
                prepare_data(changed, destination, FakeTokenizer(), "immutable-model-revision")

            path = destination / "datasets" / "rows.json"
            content = json.loads(path.read_text(encoding="utf-8"))
            content["test"][0]["reference_answer"] = "tampered answer"
            path.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed on disk"):
                load_prepared(destination)


if __name__ == "__main__":
    unittest.main()
