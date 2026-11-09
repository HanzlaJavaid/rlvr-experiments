"""Natural-thinking prompt isolation, routing, grouping, and length limits."""
from copy import deepcopy
import unittest

from science_gspo_g.data import (
    apply_subject_scaffolds, conversation_to_samples, filter_by_lengths,
    group_and_deduplicate, measure_lengths, split_grouped_rows,
)
from science_gspo_g.prompts import SUBJECTS, PROMPT_VERSION, build_system_prompt


def samples(index, category="physics", question=None, source_system="Original source system"):
    conversation = {"category": category, "messages": [
        {"role": "system", "content": source_system},
        {"role": "user", "content": f"Earlier question {index}"},
        {"role": "assistant", "content": f"Earlier fallible claim {index}"},
        {"role": "system", "content": "Another source instruction"},
        {"role": "user", "content": question or f"Target question {index}"},
        {"role": "assistant", "content": f"GOLD_REFERENCE_SENTINEL_{index}"},
    ]}
    row = conversation_to_samples(conversation, index)[-1]
    row["leakage_group"] = row["conversation_id"]
    return row


class LengthTokenizer:
    def apply_chat_template(self, prompt, *, tokenize, add_generation_prompt, return_dict, enable_thinking=True):
        assert enable_thinking is True
        return list(range(sum(len(m["content"]) for m in prompt) + len(prompt) * 2))

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": list(range(len(text.split())))}


class ScaffoldDataTests(unittest.TestCase):
    def split(self):
        return {"train": [samples(0), samples(3, "biology")],
                "dev": [samples(1, "chemistry")], "test": [samples(2, "math")]}

    def test_originals_and_judge_context_unchanged_references_not_in_policy(self):
        original = self.split()
        before = deepcopy(original)
        transformed = apply_subject_scaffolds(original)
        self.assertEqual(original, before)
        for split in original:
            for old, new in zip(original[split], transformed[split]):
                self.assertEqual(new["reference_answer"], old["reference_answer"])
                self.assertNotIn(new["reference_answer"], str(new["prompt"]))
                self.assertEqual(new["conversation_context"], old["conversation_context"])
                self.assertEqual(new["source_prompt_hash"], old["prompt_hash"])
                self.assertNotEqual(new["prompt_hash"], old["prompt_hash"])
                self.assertEqual(new["scaffold_version"], PROMPT_VERSION)
        transformed["train"][0]["conversation_context"][0]["content"] = "Mutated copy"
        self.assertEqual(original, before)

    def test_all_source_system_messages_replaced_only_in_policy(self):
        new = apply_subject_scaffolds(self.split())["train"][0]
        system = [m for m in new["prompt"] if m["role"] == "system"]
        self.assertEqual(system, [{"role": "system", "content": build_system_prompt("physics")}])
        self.assertNotIn("Original source system", str(new["prompt"]))
        self.assertNotIn("Another source instruction", str(new["prompt"]))
        self.assertIn("Original source system", str(new["conversation_context"]))
        self.assertIn("Earlier fallible claim 0", str(new["prompt"]))
        self.assertEqual(new["prompt"][-1], {"role": "user", "content": new["user_question"]})

    def test_routing_every_subject_and_explicit_fallback(self):
        for category in (*SUBJECTS, "mixed"):
            with self.subTest(category=category):
                data = self.split()
                data["train"][0]["category"] = category
                result = apply_subject_scaffolds(data)
                self.assertEqual(result["train"][0]["prompt"][0]["content"], build_system_prompt(category))

    def test_group_assignments_survive_transformation(self):
        rows, _ = group_and_deduplicate([samples(i, SUBJECTS[i % 8]) for i in range(40)])
        original = split_grouped_rows(rows)
        updated = apply_subject_scaffolds(original)
        for split in original:
            self.assertEqual([(r["example_id"], r["conversation_id"], r["leakage_group"]) for r in original[split]],
                             [(r["example_id"], r["conversation_id"], r["leakage_group"]) for r in updated[split]])

    def test_filter_counts_new_scaffold_length_not_original_prompt(self):
        tokenizer = LengthTokenizer()
        original = {"train": [samples(0), samples(3, question="Long target " + "x" * 400)],
                    "dev": [samples(1)], "test": [samples(2)]}
        scaffolded = apply_subject_scaffolds(original)
        cutoff = measure_lengths(scaffolded["train"][0], tokenizer)["prompt_tokens"] + 10
        self.assertTrue(all(measure_lengths(r, tokenizer)["prompt_tokens"] < cutoff
                            for values in original.values() for r in values))
        eligible, exclusions = filter_by_lengths(scaffolded, tokenizer, cutoff, 100)
        self.assertEqual([r["example_id"] for r in eligible["train"]], [original["train"][0]["example_id"]])
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(exclusions[0]["reasons"], ["prompt_too_long"])
        self.assertGreater(exclusions[0]["prompt_tokens"], cutoff)

    def test_post_replacement_prompt_collision_fails_closed(self):
        # Separate source systems can distinguish inputs before replacement;
        # dropping them must not silently permit equal policy inputs in two splits.
        original = {"train": [samples(0)], "dev": [samples(1)], "test": [samples(2)]}
        duplicate = deepcopy(original["train"][0])
        duplicate.update(example_id="other", conversation_id="other", leakage_group="other")
        duplicate["prompt"][0]["content"] = "Different source system"
        from science_gspo_g.data import prompt_digest
        duplicate["prompt_hash"] = prompt_digest(duplicate["prompt"])
        original["dev"] = [duplicate]
        with self.assertRaisesRegex(ValueError, "Prompt leakage"):
            apply_subject_scaffolds(original)


if __name__ == "__main__":
    unittest.main()
