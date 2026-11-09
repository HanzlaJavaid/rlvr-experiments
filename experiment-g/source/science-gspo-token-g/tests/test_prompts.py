import json
import unittest

from science_gspo_g.prompts import SUBJECTS, PROMPT_VERSION, build_system_prompt, prompt_registry


class PromptTests(unittest.TestCase):
    def test_eight_subjects_share_untagged_reasoning_contract(self):
        self.assertEqual(set(SUBJECTS), {"physics", "chemistry", "biology", "medicine", "earth_science", "geography", "math", "science"})
        prompts = [build_system_prompt(s) for s in SUBJECTS]
        self.assertEqual(len(set(prompts)), 8)
        for prompt in prompts:
            for tag in ("<model>", "<apply>", "<check>", "<answer>"):
                self.assertNotIn(tag, prompt)
            self.assertIn("false", prompt)
            self.assertIn("Keep simple questions brief", prompt)
            self.assertIn("no fixed sequence of headings is required", prompt)
            self.assertIn("In your native thinking", prompt)
            self.assertIn("exactly one native </think> boundary", prompt)
            self.assertIn("self-contained final answer", prompt)

    def test_subject_guidance_and_fallback(self):
        self.assertIn("dimensions", build_system_prompt("physics"))
        self.assertIn("equilibrium", build_system_prompt("chemistry"))
        self.assertIn("exceptions", build_system_prompt("biology"))
        self.assertEqual(build_system_prompt("mixed"), build_system_prompt("science"))

    def test_native_thinking_contains_the_three_activities_without_literal_headings(self):
        for subject in SUBJECTS:
            prompt = build_system_prompt(subject)
            self.assertIn("choose an appropriate framework", prompt)
            self.assertIn("apply them with justified", prompt)
            self.assertIn("check the result or explanation", prompt)
            self.assertIn("not separate labeled sections", prompt)
            self.assertNotIn("model:", prompt)
            self.assertNotIn("apply:", prompt)
            self.assertNotIn("check:", prompt)
            self.assertIn("Correct errors that you discover", prompt)

    def test_new_version_and_no_inherited_output_example(self):
        self.assertEqual(PROMPT_VERSION, "science-gspo-g-native-thinking-v1")
        self.assertNotIn("constant-mass object", build_system_prompt("physics"))
        self.assertNotIn("Opus", build_system_prompt("physics"))

    def test_registry_is_serializable_and_independent(self):
        registry = prompt_registry()
        self.assertEqual(registry["version"], PROMPT_VERSION)
        json.dumps(registry)
        registry["subjects"]["math"]["model"] = "mutated"
        self.assertNotEqual(prompt_registry()["subjects"]["math"]["model"], "mutated")


if __name__ == "__main__":
    unittest.main()
