import unittest

from science_gspo_g.parsing import parse_completion


class ByteTokenizer:
    """A byte-level stub: multi-byte Unicode decodes differently mid-character."""
    eos_token_id = 9001
    unk_token_id = 9999
    all_special_ids = [9001, 9002, 9003, 9004, 9005]
    specials = {"<eos>": 9001, "</think>": 9002, "<think>": 9003,
                "<role>": 9004, "<pad>": 9005}

    def convert_tokens_to_ids(self, token):
        return self.specials.get(token, self.unk_token_id)

    def encode(self, text):
        ids = []
        while text:
            special = next((s for s in self.specials if text.startswith(s)), None)
            if special:
                ids.append(self.specials[special])
                text = text[len(special):]
            else:
                ids.extend(text[0].encode("utf-8"))
                text = text[1:]
        return ids

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        inverse = {value: key for key, value in self.specials.items()}
        raw = b"".join(inverse[i].encode() if i in inverse else bytes([i])
                       for i in ids if not skip_special_tokens or i not in inverse)
        return raw.decode("utf-8", errors="replace")


VISIBLE = "The acceleration is 3 m/s²."
REASONING = "Use F=ma. With 6 N and 2 kg, a=3 m/s². Dimensions are consistent. μ 🧪"
TEXT = REASONING + "\n</think>\n" + VISIBLE


class ParsingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()

    def parse(self, text=TEXT, budget=None, eos=True):
        ids = self.tokenizer.encode(text) + ([self.tokenizer.eos_token_id] if eos else [])
        return parse_completion(ids, self.tokenizer, budget or len(ids) + 10)

    def test_plain_final_and_one_shared_whole_trace(self):
        parsed = self.parse()
        self.assertEqual(parsed["status"], "complete")
        self.assertEqual(parsed["scoring_status"], "complete")
        self.assertEqual(parsed["reasoning"], REASONING)
        self.assertEqual(parsed["final_answer"], VISIBLE)
        self.assertEqual(parsed["section_availability"], dict(model=True, apply=True, check=True, answer=True))
        self.assertTrue(all(prefix == REASONING for prefix in parsed["process_prefixes"].values()))
        self.assertEqual(parsed["process_token_spans"], [[parsed["reasoning_token_span"]]] * 3)

    def test_original_utf8_tokens_define_trace_boundary_without_reencoding(self):
        parsed = self.parse()
        ids = parsed["token_ids"]
        start, stop = parsed["reasoning_token_span"]
        self.assertEqual(ids[stop], 9002)
        self.assertEqual(self.tokenizer.decode(ids[start:stop]).strip(), REASONING)
        self.assertEqual(parsed["reasoning_tokens"], stop - start)
        self.assertEqual(parsed["visible_start_token"], stop + 1)
        self.assertEqual(len(parsed["token_section_ids"]), len(ids))
        self.assertEqual(parsed["token_section_ids"][stop], -1)
        self.assertEqual(parsed["token_section_ids"][-1], -1)

    def test_optional_opening_is_never_in_trace_or_reward_span(self):
        parsed = self.parse(" \n<think>\n" + TEXT)
        self.assertEqual(parsed["status"], "complete")
        start, stop = parsed["reasoning_token_span"]
        self.assertEqual(parsed["token_ids"][start - 1], 9003)
        self.assertTrue(all(x == -1 for x in parsed["token_section_ids"][:start]))
        self.assertNotIn("<think>", parsed["reasoning"])
        self.assertNotIn(9002, parsed["token_ids"][start:stop])
        self.assertNotIn(9003, parsed["token_ids"][start:stop])

    def test_padding_after_eos_and_eos_at_budget(self):
        ids = self.tokenizer.encode(TEXT) + [9001]
        parsed = parse_completion(ids + [9005, 9005], self.tokenizer, len(ids))
        self.assertEqual(parsed["status"], "complete")
        self.assertEqual(parsed["generated_tokens"], len(ids))
        self.assertEqual(parsed["token_ids"], ids)

    def test_truncated_final_preserves_closed_trace_but_not_accuracy(self):
        ids = self.tokenizer.encode(TEXT)
        parsed = parse_completion(ids, self.tokenizer, len(ids))
        self.assertEqual(parsed["status"], "truncated")
        self.assertEqual(parsed["parse_error"], "missing_eos")
        self.assertEqual(parsed["section_availability"], dict(model=True, apply=True, check=True, answer=False))
        self.assertEqual(parsed["scoring_status"], "partial")
        self.assertEqual(parsed["process_token_spans"], [[parsed["reasoning_token_span"]]] * 3)
        self.assertEqual(parse_completion(ids, self.tokenizer, len(ids) + 1)["status"], "abnormal_stop")

    def test_empty_final_preserves_closed_trace(self):
        parsed = self.parse(REASONING + "</think> ")
        self.assertEqual(parsed["status"], "empty_final")
        self.assertFalse(parsed["section_availability"]["answer"])
        self.assertTrue(parsed["section_availability"]["check"])

    def test_empty_trace_has_no_process_but_final_is_independently_scoreable(self):
        parsed = self.parse("<think> \n</think>" + VISIBLE)
        self.assertEqual(parsed["status"], "empty_reasoning")
        self.assertEqual(parsed["scoring_status"], "partial")
        self.assertEqual(parsed["section_availability"], dict(model=False, apply=False, check=False, answer=True))
        self.assertIsNone(parsed["reasoning_token_span"])
        self.assertEqual(parsed["process_token_spans"], [[], [], []])
        self.assertEqual(parsed["final_answer"], VISIBLE)

    def test_no_content_is_not_scoreable(self):
        parsed = self.parse("</think>")
        self.assertEqual(parsed["scoring_status"], "unavailable")
        self.assertFalse(any(parsed["section_availability"].values()))

    def test_native_corruption_is_never_salvaged(self):
        cases = [
            (TEXT.replace("</think>", ""), "native_thinking_boundary"),
            (TEXT.replace("</think>", "</think></think>"), "native_thinking_boundary"),
            ("reasoning <think>" + TEXT, "native_thinking_open_position"),
            ("<think><think>" + TEXT, "native_thinking_open_position"),
            (TEXT + "<think>reopen", "native_thinking_open_position"),
            (TEXT + "<role>", "unexpected_native_special_token"),
        ]
        for text, reason in cases:
            with self.subTest(reason=reason, text=text):
                for ended in (True, False):
                    parsed = self.parse(text, eos=ended)
                    self.assertEqual(parsed["parse_error"], reason)
                    self.assertEqual(parsed["scoring_status"], "unavailable")
                    self.assertIsNone(parsed["reasoning_token_span"])
                    self.assertFalse(any(parsed["section_availability"].values()))

    def test_open_trace_without_genuine_close_is_not_process_scored(self):
        parsed = self.parse("I am still reasoning", eos=False)
        self.assertFalse(any(parsed["section_availability"].values()))

    def test_budget_exhaustion_is_reported_even_when_native_trace_is_invalid(self):
        for text in ("I am still reasoning", TEXT.replace("</think>", "</think></think>")):
            with self.subTest(text=text):
                ids = self.tokenizer.encode(text)
                parsed = parse_completion(ids, self.tokenizer, len(ids))
                self.assertEqual(parsed["status"], "truncated")
                self.assertEqual(parsed["parse_error"], "native_thinking_boundary")
                self.assertFalse(any(parsed["section_availability"].values()))
                self.assertEqual(parsed["scoring_status"], "unavailable")

    def test_over_budget_disables_all_reward_channels(self):
        for ended in (True, False):
            parsed = self.parse(budget=1, eos=ended)
            self.assertEqual(parsed["parse_error"], "completion_exceeds_token_budget")
            self.assertFalse(any(parsed["section_availability"].values()))

    def test_literal_xml_is_ordinary_content_not_a_second_output_schema(self):
        thinking = "Consider <model>m</model>, <apply>a</apply> and <check>c</check> as literal XML."
        final = "The XML answer uses <answer> text."
        parsed = self.parse(thinking + "</think>" + final)
        self.assertEqual(parsed["status"], "complete")
        self.assertEqual(parsed["reasoning"], thinking)
        self.assertEqual(parsed["final_answer"], final)

    def test_literal_close_without_native_id_cannot_fake_a_boundary(self):
        ids = list(TEXT.encode("utf-8")) + [9001]
        result = parse_completion(ids, self.tokenizer, len(ids))
        self.assertEqual(result["parse_error"], "native_thinking_boundary")
        self.assertFalse(any(result["section_availability"].values()))

    def test_alternate_eos_override(self):
        ids = self.tokenizer.encode(TEXT) + [9005]
        self.assertEqual(parse_completion(ids, self.tokenizer, len(ids), [9001, 9005])["status"], "complete")

    def test_missing_or_overlapping_native_marker_is_configuration_error(self):
        self.tokenizer.convert_tokens_to_ids = lambda token: 9999
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.parse()
        self.tokenizer.convert_tokens_to_ids = lambda token: 9002
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.parse()

    def test_long_trace_uses_constant_number_of_decodes(self):
        class CountingTokenizer(ByteTokenizer):
            def __init__(self): self.calls = 0
            def decode(self, ids, **kwargs):
                self.calls += 1
                return super().decode(ids, **kwargs)
        tok = CountingTokenizer()
        ids = tok.encode("x" * 7900 + "</think>final") + [9001]
        parsed = parse_completion(ids, tok, 8192)
        self.assertEqual(parsed["status"], "complete")
        self.assertLess(tok.calls, 10)
        self.assertEqual(parsed["reasoning_token_span"], [0, 7900])

    def test_multi_character_tokens_do_not_move_the_native_boundary(self):
        class ChunkTokenizer(ByteTokenizer):
            chunks = {7000: "framework and application μ", 7001: "check result 🧪", 7002: "plain final"}
            def decode(self, ids, **kwargs):
                inverse = {v: k for k, v in self.specials.items()}
                return "".join(self.chunks.get(i, inverse.get(i, "")) for i in ids)
        tok = ChunkTokenizer()
        parsed = parse_completion([9003, 7000, 7001, 9002, 7002, 9001], tok, 6)
        self.assertEqual(parsed["status"], "complete")
        self.assertEqual(parsed["reasoning_token_span"], [1, 3])
        self.assertEqual(parsed["process_token_spans"], [[[1, 3]]] * 3)
        self.assertEqual(parsed["section_token_spans"]["answer"], [4, 5])
        self.assertEqual(parsed["token_section_ids"], [-1, 0, 0, -1, 3, -1])


if __name__ == "__main__":
    unittest.main()


def test_open_thinking_budget_exhaustion_reaches_callback_truncation_metric(tmp_path):
    from types import SimpleNamespace
    from science_gspo_g.judge import JevEngine, make_reward_functions

    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("Unfinished thinking without a boundary")
    cfg = {"judge": {"model": "jev-1.14.0"}, "generation": {"max_new_tokens": len(ids)}}
    columns = {"example_id": ["q1"], "conversation_id": ["c1"], "leakage_group": ["c1"],
               "category": ["physics"], "user_question": ["What is force?"],
               "conversation_context": [[]], "reference_answer": ["A reference"], "prompt_tokens": [5]}
    metrics = []
    def no_client():
        raise AssertionError("An unclosed trace must not trigger any judge request")

    with JevEngine(cfg, tmp_path / "cache.jsonl", no_client) as engine:
        callbacks = make_reward_functions(engine, tokenizer, cfg, 0, tmp_path, "test")
        scores = callbacks[0](completion_ids=[ids], trainer_state=SimpleNamespace(global_step=0),
                              log_metric=lambda name, value: metrics.append((name, value)), **columns)
    assert scores == [None]
    assert ("quality/truncated_rate", 1.0) in metrics
    assert ("quality/invalid_other_rate", 0.0) in metrics
    assert callbacks[0].reward_batch["records"][0]["parse_error"] == "native_thinking_boundary"
