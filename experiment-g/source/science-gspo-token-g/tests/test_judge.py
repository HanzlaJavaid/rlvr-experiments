"""Offline evaluator contract tests. Every client is an in-memory fake."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from science_gspo_g import judge


@pytest.fixture
def cfg():
    return {"judge": {"model": "jev-1.14.0", "concurrency_per_rank": 2},
            "generation": {"max_new_tokens": 100},
            "training": {}}


@pytest.fixture
def row():
    return {"example_id": "q1", "conversation_id": "c1", "leakage_group": "g1",
            "category": "physics", "user_question": "Find velocity.",
            "conversation_context": [], "reference_answer": "OPUS_REFERENCE_SENTINEL 6 m/s", "prompt_tokens": 15,
            "secret_future_field": "must not leak"}


@pytest.fixture
def parsed():
    return {"status": "complete", "ended_with_eos": True,
            "raw_text": "full candidate",
            "reasoning": "TRACE_SENTINEL MODELONLY F=ma; APPLYONLY v=at; CHECKONLY units",
            "final_answer": "6 m/s FINALONLY", "reasoning_token_span": [1, 30],
            "process_token_spans": [[[1, 30]], [[1, 30]], [[1, 30]]],
            "section_availability": {name: True for name in (*judge.STAGES, "answer")},
            "generated_tokens": 36, "reasoning_tokens": 29, "answer_tokens": 5}


class FakeClient:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.model = "jev-1.14.0"
        self.value = 3.0
        self.closed = False

    def system_one(self, *, state, questions):
        self.calls.append((deepcopy(state), deepcopy(questions)))
        if self.failure:
            raise self.failure
        return {"model": self.model, "answers": {
            name: {"type": "score", "score": self.value, "confidence": 0.9,
                   "probabilities": {"0": 0.05, "1": 0.20, "2": 0.25, "3": 0.50}}
            for name in questions}}

    def close(self):
        self.closed = True


def test_four_scores_use_identical_trace_and_reference_only_for_accuracy(tmp_path, cfg, row, parsed):
    client = FakeClient()
    row["conversation_context"] = [{"role": "user", "content": "CONTEXT_SENTINEL",
                                   "reference_answer": "NESTED_TARGET_SENTINEL"}]
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        records, stats = judge.score_parsed_rows([row], [parsed], engine, "protocol")
    assert len(client.calls) == 4
    assert all(len(questions) == 1 for _, questions in client.calls)
    assert not {"coverage", "explanatory_validity"} & set(records[0]["scores"])
    assert stats["new_requests"] == 4
    outcome = next(state for state, questions in client.calls if "scientific_correctness" in questions)
    assert outcome["candidate_answer"] == "6 m/s FINALONLY"
    assert outcome["reference_answer"] == row["reference_answer"]
    assert "TRACE_SENTINEL" not in json.dumps(outcome)
    assert "MODELONLY" not in json.dumps(outcome)
    assert "secret_future_field" not in outcome
    process = {state["stage"]: state for state, questions in client.calls if "stage" in state}
    assert set(process) == set(judge.STAGES)
    for state in process.values():
        assert set(state) == {"category", "user_question", "conversation_context", "stage", "candidate_reasoning"}
        assert state["candidate_reasoning"] == parsed["reasoning"]
        assert all(token in state["candidate_reasoning"] for token in ("MODELONLY", "APPLYONLY", "CHECKONLY"))
        assert "FINALONLY" not in json.dumps(state)
        assert "OPUS_REFERENCE_SENTINEL" not in json.dumps(state)
        assert "reference_answer" not in json.dumps(state)
        assert "NESTED_TARGET_SENTINEL" not in json.dumps(state)
        assert "CONTEXT_SENTINEL" in json.dumps(state)
    assert records[0]["scores"] == {name: 1.0 for name in judge.METRICS}
    assert records[0]["outcome_quality"] == pytest.approx(1.0)
    assert records[0]["q_bad"] == pytest.approx(0.25)
    assert records[0]["reasoning_token_span"] == [1, 30]
    assert records[0]["parsed"]["reasoning"] == parsed["reasoning"]
    assert client.closed


def test_target_reference_changes_only_outcome_request_and_cache_identity(tmp_path, cfg, row, parsed):
    altered = deepcopy(row)
    altered["reference_answer"] = "DIFFERENT_OPUS_SENTINEL"
    altered["unapproved_metadata"] = {"candidate_answer": "LEAK_SENTINEL"}
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", FakeClient) as engine:
        assert engine.state_cache_key(judge.build_jev_state(row, parsed["final_answer"])) != engine.state_cache_key(
            judge.build_jev_state(altered, parsed["final_answer"]))
        for stage, metric in zip(judge.STAGES, judge.PROCESS_METRICS):
            original = judge.build_process_state(row, parsed, stage)
            changed = judge.build_process_state(altered, parsed, stage)
            assert original == changed
            spec = {metric: judge.PROCESS_RUBRICS["physics"][metric]}
            assert engine.state_cache_key(original, spec) == engine.state_cache_key(changed, spec)


def test_process_state_requires_known_aspect_and_real_trace(row, parsed):
    with pytest.raises(ValueError, match="Unknown process stage"):
        judge.build_process_state(row, parsed, "summary")
    for value in (None, "", "   ", 3):
        with pytest.raises(ValueError, match="nonempty native thinking"):
            judge.build_process_state(row, {**parsed, "reasoning": value}, "model")


def test_cache_deduplicates_identical_answers_and_survives_restart(tmp_path, cfg, row, parsed):
    client = FakeClient()
    path = tmp_path / "cache.jsonl"
    with judge.JevEngine(cfg, path, lambda: client) as engine:
        records, stats = judge.score_parsed_rows([row, row], [parsed, parsed], engine, "p")
        assert stats["new_requests"] == 4
        assert stats["cache_reuses"] == 4
        assert records[0]["judge_cache_keys"] == records[1]["judge_cache_keys"]
    with judge.JevEngine(cfg, path, lambda: client) as engine:
        _, stats = judge.score_parsed_rows([row], [parsed], engine, "p")
        assert stats["new_requests"] == 0
    assert len(client.calls) == 4


def test_cache_identity_includes_rubric_and_scope(tmp_path, cfg, row):
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", FakeClient) as engine:
        state = judge.build_jev_state(row, "answer")
        original = engine.state_cache_key(state)
        spec = deepcopy(judge.OUTCOME_RUBRIC_SPEC)
        spec["scientific_correctness"]["instructions"] += " NEW RUBRIC"
        assert engine.state_cache_key(state, spec) != original
        engine.rubric_hash += "new-version"
        assert engine.state_cache_key(state) != original


def test_transport_failure_is_unavailable_and_not_scientific_zero(tmp_path, cfg, row, parsed):
    client = FakeClient()
    client.failure = TimeoutError("offline simulated timeout")
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        records, stats = judge.score_parsed_rows([row], [parsed], engine, "p")
        assert engine.api_calls == 0
        assert not engine.cache
    assert stats['failed_requests']==4
    assert all(value is None for value in records[0]['training_rewards'].values())
    assert records[0]['training_reward_censored']


def test_one_failed_stage_retains_other_scores_and_retries_later(tmp_path,cfg,row,parsed):
    class StageFailure(FakeClient):
        def system_one(self,**kw):
            if kw['state'].get('stage')=='check' and self.stage_failure:
                raise self.stage_failure
            return super().system_one(**kw)
    client=StageFailure();client.stage_failure=TimeoutError('private error body')
    with judge.JevEngine(cfg,tmp_path/'cache.jsonl',lambda:client) as engine:
        records,stats=judge.score_parsed_rows([row],[parsed],engine,'p')
        assert stats['failed_requests']==1
        assert records[0]['training_rewards']['verification_quality'] is None
        assert sum(records[0]['reward_availability'].values())==3
        assert not records[0]['training_reward_censored']
        assert len(engine.cache)==3
        client.stage_failure=None
        recovered,stats=judge.score_parsed_rows([row],[parsed],engine,'p')
        assert stats['new_requests']==1 and stats['failed_requests']==0
        assert all(recovered[0]['reward_availability'].values())


def test_invalid_accuracy_retains_process_metrics_without_poisoning_cache(tmp_path,cfg,row,parsed):
    class PartialClient(FakeClient):
        def system_one(self,**kw):
            result=super().system_one(**kw)
            if 'scientific_correctness' in result['answers']:
                result['answers']['scientific_correctness']['score']=float('nan')
            return result
    with judge.JevEngine(cfg,tmp_path/'cache.jsonl',PartialClient) as engine:
        records,stats=judge.score_parsed_rows([row],[parsed],engine,'p')
        assert records[0]['scores']['scientific_correctness'] is None
        assert records[0]['outcome_quality'] is None
        assert all(records[0]['scores'][metric]==1 for metric in judge.PROCESS_METRICS)
        assert sum(records[0]['reward_availability'].values())==3
        assert len(engine.cache)==3
        assert stats['failed_requests']==1


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0, 4.0, True])
def test_nonfinite_or_out_of_range_score_rejected(tmp_path, cfg, row, bad_value):
    client = FakeClient()
    client.value = bad_value
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        with pytest.raises(ValueError, match="Invalid Jev score"):
            engine.evaluate_states([judge.build_jev_state(row, "answer")])


def test_judge_version_drift_rejected(tmp_path, cfg, row):
    client = FakeClient()
    client.model = "jev-latest"
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        with pytest.raises(ValueError, match="Judge model changed"):
            engine.evaluate_states([judge.build_jev_state(row, "answer")])


def test_outputs_without_unambiguous_native_boundaries_are_unavailable(tmp_path, cfg, row, parsed):
    truncated, malformed = deepcopy(parsed), deepcopy(parsed)
    truncated.update(status="truncated", ended_with_eos=False, section_availability={name: False for name in (*judge.STAGES, "answer")})
    malformed.update(status="malformed", ended_with_eos=True, section_availability={name: False for name in (*judge.STAGES, "answer")})
    client = FakeClient()
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        records, stats = judge.score_parsed_rows([row, row], [truncated, malformed], engine, "p")
    assert stats["new_requests"] == 0
    assert client.calls == []
    assert all(value is None for value in records[0]["training_rewards"].values())
    assert all(value is None for value in records[1]["training_rewards"].values())
    assert records[1]["training_reward_censored"]


def test_callbacks_share_one_scoring_pass_and_expose_metadata(tmp_path, cfg, row, parsed, monkeypatch):
    monkeypatch.setattr(judge, "parse_completion", lambda *args: deepcopy(parsed))
    columns = {key: [value] for key, value in row.items()}
    client = FakeClient()
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        callbacks = judge.make_reward_functions(engine, object(), cfg, 0, tmp_path, "p")
        scores = [fn(completion_ids=[[1, 2, 3]], trainer_state=SimpleNamespace(global_step=1), **columns)
                  for fn in callbacks]
        assert scores == [[1.0]] * 4
        assert len(client.calls) == 4
        shared = callbacks[0].reward_batch
        assert all(fn.reward_batch is shared for fn in callbacks)
        assert shared["records"][0]["reasoning_token_span"] == [1, 30]
        path = tmp_path / "ranks/rank_00000/training_generations.jsonl"
        assert len(path.read_text().splitlines()) == 1
        # Repeated callback starts a new rollout audit but reuses semantic judge cache.
        callbacks[0](completion_ids=[[1, 2, 3]], trainer_state=SimpleNamespace(global_step=1), **columns)
        assert len(path.read_text().splitlines()) == 2
        assert len(client.calls) == 4


def test_all_subject_process_rubrics_are_present():
    assert len(judge.PROCESS_RUBRICS) == 8
    assert sum(len(profile) for profile in judge.PROCESS_RUBRICS.values()) == 24
    assert judge.METRICS == ["scientific_correctness", "model_quality", "application_quality", "verification_quality"]
    assert len({judge.PROCESS_RUBRICS[s]["model_quality"]["instructions"] for s in judge.PROCESS_RUBRICS}) == 8


def test_outcome_formula_uses_accuracy_only_and_preserves_valid_zero():
    scores = {"scientific_correctness": 0.0, "coverage": 1.0, "explanatory_validity": 1.0}
    assert judge.outcome_quality(scores) == 0.0
    scores["scientific_correctness"] = 0.5
    assert judge.outcome_quality(scores) == 0.5
    assert judge.outcome_quality({"scientific_correctness": None}) is None


def test_real_parser_feeds_all_four_scoped_requests(tmp_path, cfg, row):
    from test_parsing import ByteTokenizer
    tokenizer = ByteTokenizer()
    trace = "Use F=ma. a=3 m/s². Dimensions are acceleration."
    final = "The acceleration is 3 m/s²."
    ids = tokenizer.encode(trace + "</think>" + final) + [tokenizer.eos_token_id]
    client = FakeClient()
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        records = engine.evaluate_completions([row], [ids], tokenizer, len(ids))
    assert records[0]["status"] == "complete"
    assert len(client.calls) == 4
    for state, questions in client.calls:
        if "scientific_correctness" in questions:
            assert state["candidate_answer"] == final
            assert "candidate_reasoning" not in state
        else:
            assert state["candidate_reasoning"] == trace
            assert "candidate_answer" not in state
            assert "reference_answer" not in state


def test_failed_request_can_retry_without_poisoning_cache(tmp_path, cfg, row):
    client = FakeClient()
    client.failure = RuntimeError("transient test failure")
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        state = judge.build_jev_state(row, "answer")
        with pytest.raises(judge.JudgeRequestError, match="RuntimeError"):
            engine.evaluate_states([state])
        client.failure = None
        result, _, stats = engine.evaluate_states([state])
        assert result[0]["answers"]["scientific_correctness"]["score"] == 3.0
        assert stats["new_requests"] == 1


def test_service_error_traceback_excludes_credentials_and_response_body(tmp_path,cfg,row):
    import traceback
    client=FakeClient()
    client.failure=RuntimeError('API_KEY_SENTINEL response body and private request')
    client.failure.response=SimpleNamespace(status_code=403)
    with judge.JevEngine(cfg,tmp_path/'cache.jsonl',lambda:client) as engine:
        with pytest.raises(judge.JudgeRequestError) as caught:
            engine.evaluate_states([judge.build_jev_state(row,'answer')])
    rendered=''.join(traceback.format_exception(caught.value))
    assert 'API_KEY_SENTINEL' not in rendered
    assert 'response body' not in rendered
    assert caught.value.status_code == 403
    assert caught.value.source_error_type == 'RuntimeError'


def test_native_trace_scores_and_token_credit_end_to_end(tmp_path, cfg, row):
    from test_parsing import ByteTokenizer
    from science_gspo_g.advantages import build_advantages
    tokenizer = ByteTokenizer()
    good = "<think>MODEL F=ma. APPLY a=3 m/s². CHECK units are acceleration.</think>The acceleration is 3 m/s²."
    bad = good.replace("a=3", "a=9")

    class VaryingClient(FakeClient):
        def system_one(self, **kw):
            value = 0.0 if "a=9" in kw["state"].get("candidate_reasoning", "") else 3.0
            response = super().system_one(**kw)
            for answer in response["answers"].values():
                answer["score"] = value
            return response

    client = VaryingClient()
    sequences = [tokenizer.encode(text) + [tokenizer.eos_token_id] for text in (good, bad)]
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        records = engine.evaluate_completions([row, row], sequences, tokenizer, max(map(len, sequences)))
    assert all(record["status"] == "complete" for record in records)
    metadata = [dict(example_id="q1", length=len(ids), **{k: r[k] for k in
                ("status", "reasoning_token_span", "process_token_spans")})
                for ids, r in zip(sequences, records)]
    scores = [[r["scores"][metric] for metric in judge.METRICS] for r in records]
    built = build_advantages(scores, metadata, group_size=2)
    assert built["outcome_advantages"] == [0, 0]  # Same final answer; process alone distinguishes traces.
    for i, record in enumerate(records):
        start, stop = record["reasoning_token_span"]
        advantage = 0.25 if i == 0 else -0.25
        assert built["advantages"][i][:start] == [0] * start
        assert built["advantages"][i][start:stop] == pytest.approx([advantage] * (stop-start))
        assert built["advantages"][i][stop:len(sequences[i])] == [0] * (len(sequences[i])-stop)


@pytest.mark.parametrize("eos", [True, False])
def test_closed_trace_without_final_retains_all_process_rewards(tmp_path, cfg, row, eos):
    from test_parsing import ByteTokenizer
    tokenizer = ByteTokenizer()
    text = "Use F=ma. a=3 m/s². Units are acceleration.</think>"
    ids = tokenizer.encode(text) + ([tokenizer.eos_token_id] if eos else [])
    parsed = judge.parse_completion(ids, tokenizer, len(ids))
    client = FakeClient()
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        records, stats = judge.score_parsed_rows([row], [parsed], engine, "p")
    assert stats["new_requests"] == 3
    assert records[0]["scores"]["scientific_correctness"] is None
    assert all(records[0]["scores"][metric] == 1 for metric in judge.PROCESS_METRICS)
    assert all("candidate_reasoning" in state and "reference_answer" not in state for state, _ in client.calls)


def test_empty_thinking_preserves_independent_final_accuracy(tmp_path, cfg, row):
    from test_parsing import ByteTokenizer
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("</think>The acceleration is 3 m/s².") + [tokenizer.eos_token_id]
    client = FakeClient()
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        records = engine.evaluate_completions([row], [ids], tokenizer, len(ids))
    assert len(client.calls) == 1
    assert records[0]["scores"]["scientific_correctness"] == 1
    assert all(records[0]["scores"][metric] is None for metric in judge.PROCESS_METRICS)


@pytest.mark.parametrize("text", ["No closing boundary.", "Trace.</think>Answer.</think>Other.", "Trace.<think>More.</think>Answer."])
def test_boundary_corruption_never_sends_any_candidate_to_judge(tmp_path, cfg, row, text):
    from test_parsing import ByteTokenizer
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
    client = FakeClient()
    with judge.JevEngine(cfg, tmp_path / "cache.jsonl", lambda: client) as engine:
        records = engine.evaluate_completions([row], [ids], tokenizer, len(ids))
    assert client.calls == []
    assert all(value is None for value in records[0]["scores"].values())


def test_process_rubrics_grade_whole_trace_without_reference_or_section_contracts():
    assert judge.RUBRIC_VERSION.startswith("science-gspo-g-")
    for profile in judge.PROCESS_RUBRICS.values():
        for rubric in profile.values():
            instructions = rubric["instructions"]
            assert "candidate_reasoning" in instructions
            assert "whole-trace feedback" in instructions
            assert "reference" not in instructions.lower()
            assert "candidate_prefix" not in instructions
            assert "candidate_section" not in instructions
            assert "Appropriate brevity" in instructions
    assert "reference_answer" in judge.OUTCOME_RUBRIC_SPEC["scientific_correctness"]["instructions"]
