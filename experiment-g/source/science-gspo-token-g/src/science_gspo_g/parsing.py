"""Parse native thinking and an untagged final answer from original token IDs.

The genuine native closing token separates the two channels. No re-encoding or
heuristic search for a convenient textual delimiter is used. All three process
rubrics share one trace and one half-open token span; these are whole-trace scores,
not labels for three separately located steps.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

STAGES = ("model", "apply", "check")
COMPLETION_FORMAT_VERSION = "native-thinking-plain-final-v1"


def _decode(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def parse_completion(
    token_ids: Iterable[int],
    tokenizer: Any,
    max_new_tokens: int,
    eos_token_ids: int | Iterable[int] | None = None,
) -> dict[str, Any]:
    """Extract an unambiguous closed trace and, independently, a complete final.

    Qwen's template normally supplies the opening marker, so an opening token in
    the generation is optional. A generated opening is allowed only once at the
    start, apart from whitespace. Exactly one genuine closing token is required.
    Duplicate boundaries, role corruption, and over-budget sequences are never
    salvaged. Padding after the first EOS is ignored.

    A closed nonempty trace remains process-scoreable if the final is empty or
    truncated. Scientific correctness requires a nonempty final followed by EOS,
    but does not require a nonempty trace. No process bonus reaches native markers,
    final-answer tokens, or EOS. Whitespace within the trace belongs to its span.
    """
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    ids = [int(token) for token in token_ids]
    configured = eos_token_ids if eos_token_ids is not None else tokenizer.eos_token_id
    eos_ids = {configured} if isinstance(configured, int) else set(configured or [])
    if not eos_ids:
        raise ValueError("No EOS token IDs configured for completion parsing")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    if close_id is None or close_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("Tokenizer has no distinct </think> token")
    open_id = tokenizer.convert_tokens_to_ids("<think>")
    if open_id == getattr(tokenizer, "unk_token_id", None):
        open_id = None
    if close_id in eos_ids or open_id is not None and (open_id in eos_ids or open_id == close_id):
        raise ValueError("Native thinking markers must be distinct from each other and EOS")

    eos_at = next((i for i, token in enumerate(ids) if token in eos_ids), None)
    ended = eos_at is not None
    if ended:
        ids = ids[:eos_at + 1]
    payload = ids[:-1] if ended else ids
    close_positions = [i for i, token in enumerate(payload) if token == close_id]
    open_positions = [i for i, token in enumerate(payload) if token == open_id]
    boundary = close_positions[0] if len(close_positions) == 1 else None
    valid_open = not open_positions or (
        len(open_positions) == 1 and boundary is not None and open_positions[0] < boundary
        and not _decode(tokenizer, payload[:open_positions[0]]).strip()
    )
    extra_special = set(getattr(tokenizer, "all_special_ids", [])) - {close_id, open_id}
    native_error = (
        "completion_exceeds_token_budget" if len(ids) > max_new_tokens else
        "native_thinking_boundary" if boundary is None else
        "native_thinking_open_position" if not valid_open else
        "unexpected_native_special_token" if any(token in extra_special for token in payload) else
        ""
    )
    native_valid = not native_error
    start = open_positions[0] + 1 if open_positions and valid_open else 0
    reasoning = _decode(tokenizer, payload[start:boundary]).strip() if native_valid else ""
    final_answer = _decode(tokenizer, payload[boundary + 1:]).strip() if native_valid else ""
    reasoning_available = native_valid and bool(reasoning)
    answer_available = native_valid and ended and bool(final_answer)

    # Termination and reward eligibility are separate. A budget-exhausted open
    # or corrupt trace is still a truncation in diagnostics, but cannot be scored.
    if not ended and len(ids) == max_new_tokens:
        status, reason = "truncated", native_error or "missing_eos"
    elif native_error:
        status, reason = "malformed", native_error
    elif not ended:
        status = "truncated" if len(ids) >= max_new_tokens else "abnormal_stop"
        reason = "missing_eos"
    elif not final_answer:
        status, reason = "empty_final", "empty_answer"
    elif not reasoning:
        status, reason = "empty_reasoning", "empty_reasoning"
    else:
        status, reason = "complete", ""

    trace_span = [start, boundary] if reasoning_available else None
    answer_span = [boundary + 1, len(payload)] if answer_available else None
    availability = {stage: reasoning_available for stage in STAGES}
    availability["answer"] = answer_available
    scoring_status = ("complete" if all(availability.values()) else
                      "partial" if any(availability.values()) else "unavailable")
    # Compatibility field names are retained for journals/readers. They now mirror
    # one whole trace; they do not describe separately tagged sections or prefixes.
    sections = {stage: reasoning for stage in STAGES} if reasoning_available else {}
    spans = {stage: list(trace_span) for stage in STAGES} if trace_span else {}
    if answer_span:
        spans["answer"] = answer_span
    token_section_ids = [-1] * len(ids)
    if trace_span:
        token_section_ids[start:boundary] = [0] * (boundary - start)
    if answer_span:
        token_section_ids[answer_span[0]:answer_span[1]] = [3] * (answer_span[1] - answer_span[0])
    return {
        "raw_text": _decode(tokenizer, ids),
        "completion_format_version": COMPLETION_FORMAT_VERSION,
        "token_ids": ids,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "status": status,
        "parse_error": reason,
        "ended_with_eos": ended,
        "generated_tokens": len(ids),
        "reasoning_tokens": boundary - start if native_valid else 0,
        "reasoning_token_span": trace_span,
        "native_boundary_count": len(close_positions),
        "native_open_count": len(open_positions),
        "visible_start_token": boundary + 1 if native_valid else None,
        "visible_tokens": len(payload) - boundary - 1 if native_valid else 0,
        "answer_tokens": answer_span[1] - answer_span[0] if answer_span else 0,
        "sections": sections,
        "process_prefixes": dict(sections),
        "section_end_tokens": [boundary - 1] * 3 if reasoning_available else [None] * 3,
        "process_start_token": start if reasoning_available else None,
        "section_token_spans": spans,
        "token_section_ids": token_section_ids,
        "boundary_straddling_tokens": [],
        "section_availability": availability,
        "scoring_status": scoring_status,
        "process_token_spans": [[list(trace_span)] for _ in STAGES] if trace_span else [[], [], []],
    }
