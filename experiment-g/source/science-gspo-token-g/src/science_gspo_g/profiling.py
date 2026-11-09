"""Optimizer-update measurements; no CUDA, model, or service imports."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import statistics
import time

from .io import append_jsonl, atomic_json, read_jsonl


class StepProfiler:
    def __init__(self, path: Path, *, rank: int, clock=time.perf_counter):
        self.path, self.rank, self.clock = path, rank, clock
        self.started = None

    def begin(self, step: int):
        if self.started is not None:
            raise RuntimeError("Previous optimizer-update measurement is still open")
        self.step, self.started = step, self.clock()
        self.stages = defaultdict(float)
        self.tokens = self.completions = 0

    def add_stage(self, name: str, seconds: float):
        if self.started is not None:
            self.stages[name] += seconds

    def add_tokens(self, tokens: int, completions: int):
        if self.started is not None:
            self.tokens += tokens
            self.completions += completions

    def end(self, step: int, **memory):
        if self.started is None or self.step != step:
            raise RuntimeError("Mismatched optimizer-update measurement")
        row = {
            "rank": self.rank, "global_step": step,
            "update_seconds": self.clock() - self.started,
            "stages_seconds": dict(self.stages),
            "generated_tokens": self.tokens, "generated_completions": self.completions,
            **memory,
        }
        append_jsonl(self.path, [row])
        self.started = None
        return row


