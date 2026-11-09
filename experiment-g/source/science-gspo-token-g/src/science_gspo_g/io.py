"""Small IO helpers. A JSONL journal must have a single process owner."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path | str, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: Path | str, *, repair_tail: bool = True) -> list[dict]:
    """Repair only an incomplete final write, never silently ignore earlier corruption.

    Set repair_tail=False when reading another rank's journal; only its owner may repair it.
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows, valid_bytes = [], 0
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                if repair_tail:
                    with path.open("r+b") as stream:
                        stream.truncate(valid_bytes)
                return rows
            raise ValueError(f"Corrupt JSONL record {index + 1} in {path}") from None
        valid_bytes += len(line)
    if lines and not lines[-1].endswith(b"\n") and repair_tail:
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows

