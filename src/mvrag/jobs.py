"""Resumable batched generation for arbitrary jobs.

A job is (key, chat messages, extra fields). Each finished job is appended to a
JSONL file as {"key": ..., **extra, "output": ...}; keys already in the file are
skipped, so a disconnected session only loses the current batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

Job = tuple[str, list[dict], dict]


def ensure_trailing_newline(path: Path) -> None:
    """If an interrupted run left a half-written last line, end it so that the
    next record starts on its own line."""
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as f:
            f.seek(-1, 2)
            last = f.read(1)
        if last != b"\n":
            with path.open("a", encoding="utf-8") as f:
                f.write("\n")


def read_outputs(path: str | Path) -> dict[str, dict]:
    """key -> record for every complete line in a jobs output file."""
    path = Path(path)
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records[rec["key"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue  # truncated last line from an interrupted run
    return records


def run_jobs(
    jobs: Iterable[Job],
    generate_fn: Callable[[list[list[dict]]], list[str]],
    out_path: str | Path,
    batch_size: int = 16,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Run every job whose key is not already in `out_path`; return how many ran."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(read_outputs(out_path))
    todo, seen = [], set()
    for key, messages, extra in jobs:
        if key not in done and key not in seen:
            seen.add(key)
            todo.append((key, messages, extra))

    finished = 0
    ensure_trailing_newline(out_path)
    with out_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            outputs = generate_fn([messages for _, messages, _ in batch])
            if len(outputs) != len(batch):
                raise RuntimeError(f"expected {len(batch)} outputs, got {len(outputs)}")
            for (key, _, extra), output in zip(batch, outputs):
                f.write(json.dumps({"key": key, **extra, "output": output}, ensure_ascii=False) + "\n")
            f.flush()
            finished += len(batch)
            if progress:
                progress(finished, len(todo))
    return finished
