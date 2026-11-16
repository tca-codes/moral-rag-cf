"""Run the five moral agents over a set of questions, resumably.

Output is JSONL, one line per question:
    {"uid": ..., "question": ..., "answers": {"care": ..., "fairness": ..., ...}}
Lines are appended as soon as a chunk finishes, and questions already present in
the output file are skipped, so a Colab disconnect only loses the current chunk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from mvrag.agents import VALUES, build_agent_messages

GenerateFn = Callable[[list[list[dict]]], list[str]]


def load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                continue  # ignore a truncated last line from an interrupted run
    return done


def run_agents(
    questions: list[tuple[str, str]],
    generate_fn: GenerateFn,
    out_path: str | Path,
    questions_per_chunk: int = 2,
    progress: Callable[[int, int], None] | None = None,
    builders: dict[str, Callable[[str], list[dict]]] | None = None,
) -> int:
    """Generate answers for every (uid, question) not already in `out_path`.

    `generate_fn` maps a list of chats to a list of replies (same order).
    `builders` maps a role name to a function question -> chat messages.
    Returns the number of questions newly processed.
    """
    if builders is None:  # default: the five moral agents
        builders = {v: (lambda q, v=v: build_agent_messages(v, q)) for v in VALUES}
    names = list(builders)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [(uid, q) for uid, q in questions if uid not in done]

    processed = 0
    with out_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), questions_per_chunk):
            chunk = todo[start : start + questions_per_chunk]
            chats = [builders[name](q) for _, q in chunk for name in names]
            replies = generate_fn(chats)
            if len(replies) != len(chats):
                raise RuntimeError(f"expected {len(chats)} replies, got {len(replies)}")
            for i, (uid, q) in enumerate(chunk):
                block = replies[i * len(names) : (i + 1) * len(names)]
                record = {"uid": uid, "question": q, "answers": dict(zip(names, block))}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            processed += len(chunk)
            if progress:
                progress(processed, len(todo))
    return processed
