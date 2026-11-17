"""Reward-based training of the moral agents.

Paper: each moral agent is the base LLM fine-tuned with PPO, using one value's
classifier as the reward (probability that the answer follows the value).

Ours: a simpler reward-based method with the same reward signal. For each
question the value-prompted model samples several candidate answers, the value's
judge scores them, and the best-scoring candidate is kept. The model is then
LoRA fine-tuned to produce those winners ("best-of-N" selection followed by
supervised fine-tuning; a form of reward-filtered policy improvement). Staying
close to the base model, which PPO enforces with a KL penalty, comes for free
because every training target is one of the base model's own samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from mvrag.agents import build_agent_messages
from mvrag.jobs import ensure_trailing_newline, read_outputs


def select_best(candidates: list[str], scores: list[float]) -> tuple[str, float]:
    """Highest-scoring non-empty candidate (first one on ties)."""
    best_i = None
    for i, (cand, score) in enumerate(zip(candidates, scores)):
        if cand.strip() and (best_i is None or score > scores[best_i]):
            best_i = i
    if best_i is None:
        return "", 0.0
    return candidates[best_i], scores[best_i]


def sample_and_score(
    questions: list[tuple[str, str]],
    value: str,
    sample_fn: Callable[[list[list[dict]]], list[list[str]]],
    score_fn: Callable[[list[str]], list[float]],
    out_path: str | Path,
    batch_size: int = 16,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Sample candidates for each question, score them with the value's judge, save all.

    Resumable: questions already in `out_path` are skipped.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(read_outputs(out_path))
    todo = [(uid, q) for uid, q in questions if uid not in done]

    finished = 0
    ensure_trailing_newline(out_path)
    with out_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            candidates = sample_fn([build_agent_messages(value, q) for _, q in batch])
            if len(candidates) != len(batch):
                raise RuntimeError(f"expected {len(batch)} candidate lists, got {len(candidates)}")
            flat = [c for cands in candidates for c in cands]
            flat_scores = score_fn([c if c.strip() else "." for c in flat])
            pos = 0
            for (uid, q), cands in zip(batch, candidates):
                scores = [float(x) for x in flat_scores[pos : pos + len(cands)]]
                pos += len(cands)
                record = {"key": uid, "uid": uid, "question": q, "value": value,
                          "candidates": cands, "scores": scores}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            finished += len(batch)
            if progress:
                progress(finished, len(todo))
    return finished


def load_selections(path: str | Path) -> list[dict]:
    """One dict per question: the winning candidate plus summary scores."""
    out = []
    for rec in read_outputs(path).values():
        best, best_score = select_best(rec["candidates"], rec["scores"])
        if not best:
            continue
        out.append({
            "uid": rec["uid"], "question": rec["question"], "value": rec["value"],
            "best": best, "best_score": best_score,
            "mean_score": sum(rec["scores"]) / len(rec["scores"]),
        })
    return sorted(out, key=lambda r: r["uid"])


def build_agent_training_examples(selections: list[dict], value: str) -> list[dict]:
    """Training samples for one agent: its value prompt + question -> winning answer."""
    return [{"messages": build_agent_messages(value, s["question"]), "target": s["best"]} for s in selections]


def reward_summary(selections: list[dict]) -> dict:
    n = len(selections)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean_candidate_reward": sum(s["mean_score"] for s in selections) / n,
        "mean_best_reward": sum(s["best_score"] for s in selections) / n,
    }
