"""The paper's prompted baselines (Figure 2), rebuilt from its description.

Paper, section 5.2:
  (2) Llama-13b / Llama-7b: "The prompt defines the desired morals, includes 5
      pairs (i.e., questions and answers) of examples per desired moral taken
      from the MIC (test) data, and requests a response that follows the defined
      moral values."
  (3) Agg13Llama: Llama-2-13b-chat, "the morals are again defined but the
      examples are the results of passing the user question through the
      corresponding learned Moral Agents. The prompt asks the model to aggregate
      these answers."

The exact prompt wording is not published, so the wording here is ours.
(3) is our aggregator prompt (mvrag.aggregator) run with Llama-2-13b-chat.
Few-shot examples use MIC's human-written reply (worker_answer), the same
reference we treat as ground truth, and never come from the question being
answered.
"""

from __future__ import annotations

import random

from mvrag.agents import VALUE_DESCRIPTIONS
from mvrag.data import PAPER_VALUES


def example_pool(examples) -> dict[str, list[tuple[str, str, str]]]:
    """value -> sorted list of (uid, question, worker_answer) usable as few-shot examples."""
    pool: dict[str, dict[tuple[str, str], str]] = {v: {} for v in PAPER_VALUES}
    for ex in examples:
        if not ex.worker_answer or not ex.question:
            continue
        for value in ex.morals:
            if value in pool:
                pool[value].setdefault((ex.uid, ex.worker_answer), ex.question)
    return {v: sorted((uid, q, a) for (uid, a), q in items.items()) for v, items in pool.items()}


def pick_examples(pool, value: str, exclude_uid: str, k: int, seed_key: str) -> list[tuple[str, str]]:
    """k (question, answer) examples for one value, deterministic per seed_key, never from exclude_uid."""
    candidates = [(q, a) for uid, q, a in pool.get(value, []) if uid != exclude_uid]
    rng = random.Random(f"{seed_key}|{value}")
    return rng.sample(candidates, min(k, len(candidates)))


def build_fewshot_messages(question: str, profile, pool, uid: str, k: int = 5) -> list[dict]:
    """Paper baseline (2): desired morals defined, k examples per desired moral."""
    desired = [v for v, on in zip(PAPER_VALUES, profile) if on]
    system = (
        "You reply to questions on an online forum. Your reply must follow these moral values:\n"
        + "\n".join(f"- {v}: {VALUE_DESCRIPTIONS[v]}" for v in desired)
        + "\nReply in the first person, in one or two short, casual sentences. Output only the reply."
    )
    lines = ["Here are example questions with replies that follow these values."]
    for value in desired:
        lines.append(f"\nExamples for {value}:")
        for q, a in pick_examples(pool, value, uid, k, uid):
            lines.append(f"Q: {q}\nA: {a}")
    lines += ["", "Now reply to this question in the same way.", f"Q: {question}", "A:"]
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]
