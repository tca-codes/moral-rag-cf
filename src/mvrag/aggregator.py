"""The Contextual Aggregator ("moderator").

Paper: the aggregator takes the question, the five moral agents' answers, and
the user's moral profile vector, and produces one answer; it is trained to match
the ground-truth answer. The paper does not publish the input format, so the
layout below is ours.

Two ways to use it:
  - prompt-only (no training): comparable to the paper's "Agg13Llama" baseline,
    where an LLM is simply asked to aggregate the agents' answers;
  - LoRA fine-tuned on MIC: the paper's CMVA-GS system (added in a later step).

Inputs are passed as plain dicts/lists so that the counterfactual experiments
can drop or swap any single agent answer or retrieved example.
"""

from __future__ import annotations

from mvrag.data import PAPER_VALUES

SYSTEM_PROMPT = (
    "You write the final reply to a question on an online forum. "
    "You are given candidate replies, each written from one moral perspective, "
    "and the moral profile of the person asking, which says which moral values "
    "matter to them (1 = matters, 0 = not emphasized). "
    "Write one reply that fits that profile, drawing mainly on the candidates "
    "whose values the profile emphasizes. "
    "Reply in the first person, in one or two short, casual sentences. "
    "Output only the reply."
)


def profile_from_morals(morals) -> tuple[int, ...]:
    """Five-entry 0/1 moral profile in PAPER_VALUES order.

    MIC also annotates a sixth foundation (liberty) that the paper does not
    use; it is ignored here, so a liberty-only row has an all-zero profile.
    """
    morals = set(morals or ())
    return tuple(1 if v in morals else 0 for v in PAPER_VALUES)


def profile_text(profile) -> str:
    return ", ".join(f"{v} = {int(x)}" for v, x in zip(PAPER_VALUES, profile))


def profile_key(uid: str, profile) -> str:
    """Cache key: the aggregator's input is fully determined by question + profile."""
    return f"{uid}|{''.join(str(int(x)) for x in profile)}"


def build_aggregator_messages(
    question: str,
    answers: dict[str, str],
    profile,
    retrieved: list[dict] | None = None,
) -> list[dict]:
    """Chat messages for the aggregator.

    answers:   {value: agent answer}; values may be missing (leave-one-out).
    profile:   five 0/1 numbers in PAPER_VALUES order.
    retrieved: optional list of {"question": ..., "answer": ...} examples (RAG).
    """
    lines = [f"Question: {question}", "", f"Moral profile: {profile_text(profile)}", "", "Candidate replies:"]
    shown = [v for v in PAPER_VALUES if answers.get(v)]
    if not shown:
        lines.append("(none)")
    for value in shown:
        lines.append(f"[{value}] {answers[value]}")
    if retrieved:
        lines += ["", "Similar past questions and the replies given:"]
        for i, ex in enumerate(retrieved, 1):
            lines.append(f"({i}) Q: {ex['question']} A: {ex['answer']}")
    lines += ["", "Final reply:"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
