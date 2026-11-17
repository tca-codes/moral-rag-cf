"""Counterfactual explanations for the aggregator (course item G15).

For one aggregator input (question, profile, five agent answers, optional
retrieved examples) we build controlled variants, regenerate the output for
each with greedy decoding, and compare it with the output for the full input:

  full            the unchanged input
  drop:<value>    that agent's answer removed            (which agents matter?)
  drop_ret:<i>    retrieved example i removed            (which evidence matters?)
  no_ret          all retrieved examples removed         (does retrieval matter?)
  swap_ret:<i>    retrieved example i replaced with one  (is it reading the content,
                  retrieved for a different input        or just the format?)

Greedy decoding makes the full output deterministic, so any change is caused by
the edit, not by sampling noise.
"""

from __future__ import annotations

import random

from mvrag.data import PAPER_VALUES


def make_variants(answers: dict, retrieved: list[dict] | None, swap_pool: list[dict] | None = None,
                  seed_key: str = "") -> dict[str, tuple[dict, list[dict] | None]]:
    """variant name -> (agent answers, retrieved examples) for one input."""
    variants: dict[str, tuple[dict, list[dict] | None]] = {"full": (dict(answers), retrieved)}
    for value in PAPER_VALUES:
        if answers.get(value):
            variants[f"drop:{value}"] = ({v: a for v, a in answers.items() if v != value}, retrieved)
    if retrieved:
        variants["no_ret"] = (dict(answers), [])
        rng = random.Random(f"swap|{seed_key}")
        own_questions = {r["question"] for r in retrieved}
        pool = [r for r in (swap_pool or []) if r["question"] not in own_questions]
        for i in range(len(retrieved)):
            variants[f"drop_ret:{i + 1}"] = (dict(answers), retrieved[:i] + retrieved[i + 1:])
            if pool:
                replacement = rng.choice(pool)
                variants[f"swap_ret:{i + 1}"] = (dict(answers), retrieved[:i] + [replacement] + retrieved[i + 1:])
    return variants


def variant_family(name: str) -> str:
    """'drop:care' -> 'drop', 'drop_ret:2' -> 'drop_ret', 'full' -> 'full'."""
    return name.split(":", 1)[0]


def flips(p_full: float, p_variant: float, threshold: float = 0.5) -> bool:
    """Did the judge's yes/no decision change between the two outputs?"""
    return (p_full >= threshold) != (p_variant >= threshold)
