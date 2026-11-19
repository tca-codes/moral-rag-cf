"""Paired significance test for comparing two systems on the same test rows."""

from __future__ import annotations

import random


def paired_bootstrap(a: list[float], b: list[float], groups: list[str] | None = None,
                     n_resamples: int = 10_000, seed: int = 0) -> dict:
    """Is mean(b) - mean(a) different from zero?

    Rows are resampled with replacement, keeping each row's two scores together
    (paired). With `groups` (e.g. the question uid), whole groups are resampled,
    because rows that share a question are not independent.
    Returns the observed difference, a 95% interval and a two-sided p-value.
    """
    if len(a) != len(b) or not a:
        raise ValueError("a and b must be non-empty and the same length")
    groups = groups or [str(i) for i in range(len(a))]
    by_group: dict[str, list[float]] = {}
    for g, x, y in zip(groups, a, b):
        by_group.setdefault(g, []).append(y - x)
    sums = [(sum(d), len(d)) for d in by_group.values()]
    observed = sum(s for s, _ in sums) / sum(n for _, n in sums)

    rng = random.Random(seed)
    k = len(sums)
    diffs = []
    for _ in range(n_resamples):
        s = n = 0
        for _ in range(k):
            gs, gn = sums[rng.randrange(k)]
            s += gs
            n += gn
        diffs.append(s / n)
    diffs.sort()
    low, high = diffs[int(0.025 * n_resamples)], diffs[int(0.975 * n_resamples) - 1]
    # two-sided p: how often the resampled difference falls on the other side of zero
    other_side = sum(1 for d in diffs if (d <= 0 if observed > 0 else d >= 0))
    p = min(1.0, 2 * (other_side + 1) / (n_resamples + 1))
    return {"diff": observed, "ci95": [low, high], "p": p, "rows": len(a), "groups": k,
            "resamples": n_resamples}
