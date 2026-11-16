"""Sanity-check ROUGE of each moral agent's answers against MIC replies.

    python scripts/eval_rouge.py --agents outputs/agents_test.jsonl --split test

A question can have several reference replies (different chatbots/annotations);
for each metric we take the best score over its references. This is only a first
sanity number: the paper's real metric is the aggregator's ROUGE against the
ground-truth answer, which comes later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvrag.agents import VALUES
from mvrag.data import group_by_question, load_split

METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def main() -> None:
    from rouge_score import rouge_scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", default="outputs/agents_test.jsonl")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--out", default="outputs/rouge_agents.json")
    args = parser.parse_args()

    groups = group_by_question(load_split(args.split, args.data_dir))
    scorer = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=True)

    sums = {v: {m: 0.0 for m in METRICS} for v in VALUES}
    n = 0
    with open(args.agents, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            refs = list({ex.answer for ex in groups.get(rec["uid"], []) if ex.answer})
            if not refs:
                continue
            n += 1
            for v in VALUES:
                for m in METRICS:
                    sums[v][m] += max(scorer.score(r, rec["answers"][v])[m].fmeasure for r in refs)

    if n == 0:
        raise SystemExit("No scorable questions found; check --agents and --split.")

    results = {v: {m: sums[v][m] / n for m in METRICS} for v in VALUES}
    print(f"questions scored: {n}\n")
    print(f"{'agent':<10}" + "".join(f"{m:>11}" for m in METRICS))
    for v in VALUES:
        print(f"{v:<10}" + "".join(f"{results[v][m]:>11.4f}" for m in METRICS))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n": n, "results": results}, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
