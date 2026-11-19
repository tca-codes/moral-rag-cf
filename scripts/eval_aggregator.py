"""ROUGE for the aggregator, next to simple baselines (small-scale Figure 2).

    python scripts/eval_aggregator.py --aggregator outputs/aggregator_test_prompt.jsonl \
        --agents outputs/agents_test_7B_full.jsonl --plain outputs/plain_test_7B_full.jsonl

Every MIC test row is one sample. Each system's output for that row is scored
against the row's reference reply. The paper does not say whether its "ground
truth answer" is MIC's human-written reply (`worker_answer`) or the chatbot
reply (`A`), so both are reported.

Systems compared (all on exactly the same rows):
  plain       the model with no moral instruction (paper: "PT-model")
  best_agent  the single moral agent matching the row's profile (first value set)
  aggregator  the moderator's output
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mvrag.aggregator import profile_from_morals, profile_key
from mvrag.data import PAPER_VALUES, load_split
from mvrag.jobs import read_outputs
from mvrag.stats import paired_bootstrap

METRICS = ("rouge1", "rouge2", "rougeL", "rougeLsum")
LABEL_PREFIX = re.compile(r"^\s*(?:A|Answer|Reply|Final reply)\s*:\s*", re.IGNORECASE)


def clean_output(text: str) -> str:
    """Drop a leading answer label copied from the prompt (e.g. "A: ..."); applied to every system."""
    return LABEL_PREFIX.sub("", text or "", count=1).strip()


def read_by_uid(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["uid"]: r for r in (json.loads(line) for line in f if line.strip())}


def main() -> None:
    from rouge_score import rouge_scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregator", default="outputs/aggregator_test_prompt.jsonl")
    parser.add_argument("--agents", default="outputs/agents_test_7B_full.jsonl")
    parser.add_argument("--plain", default="outputs/plain_test_7B_full.jsonl",
                        help="plain-model answers; pass \"\" to leave the plain model out (e.g. on val)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--out", default="outputs/rouge_aggregator_prompt.json")
    parser.add_argument("--extra", nargs="*", default=[], metavar="NAME=FILE",
                        help="more aggregator output files to score on the same rows, e.g. trained=outputs/aggregator_test_trained.jsonl")
    parser.add_argument("--compare", nargs="*", default=[], metavar="A:B",
                        help="significance of B minus A on ROUGE-L (paired bootstrap over questions), e.g. trained:trained_rag")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    agg, agents = read_outputs(args.aggregator), read_by_uid(args.agents)
    plain = read_by_uid(args.plain) if args.plain else None
    scorer = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=True)
    extra = {name: read_outputs(path) for name, path in (item.split("=", 1) for item in args.extra)}
    systems = (("plain",) if plain is not None else ()) + ("best_agent", "aggregator", *extra)
    refs = {"worker_answer": "human-written reply (worker_answer)", "answer": "chatbot reply (A)"}
    sums = {r: {s: {m: 0.0 for m in METRICS} for s in systems} for r in refs}
    counts = {r: 0 for r in refs}
    per_row = {r: {s: [] for s in systems} for r in refs}  # ROUGE-L per row, for --compare
    row_uids = {r: [] for r in refs}

    for ex in load_split(args.split, args.data_dir):
        profile = profile_from_morals(ex.morals)
        key = profile_key(ex.uid, profile)
        if key not in agg or ex.uid not in agents or (plain is not None and ex.uid not in plain):
            continue
        if any(key not in outputs_ for outputs_ in extra.values()):
            continue  # keep every system on exactly the same rows
        own = next((v for v, on in zip(PAPER_VALUES, profile) if on), None)
        if own is None:
            continue  # no profile among the five values (e.g. liberty-only rows)
        outputs = {
            **({"plain": plain[ex.uid]["answers"]["plain"]} if plain is not None else {}),
            "best_agent": agents[ex.uid]["answers"][own],
            "aggregator": agg[key]["output"],
            **{name: outputs_[key]["output"] for name, outputs_ in extra.items()},
        }
        for ref_name in refs:
            reference = getattr(ex, ref_name)
            if not reference:
                continue
            counts[ref_name] += 1
            row_uids[ref_name].append(ex.uid)
            for system, text in outputs.items():
                scores = scorer.score(reference, clean_output(text))
                for m in METRICS:
                    sums[ref_name][system][m] += scores[m].fmeasure
                per_row[ref_name][system].append(scores["rougeL"].fmeasure)

    results = {}
    for ref_name, label in refs.items():
        n = counts[ref_name]
        if n == 0:
            print(f"\nNo rows could be scored against {label}.")
            continue
        results[ref_name] = {"n": n, **{s: {m: sums[ref_name][s][m] / n for m in METRICS} for s in systems}}
        print(f"\nReference: {label}   rows scored: {n}")
        width = max(12, *(len(s) + 2 for s in systems))
        print(f"{'system':<{width}}" + "".join(f"{m:>11}" for m in METRICS))
        for s in systems:
            print(f"{s:<{width}}" + "".join(f"{results[ref_name][s][m]:>11.4f}" for m in METRICS))
        for pair in args.compare:
            a, b = pair.split(":", 1)
            if a not in systems or b not in systems:
                raise SystemExit(f"--compare {pair}: both names must be among {systems}")
            t = paired_bootstrap(per_row[ref_name][a], per_row[ref_name][b], row_uids[ref_name],
                                 n_resamples=args.resamples)
            results[ref_name].setdefault("compare", {})[pair] = t
            print(f"  {b} - {a}: ROUGE-L {t['diff']:+.4f}  95% CI [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]  "
                  f"p = {t['p']:.4f}  ({t['groups']} questions resampled)")

    if not results:
        raise SystemExit("Nothing scored; check the input files.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
