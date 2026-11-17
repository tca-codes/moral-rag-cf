"""Analyse counterfactual variants (G15): which inputs drive the aggregator's reply?

    python scripts/analyze_counterfactuals.py --cf outputs/cf_rag.jsonl --judges-dir judges/best

For every variant it measures, against the full input's reply:
  changed     the reply text is different
  influence   1 - ROUGE-L(variant reply, full reply): how much the reply moved
  quality     change in ROUGE-L against MIC's human-written reply
  alignment   change in the judges' scores for the values in the profile
  flip        a judge's yes/no decision (p >= 0.5) for a profile value changed

Results: a summary table by variant type, agent influence split into agents
whose value is in the profile vs not, retrieved-example influence by rank, and
a file of concrete examples (largest changes) for the error analysis.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from mvrag.counterfactual import flips, variant_family
from mvrag.data import PAPER_VALUES, load_split
from mvrag.aggregator import profile_from_morals, profile_key
from mvrag.jobs import read_outputs


def main() -> None:
    from rouge_score import rouge_scorer

    from mvrag.judges import load_judge, predict_probs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cf", required=True)
    parser.add_argument("--judges-dir", default="judges/best")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--out", default=None)
    parser.add_argument("--examples", type=int, default=15, help="how many illustrative cases to save")
    args = parser.parse_args()
    out = Path(args.out or str(Path(args.cf).with_suffix("")) + "_analysis.json")

    records = list(read_outputs(args.cf).values())
    by_input: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in records:
        by_input[rec["input"]][rec["variant"]] = rec
    by_input = {k: v for k, v in by_input.items() if "full" in v}
    print(f"{len(by_input)} inputs with a full reply, {len(records)} replies in total")

    refs: dict[str, list[str]] = defaultdict(list)
    for ex in load_split(args.split, args.data_dir):
        key = profile_key(ex.uid, profile_from_morals(ex.morals))
        if key in by_input and ex.worker_answer:
            refs[key].append(ex.worker_answer)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def rl(a: str, b: str) -> float:
        return scorer.score(a, b)["rougeL"].fmeasure

    # judge probabilities: only for values in each input's profile
    probs: dict[tuple[str, str], float] = {}
    for value in PAPER_VALUES:
        todo = [(rec["key"], rec["output"]) for variants in by_input.values() for rec in variants.values()
                if rec["profile"][PAPER_VALUES.index(value)]]
        if not todo:
            continue
        model, tokenizer = load_judge(Path(args.judges_dir) / value)
        for (k, _), p in zip(todo, predict_probs(model, tokenizer, [t or "." for _, t in todo])):
            probs[(k, value)] = p
        del model

    rows = []
    for key, variants in by_input.items():
        full = variants["full"]
        desired = [v for v, on in zip(PAPER_VALUES, full["profile"]) if on]
        q_full = mean(rl(r, full["output"]) for r in refs[key]) if refs[key] else None
        for name, rec in variants.items():
            if name == "full":
                continue
            q_var = mean(rl(r, rec["output"]) for r in refs[key]) if refs[key] else None
            dropped = name.split(":", 1)[1] if name.startswith("drop:") else None
            rows.append({
                "input": key, "variant": name, "family": variant_family(name),
                "on_profile": (dropped in desired) if dropped else None,
                "changed": rec["output"].strip() != full["output"].strip(),
                "influence": 1 - rl(full["output"], rec["output"]),
                "quality_delta": (q_var - q_full) if q_full is not None else None,
                "alignment_delta": mean(probs[(rec["key"], v)] - probs[(full["key"], v)] for v in desired),
                "flip": any(flips(probs[(full["key"], v)], probs[(rec["key"], v)]) for v in desired),
                "full_output": full["output"], "variant_output": rec["output"],
            })

    def summarise(group: list[dict]) -> dict:
        q = [r["quality_delta"] for r in group if r["quality_delta"] is not None]
        return {
            "n": len(group),
            "changed_rate": mean(r["changed"] for r in group),
            "mean_influence": mean(r["influence"] for r in group),
            "mean_quality_delta": mean(q) if q else None,
            "mean_alignment_delta": mean(r["alignment_delta"] for r in group),
            "flip_rate": mean(r["flip"] for r in group),
        }

    def table(title: str, groups: dict[str, list[dict]]) -> dict:
        print(f"\n{title}")
        print(f"{'':<22}{'n':>6}{'changed':>9}{'influence':>11}{'quality Δ':>11}{'alignment Δ':>13}{'flips':>8}")
        result = {}
        for name, group in groups.items():
            if not group:
                continue
            s = result[name] = summarise(group)
            qd = f"{s['mean_quality_delta']:>+11.4f}" if s["mean_quality_delta"] is not None else f"{'-':>11}"
            print(f"{name:<22}{s['n']:>6}{s['changed_rate']:>9.2f}{s['mean_influence']:>11.3f}{qd}"
                  f"{s['mean_alignment_delta']:>+13.4f}{s['flip_rate']:>8.2f}")
        return result

    families = defaultdict(list)
    for r in rows:
        families[r["family"]].append(r)
    report = {"inputs": len(by_input)}
    report["by_variant_type"] = table("By variant type", dict(sorted(families.items())))
    report["agents_on_vs_off_profile"] = table("Removing one agent: is its value in the profile?", {
        "agent in profile": [r for r in rows if r["family"] == "drop" and r["on_profile"]],
        "agent not in profile": [r for r in rows if r["family"] == "drop" and r["on_profile"] is False],
    })
    report["by_agent"] = table("Removing one agent, by value",
                               {f"drop:{v}": [r for r in rows if r["variant"] == f"drop:{v}"] for v in PAPER_VALUES})
    ranks = sorted({r["variant"] for r in rows if r["family"] in ("drop_ret", "swap_ret")})
    if ranks:
        report["retrieval_by_rank"] = table("Retrieved examples, by rank (1 = most similar)",
                                            {v: [r for r in rows if r["variant"] == v] for v in ranks})

    examples = sorted([r for r in rows if r["flip"] or r["changed"]],
                      key=lambda r: (-r["flip"], -r["influence"]))[: args.examples]
    report["examples"] = examples
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved analysis and {len(examples)} example cases to {out}")


if __name__ == "__main__":
    main()
