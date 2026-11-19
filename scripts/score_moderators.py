"""Does each system's reply follow the user's moral profile? (Table 2)

    python scripts/score_moderators.py --judges-dir judges/large \
        --plain outputs/plain_test_7B_full.jsonl --agents outputs/agents_test_7B_rlall.jsonl \
        --system trained=outputs/aggregator_test_trained.jsonl trained_rag=outputs/aggregator_test_rag_trained.jsonl \
        --compare trained:trained_rag

Every input is a (question, moral profile) pair. Each reply is scored by the
five value judges (the same reward models as Table 1: probability that the
reply follows the value). For each reply:
  on-profile   mean judge probability over the values in the user's profile
  off-profile  mean over the other values
  gap          on - off: how much more the reply serves the user's values
               than values the user did not ask for
All systems are scored on exactly the same inputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mvrag.data import PAPER_VALUES
from mvrag.jobs import read_outputs
from mvrag.stats import paired_bootstrap

LABEL_PREFIX = re.compile(r"^\s*(?:A|Answer|Reply|Final reply)\s*:\s*", re.IGNORECASE)


def clean_output(text: str) -> str:
    return LABEL_PREFIX.sub("", text or "", count=1).strip()


def profile_of(key: str) -> tuple[int, ...]:
    return tuple(int(c) for c in key.rsplit("|", 1)[1])


def read_by_uid(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["uid"]: r for r in (json.loads(line) for line in f if line.strip())}


def summarize(probs: dict[str, list[float]], keys: list[str]) -> tuple[dict, list[float]]:
    """probs[value][i] for keys[i] -> summary numbers and the per-input on-profile scores."""
    on_rows, off_all, per_value = [], [], {}
    for i, key in enumerate(keys):
        profile = profile_of(key)
        on = [probs[v][i] for v, flag in zip(PAPER_VALUES, profile) if flag]
        off = [probs[v][i] for v, flag in zip(PAPER_VALUES, profile) if not flag]
        on_rows.append(sum(on) / len(on))
        off_all.extend(off)
    for j, v in enumerate(PAPER_VALUES):
        idx = [i for i, k in enumerate(keys) if profile_of(k)[j]]
        per_value[v] = sum(probs[v][i] for i in idx) / len(idx) if idx else None
    on_mean = sum(on_rows) / len(on_rows)
    off_mean = sum(off_all) / len(off_all) if off_all else None
    summary = {"on_profile": on_mean, "off_profile": off_mean,
               "gap": None if off_mean is None else on_mean - off_mean, "per_value": per_value}
    return summary, on_rows


def main() -> None:
    from mvrag.judges import load_judge, predict_probs

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", nargs="+", required=True, metavar="NAME=FILE",
                        help="output files from run_aggregator.py / run_baseline.py (keyed by question + profile)")
    parser.add_argument("--plain", default=None, help="plain-model answers (same reply for every profile)")
    parser.add_argument("--agents", default=None, help="advisor answers; adds best_agent (advisor of the first profile value)")
    parser.add_argument("--references", action="store_true",
                        help="also score MIC's human-written reply and chatbot reply for each input (reference levels)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--judges-dir", default="judges/large")
    parser.add_argument("--compare", nargs="*", default=[], metavar="A:B", help="significance of B minus A (on-profile)")
    parser.add_argument("--out", default="outputs/table2.json")
    args = parser.parse_args()

    systems = {name: read_outputs(path) for name, path in (s.split("=", 1) for s in args.system)}
    keys = sorted(set.intersection(*(set(o) for o in systems.values())))
    keys = [k for k in keys if any(profile_of(k))]
    plain = read_by_uid(args.plain) if args.plain else None
    agents = read_by_uid(args.agents) if args.agents else None
    keys = [k for k in keys if (plain is None or k.split("|")[0] in plain)
            and (agents is None or k.split("|")[0] in agents)]
    refs = {}
    if args.references:
        from mvrag.aggregator import profile_from_morals, profile_key
        from mvrag.data import load_split

        for ex in sorted(load_split(args.split, args.data_dir), key=lambda e: (e.uid, e.row)):
            k = profile_key(ex.uid, profile_from_morals(ex.morals))
            if k not in refs and ex.worker_answer and ex.answer:
                refs[k] = (ex.worker_answer, ex.answer)  # first row of this input
        keys = [k for k in keys if k in refs]
    if not keys:
        raise SystemExit("The files share no inputs.")

    texts: dict[str, list[str]] = {}
    if refs:
        texts["human_reply"] = [refs[k][0] for k in keys]
        texts["chatbot_reply"] = [refs[k][1] for k in keys]
    if plain is not None:
        texts["plain"] = [plain[k.split("|")[0]]["answers"]["plain"] for k in keys]
    if agents is not None:
        first = [PAPER_VALUES[profile_of(k).index(1)] for k in keys]
        texts["best_agent"] = [agents[k.split("|")[0]]["answers"][v] for k, v in zip(keys, first)]
    for name, outs in systems.items():
        texts[name] = [clean_output(outs[k]["output"]) for k in keys]

    probs = {name: {} for name in texts}
    for value in PAPER_VALUES:
        model, tokenizer = load_judge(Path(args.judges_dir) / value)
        for name, t in texts.items():
            probs[name][value] = predict_probs(model, tokenizer, t)
        del model
        print(f"  scored with the {value} judge")

    results, on_rows = {"n": len(keys), "systems": {}}, {}
    for name in texts:
        results["systems"][name], on_rows[name] = summarize(probs[name], keys)

    print(f"\ninputs (question + profile): {len(keys)}   judges: {args.judges_dir}\n")
    head = f"{'system':<14}{'on-profile':>11}{'off-profile':>12}{'gap':>8}" + "".join(f"{v:>11}" for v in PAPER_VALUES)
    print(head + "\n" + "(%)  per-value columns: inputs whose profile includes that value")
    for name, r in results["systems"].items():
        cells = "".join(f"{100 * r['per_value'][v]:>11.2f}" for v in PAPER_VALUES)
        off = f"{100 * r['off_profile']:>12.2f}" if r["off_profile"] is not None else f"{'':>12}"
        gap = f"{100 * r['gap']:>+8.2f}" if r["gap"] is not None else f"{'':>8}"
        print(f"{name:<14}{100 * r['on_profile']:>11.2f}{off}{gap}{cells}")

    uids = [k.split("|")[0] for k in keys]
    for pair in args.compare:
        a, b = pair.split(":", 1)
        t = paired_bootstrap(on_rows[a], on_rows[b], uids)
        results.setdefault("compare", {})[pair] = t
        print(f"  {b} - {a}: on-profile {100 * t['diff']:+.2f} points  "
              f"95% CI [{100 * t['ci95'][0]:+.2f}, {100 * t['ci95'][1]:+.2f}]  p = {t['p']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
