"""Score the five advisors' answers with the five judges.

    python scripts/score_agents.py --agents outputs/agents_test_v2.jsonl

Prints a 5x5 table: rows are advisors, columns are judges, each cell is the
average probability that the advisor's answers follow the judge's value.
If the advisors were truly different, each one would score highest with its own
judge (the diagonal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvrag.agents import VALUES
from mvrag.judges import load_judge, predict_probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", default="outputs/agents_test_v2.jsonl")
    parser.add_argument("--judges-dir", default="judges")
    parser.add_argument("--out", default="outputs/judge_scores.json")
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.agents, encoding="utf-8") if line.strip()]
    if not records:
        raise SystemExit(f"No records in {args.agents}")

    # table[advisor][judge] = mean P(follows judge's value)
    table = {a: {} for a in VALUES}
    for judge in VALUES:
        model, tokenizer = load_judge(Path(args.judges_dir) / judge)
        for advisor in VALUES:
            probs = predict_probs(model, tokenizer, [r["answers"][advisor] for r in records])
            table[advisor][judge] = sum(probs) / len(probs)
        del model

    print(f"questions: {len(records)}   rows = advisors, columns = judges\n")
    print(f"{'advisor':<11}" + "".join(f"{j:>11}" for j in VALUES))
    for a in VALUES:
        print(f"{a:<11}" + "".join(f"{table[a][j]:>11.3f}" for j in VALUES))

    own = sum(table[v][v] for v in VALUES) / len(VALUES)
    others = sum(table[a][j] for a in VALUES for j in VALUES if a != j) / (len(VALUES) * (len(VALUES) - 1))
    spread = {j: max(table[a][j] for a in VALUES) - min(table[a][j] for a in VALUES) for j in VALUES}
    print(f"\nown judge (diagonal) mean:  {own:.3f}")
    print(f"other judges mean:          {others:.3f}")
    print("spread across advisors per judge: " + ", ".join(f"{j} {s:.3f}" for j, s in spread.items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"n": len(records), "table": table, "own_mean": own, "others_mean": others, "spread": spread}, indent=2)
    )
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
