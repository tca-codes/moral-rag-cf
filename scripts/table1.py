"""Recreate the paper's Table 1 at small scale.

    python scripts/table1.py --plain outputs/plain_test.jsonl --agents outputs/agents_test.jsonl

For each moral value, compares the probability (according to that value's judge)
that answers follow the value, for:
  - the plain model with no value instruction (the paper's "PT model"), and
  - the moral agent for that value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvrag.agents import VALUES
from mvrag.judges import load_judge, predict_probs


def read_jsonl(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["uid"]: r for r in (json.loads(line) for line in f if line.strip())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", default="outputs/plain_test.jsonl")
    parser.add_argument("--agents", default="outputs/agents_test.jsonl")
    parser.add_argument("--judges-dir", default="judges")
    parser.add_argument("--out", default="outputs/table1.json")
    args = parser.parse_args()

    plain, agents = read_jsonl(args.plain), read_jsonl(args.agents)
    uids = sorted(set(plain) & set(agents))
    if not uids:
        raise SystemExit("The two files share no questions.")
    plain_texts = [plain[u]["answers"]["plain"] for u in uids]

    rows = {}
    for value in VALUES:
        model, tokenizer = load_judge(Path(args.judges_dir) / value)
        p_plain = predict_probs(model, tokenizer, plain_texts)
        p_agent = predict_probs(model, tokenizer, [agents[u]["answers"][value] for u in uids])
        rows[value] = {"plain": sum(p_plain) / len(uids), "agent": sum(p_agent) / len(uids)}
        del model

    print(f"questions: {len(uids)}\n")
    print(f"{'moral value':<13}{'plain model':>13}{'moral agent':>13}{'change':>9}")
    for value in VALUES:
        r = rows[value]
        print(f"{value:<13}{r['plain']:>12.2%} {r['agent']:>12.2%} {r['agent'] - r['plain']:>+8.2%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n": len(uids), "rows": rows}, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
