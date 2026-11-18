"""Re-score saved reward candidates with a given set of judges (no new generation).

    python scripts/rescore_candidates.py --reward-dir outputs/reward --judges-dir judges/large

Each value's candidates are re-scored by that value's judge, so every candidate
in the reward data is scored by the same final judges. Files are rewritten
atomically (written to a temporary file, then renamed). Prints the reward
summary before and after.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mvrag.data import PAPER_VALUES
from mvrag.jobs import read_outputs
from mvrag.reward import load_selections, reward_summary


def rescore_file(path: Path, score_fn, batch: int = 4096) -> int:
    records = sorted(read_outputs(path).values(), key=lambda r: r["uid"])
    texts = [c if c.strip() else "." for r in records for c in r["candidates"]]
    scores = []
    for i in range(0, len(texts), batch):
        scores.extend(float(s) for s in score_fn(texts[i : i + batch]))
    pos = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            n = len(r["candidates"])
            r["scores"] = scores[pos : pos + n]
            pos += n
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return len(records)


def main() -> None:
    from mvrag.judges import load_judge, predict_probs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-dir", default="outputs/reward")
    parser.add_argument("--judges-dir", default="judges/large")
    parser.add_argument("--values", nargs="+", default=list(PAPER_VALUES), choices=list(PAPER_VALUES))
    args = parser.parse_args()

    summary = {}
    print(f"{'value':<11}{'questions':>10}{'before: all / best':>22}{'after: all / best':>22}")
    for v in args.values:
        path = Path(args.reward_dir) / f"{v}.jsonl"
        before = reward_summary(load_selections(path))
        model, tok = load_judge(Path(args.judges_dir) / v)
        n = rescore_file(path, lambda texts: predict_probs(model, tok, texts, batch_size=256))
        after = summary[v] = reward_summary(load_selections(path))
        del model
        print(f"{v:<11}{n:>10}{before['mean_candidate_reward']:>11.3f} / {before['mean_best_reward']:.3f}"
              f"{after['mean_candidate_reward']:>11.3f} / {after['mean_best_reward']:.3f}")
    (Path(args.reward_dir) / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
