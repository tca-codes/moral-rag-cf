"""Build one judge set from the most accurate judge for each value.

    python scripts/pick_judges.py --candidates judges/full judges/large --out judges/best

Each value's judge is trained and scored independently on the same SocialChem
dev examples, so picking per value is fair. Note that dev accuracy is used both
to pick and to report, so the reported accuracy of the picked set is slightly
optimistic; the test answers being scored (MIC) are never involved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mvrag.data import PAPER_VALUES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", default=["judges/full", "judges/large"])
    parser.add_argument("--out", default="judges/best")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    picked = {}
    print(f"{'judge':<10}" + "".join(f"{Path(c).name:>10}" for c in args.candidates) + "   picked")
    for value in PAPER_VALUES:
        scores = {}
        for cand in args.candidates:
            metrics_file = Path(cand) / value / "metrics.json"
            if metrics_file.exists():
                scores[cand] = json.loads(metrics_file.read_text())
        if not scores:
            raise SystemExit(f"No trained judge for {value} in {args.candidates}")
        best = max(scores, key=lambda c: scores[c]["accuracy"])
        shutil.copytree(Path(best) / value, out / value)
        picked[value] = {**scores[best], "source": best}
        row = "".join(f"{scores[c]['accuracy']:>10.3f}" if c in scores else f"{'-':>10}" for c in args.candidates)
        print(f"{value:<10}{row}   {best}")
    (out / "metrics.json").write_text(json.dumps(picked, indent=2))
    mean = sum(m["accuracy"] for m in picked.values()) / len(picked)
    print(f"\nMean accuracy of {out}: {mean:.4f}")


if __name__ == "__main__":
    main()
