"""Test every judge's real accuracy on SocialChem dev data (not the saved notes).

    python scripts/check_judges.py --judges-dirs judges/large judges/full

A judge file can be swapped for an older version (it happened with Colab's
Drive mount), while its metrics.json still shows the newer accuracy. This loads
each judge's actual weights, measures accuracy on up to --max-dev dev examples,
and flags a judge as BROKEN when it is no better than always guessing the most
common label, or as MISMATCH when it differs from its metrics.json.
Exit code 1 if any judge is BROKEN or missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mvrag.data import PAPER_VALUES
from mvrag.judges import binary_metrics, load_judge, predict_probs
from mvrag.socialchem import build_value_examples, ensure_socialchem, load_rows, subsample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges-dirs", nargs="+", required=True)
    parser.add_argument("--values", nargs="+", default=list(PAPER_VALUES), choices=list(PAPER_VALUES))
    parser.add_argument("--data-dir", default="data/socialchem")
    parser.add_argument("--max-dev", type=int, default=2000)
    args = parser.parse_args()

    rows = load_rows(ensure_socialchem(args.data_dir))
    dev = {v: subsample(build_value_examples(rows, v, "dev"), args.max_dev) for v in args.values}
    failed = False
    print(f"{'judge':<34}{'measured':>10}{'saved':>8}{'majority':>10}   status")
    for d in args.judges_dirs:
        for v in args.values:
            path = Path(d) / v
            name = f"{d}/{v}"
            if not (path / "model.safetensors").exists():
                print(f"{name:<34}{'-':>10}{'-':>8}{'-':>10}   MISSING")
                failed = True
                continue
            model, tok = load_judge(path)
            probs = predict_probs(model, tok, [t for t, _ in dev[v]])
            m = binary_metrics([y for _, y in dev[v]], [1 if p >= 0.5 else 0 for p in probs])
            spread = max(probs) - min(probs)
            saved = json.loads((path / "metrics.json").read_text())["accuracy"] if (path / "metrics.json").exists() else None
            if m["accuracy"] < m["majority_baseline"] + 0.02 or spread < 0.2:
                status = "BROKEN (no better than guessing)"
                failed = True
            elif saved is not None and abs(saved - m["accuracy"]) > 0.03:
                status = "MISMATCH with saved notes"
            else:
                status = "OK"
            saved_s = f"{saved:.3f}" if saved is not None else "-"
            print(f"{name:<34}{m['accuracy']:>10.3f}{saved_s:>8}{m['majority_baseline']:>10.3f}   {status}")
            del model
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
