"""Train the five value judges (the paper's reward models) on Social Chemistry 101.

    python scripts/train_judges.py

One classifier per moral value. Each is trained on actions tagged with that
value, labeled 0 if judged morally negative and 1 otherwise (as in the paper).
Judges that already exist in --judges-dir are skipped, so it is safe to re-run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mvrag.data import PAPER_VALUES
from mvrag.judges import train_judge
from mvrag.socialchem import build_value_examples, ensure_socialchem, load_rows, subsample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/socialchem")
    parser.add_argument("--judges-dir", default="judges")
    parser.add_argument("--base-model", default="distilroberta-base")
    parser.add_argument("--values", nargs="+", default=list(PAPER_VALUES), choices=list(PAPER_VALUES))
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-dev", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="retrain judges that already exist")
    args = parser.parse_args()

    tsv = ensure_socialchem(args.data_dir)
    rows = load_rows(tsv)
    print(f"SocialChem: {len(rows)} rows from {tsv}")
    print("  split counts:", dict(Counter(r.get("split") for r in rows)))

    all_metrics = {}
    for value in args.values:
        out_dir = Path(args.judges_dir) / value
        metrics_file = out_dir / "metrics.json"
        if metrics_file.exists() and not args.force:
            all_metrics[value] = json.loads(metrics_file.read_text())
            print(f"[{value}] already trained, skipping")
            continue
        train = subsample(build_value_examples(rows, value, "train"), args.max_train)
        dev = subsample(build_value_examples(rows, value, "dev"), args.max_dev)
        if not train or not dev:
            raise SystemExit(f"[{value}] no examples found; check the SocialChem columns")
        pos = sum(y for _, y in train) / len(train)
        print(f"[{value}] train={len(train)} dev={len(dev)}  share labeled good={pos:.2f}")
        all_metrics[value] = train_judge(train, dev, out_dir, args.base_model, args.epochs)

    print(f"\n{'judge':<10}{'accuracy':>10}{'macro_f1':>10}{'majority':>10}{'n_dev':>8}")
    for value, m in all_metrics.items():
        print(f"{value:<10}{m['accuracy']:>10.3f}{m['macro_f1']:>10.3f}{m['majority_baseline']:>10.3f}{m['n']:>8}")
    print("\n('majority' = accuracy of always guessing the most common label; a useful judge beats it.)")
    Path(args.judges_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.judges_dir) / "metrics.json").write_text(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
