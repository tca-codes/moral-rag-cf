"""Download MIC (if needed) and print its real schema. Run this FIRST.

    python scripts/inspect_mic.py --data-dir data/MIC

Compare the printed columns with FIELD_ALIASES in src/mvrag/data.py and fix any
mismatch before running anything else.
"""

from __future__ import annotations

import argparse
from collections import Counter

from mvrag.data import SPLIT_FILES, ensure_mic, group_by_question, read_records, to_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/MIC")
    args = parser.parse_args()

    root = ensure_mic(args.data_dir)
    print(f"MIC files found in: {root}\n")

    total_rows = 0
    for split, fname in SPLIT_FILES.items():
        records = read_records(root / fname)
        examples = [to_example(r, i) for i, r in enumerate(records)]
        total_rows += len(records)
        groups = group_by_question(examples)
        print(f"[{split}] rows={len(records)}  unique questions={len(groups)}")
        if split == "train":
            print("  raw columns:", sorted(records[0].keys()))
            print("  first raw record:")
            for k, v in records[0].items():
                print(f"    {k!r}: {str(v)[:100]!r}")
            counts = Counter(m for ex in examples for m in ex.morals)
            print("  foundation counts (parsed):", dict(counts))
            empty = sum(1 for ex in examples if not ex.question or not ex.answer)
            print(f"  rows with empty question/answer after parsing: {empty}")
        print()

    print(f"Total rows across splits: {total_rows}")
    print("(The CMVA-GS paper reports 91.0K/11.4K/11.4K samples, ~113.8K in total.)")


if __name__ == "__main__":
    main()
