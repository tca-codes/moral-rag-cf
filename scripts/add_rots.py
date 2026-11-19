"""RAG v2: add each retrieved example's rule of thumb (RoT) to a retrieval file.

    python scripts/add_rots.py --retrieved outputs/retrieved_test.jsonl --out outputs/retrieved_rot_test.jsonl

The retrieved examples stay exactly the same (same questions, same replies, same
order); only their MIC rule of thumb and its moral foundations are added. So any
difference between RAG and RAG v2 comes from the moral rule alone.

MIC rows are (question, chatbot reply, rule of thumb, human reply); a retrieved
example is one (question uid, human reply). If several rows share it, the rule
of thumb whose foundations overlap the input's profile most is used (ties: the
first row).
"""

from __future__ import annotations

import argparse
import json

from mvrag.data import PAPER_VALUES, load_split


def rot_index(rows) -> dict[tuple[str, str], list[tuple[str, list[str]]]]:
    """(uid, human reply) -> [(rule of thumb, its foundations)] in row order."""
    index: dict[tuple[str, str], list[tuple[str, list[str]]]] = {}
    for ex in sorted(rows, key=lambda e: (e.uid, e.row)):
        if ex.rot and ex.worker_answer:
            morals = [m for m in PAPER_VALUES if m in set(ex.morals or ())]
            index.setdefault((ex.uid, ex.worker_answer), []).append((ex.rot, morals))
    return index


def pick_rot(options: list[tuple[str, list[str]]], profile) -> tuple[str, list[str]]:
    desired = {v for v, on in zip(PAPER_VALUES, profile) if on}
    return max(options, key=lambda o: len(set(o[1]) & desired))  # max keeps the first on ties


def add_rots(record: dict, index) -> tuple[dict, int]:
    missing = 0
    items = []
    for item in record["retrieved"]:
        options = index.get((item["uid"], item["answer"]))
        if options:
            rot, morals = pick_rot(options, record["profile"])
            item = {**item, "rot": rot, "rot_morals": morals}
        else:
            missing += 1
        items.append(item)
    return {**record, "retrieved": items}, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieved", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default="data/MIC")
    args = parser.parse_args()

    index = rot_index(load_split("train", args.data_dir))  # the retrieval pool is the training split
    n = missing = 0
    with open(args.retrieved, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record, m = add_rots(json.loads(line), index)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            missing += m
    print(f"{n} inputs written to {args.out}; {missing} retrieved examples had no rule of thumb")
    if missing:
        print("Note: those examples are shown without a rule of thumb")


if __name__ == "__main__":
    main()
