"""Retrieve similar training examples for every aggregator input (RAG, item G11).

    python scripts/build_retrieval.py --split test
    python scripts/build_retrieval.py --split train

Writes outputs/retrieved_<split>.jsonl: one line per distinct (question, profile)
input, keyed like the aggregator outputs, with k retrieved examples from the MIC
training split. The input's own question is never retrieved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvrag.aggregator import profile_from_morals, profile_key
from mvrag.data import load_split
from mvrag.retrieval import Retriever, sentence_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--out", default=None)
    parser.add_argument("--chunk", type=int, default=2048)
    args = parser.parse_args()
    out = Path(args.out or f"outputs/retrieved_{args.split}.jsonl")

    embed = sentence_embedder(args.model)
    retriever = Retriever(load_split("train", args.data_dir), embed)
    print(f"Retrieval pool: {len(retriever.entries)} training questions")

    queries, seen = [], set()
    for ex in sorted(load_split(args.split, args.data_dir), key=lambda e: (e.uid, e.row)):
        profile = profile_from_morals(ex.morals)
        key = profile_key(ex.uid, profile)
        if key not in seen:
            seen.add(key)
            queries.append((key, ex.uid, ex.question, profile))
    print(f"{len(queries)} distinct (question, profile) inputs in split={args.split}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for start in range(0, len(queries), args.chunk):
            chunk = queries[start : start + args.chunk]
            results = retriever.search_many([(u, q, p) for _, u, q, p in chunk], k=args.k)
            for (key, uid, _, profile), retrieved in zip(chunk, results):
                f.write(json.dumps({"key": key, "uid": uid, "profile": list(profile),
                                    "retrieved": retrieved}, ensure_ascii=False) + "\n")
            print(f"  {min(start + args.chunk, len(queries))}/{len(queries)}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
