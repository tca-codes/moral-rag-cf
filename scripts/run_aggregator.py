"""Run the prompt-only aggregator (moderator) over a MIC split.

    python scripts/run_aggregator.py --agents outputs/agents_test_7B_full.jsonl

Each MIC row is one sample: its question, its moral profile, and the five agent
answers for that question. Rows that share a question and a profile have the
same input, so each distinct (question, profile) pair is generated once.
Safe to re-run: finished pairs are skipped.
"""

from __future__ import annotations

import argparse
import json
import time

from mvrag.aggregator import build_aggregator_messages, profile_from_morals, profile_key
from mvrag.data import load_split
from mvrag.generate import generate_batch, load_model
from mvrag.jobs import run_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", default="outputs/agents_test_7B_full.jsonl")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="outputs/aggregator_test_prompt.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="only the first N distinct inputs (for a quick test)")
    parser.add_argument("--no-4bit", action="store_true", help="use fp16 instead of 4-bit")
    parser.add_argument("--adapter", default=None, help="folder of a trained LoRA adapter (trained aggregator)")
    args = parser.parse_args()

    agents = {}
    with open(args.agents, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                agents[rec["uid"]] = rec

    jobs, seen, skipped = [], set(), 0
    for ex in sorted(load_split(args.split, args.data_dir), key=lambda e: (e.uid, e.row)):
        if ex.uid not in agents:
            skipped += 1
            continue
        profile = profile_from_morals(ex.morals)
        key = profile_key(ex.uid, profile)
        if key in seen:
            continue
        seen.add(key)
        messages = build_aggregator_messages(ex.question, agents[ex.uid]["answers"], profile)
        jobs.append((key, messages, {"uid": ex.uid, "profile": list(profile)}))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} distinct (question, profile) inputs; {skipped} rows skipped (no agent answers yet)")

    model, tokenizer = load_model(args.model, four_bit=not args.no_4bit, adapter=args.adapter)
    start = time.time()

    def gen(chats):
        return generate_batch(model, tokenizer, chats, args.max_new_tokens)

    def progress(done: int, total: int) -> None:
        if done % (args.batch_size * 10) == 0 or done == total:
            print(f"  {done}/{total}  ({time.time() - start:.0f}s)")

    n = run_jobs(jobs, gen, args.out, args.batch_size, progress)
    print(f"Done. {n} new outputs written to {args.out}")


if __name__ == "__main__":
    main()
