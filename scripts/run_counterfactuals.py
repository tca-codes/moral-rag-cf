"""Generate counterfactual variants of aggregator inputs (G15).

    python scripts/run_counterfactuals.py --agents outputs/agents_test_7B_rl.jsonl \
        --retrieved outputs/retrieved_test.jsonl --adapter outputs/adapters/aggregator_rag \
        --n 500 --out outputs/cf_rag.jsonl

Picks N test inputs (seeded random sample of distinct question+profile pairs
whose profile has at least one of the five values), builds every variant from
mvrag.counterfactual, and generates the aggregator's reply for each. Resumable.
Analyse with scripts/analyze_counterfactuals.py.
"""

from __future__ import annotations

import argparse
import json
import random
import time

from mvrag.aggregator import build_aggregator_messages, profile_from_morals, profile_key
from mvrag.counterfactual import make_variants
from mvrag.data import load_split
from mvrag.generate import generate_batch, load_model
from mvrag.jobs import read_outputs, run_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", required=True)
    parser.add_argument("--retrieved", default=None, help="omit to test agent answers only")
    parser.add_argument("--adapter", default=None, help="trained aggregator adapter; omit for the prompt-only aggregator")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    agents = {}
    with open(args.agents, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                agents[rec["uid"]] = rec
    retrieved = read_outputs(args.retrieved) if args.retrieved else None

    inputs, seen = [], set()
    for ex in sorted(load_split(args.split, args.data_dir), key=lambda e: (e.uid, e.row)):
        profile = profile_from_morals(ex.morals)
        key = profile_key(ex.uid, profile)
        if key in seen or not any(profile) or ex.uid not in agents:
            continue
        if retrieved is not None and key not in retrieved:
            continue
        seen.add(key)
        inputs.append((key, ex.uid, ex.question, profile))
    random.Random(args.seed).shuffle(inputs)
    inputs = sorted(inputs[: args.n])
    swap_pool = [r for rec in (retrieved or {}).values() for r in rec["retrieved"]]

    jobs = []
    for key, uid, question, profile in inputs:
        context = retrieved[key]["retrieved"] if retrieved is not None else None
        for name, (answers, ctx) in make_variants(agents[uid]["answers"], context, swap_pool, key).items():
            messages = build_aggregator_messages(question, answers, profile, ctx)
            jobs.append((f"{key}#{name}", messages, {"input": key, "uid": uid, "profile": list(profile),
                                                     "variant": name}))
    print(f"{len(inputs)} inputs, {len(jobs)} generations "
          f"({'with' if retrieved is not None else 'without'} retrieval, "
          f"{'trained' if args.adapter else 'prompt-only'} aggregator)")

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
