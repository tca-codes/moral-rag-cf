"""Run one of the paper's prompted baselines over a MIC split.

    # paper's Llama-7b / Llama-13b baselines (5 examples per desired moral)
    python scripts/run_baseline.py --kind fewshot --model meta-llama/Llama-2-7b-chat-hf --out outputs/baseline_llama2_7b.jsonl
    # paper's Agg13Llama (aggregate the moral agents' answers)
    python scripts/run_baseline.py --kind aggregate --model meta-llama/Llama-2-13b-chat-hf \
        --agents outputs/agents_test_7B_rl.jsonl --out outputs/baseline_agg13llama.jsonl

Outputs use the same keys as run_aggregator.py, so they can be scored with
    python scripts/eval_aggregator.py --extra llama2_7b=outputs/baseline_llama2_7b.jsonl ...
Rows whose profile has none of the five values are skipped (they are not scored).
Needs a Hugging Face token with access to the Llama models (HF_TOKEN).
"""

from __future__ import annotations

import argparse
import json
import time

from mvrag.aggregator import build_aggregator_messages, profile_from_morals, profile_key
from mvrag.baselines import build_fewshot_messages, example_pool
from mvrag.data import load_split
from mvrag.generate import generate_batch, load_model
from mvrag.jobs import run_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=["fewshot", "aggregate"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--agents", default=None, help="agent answers (required for --kind aggregate)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--k", type=int, default=5, help="few-shot examples per desired moral")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="only the first N inputs (quick test)")
    parser.add_argument("--no-4bit", action="store_true", help="use fp16 instead of 4-bit")
    args = parser.parse_args()
    if args.kind == "aggregate" and not args.agents:
        raise SystemExit("--kind aggregate needs --agents")

    rows = load_split(args.split, args.data_dir)
    pool = example_pool(rows) if args.kind == "fewshot" else None
    agents = {}
    if args.agents:
        with open(args.agents, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    agents[rec["uid"]] = rec

    jobs, seen = [], set()
    for ex in sorted(rows, key=lambda e: (e.uid, e.row)):
        profile = profile_from_morals(ex.morals)
        key = profile_key(ex.uid, profile)
        if not any(profile) or key in seen:
            continue
        if args.kind == "aggregate" and ex.uid not in agents:
            continue
        seen.add(key)
        if args.kind == "fewshot":
            messages = build_fewshot_messages(ex.question, profile, pool, ex.uid, args.k)
        else:
            messages = build_aggregator_messages(ex.question, agents[ex.uid]["answers"], profile)
        jobs.append((key, messages, {"uid": ex.uid, "profile": list(profile)}))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} distinct (question, profile) inputs for --kind {args.kind} with {args.model}")

    model, tokenizer = load_model(args.model, four_bit=not args.no_4bit)
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
