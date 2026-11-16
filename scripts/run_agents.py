"""Generate the five moral agents' answers for a reproducible subset of questions.

    python scripts/run_agents.py --split test --n 100 \
        --model Qwen/Qwen2.5-1.5B-Instruct --out outputs/agents_test.jsonl

Safe to re-run after a Colab disconnect: finished questions are skipped.
"""

from __future__ import annotations

import argparse
import time

from mvrag.agent_runner import run_agents
from mvrag.agents import build_plain_messages
from mvrag.data import load_split, unique_questions
from mvrag.generate import generate_batch, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=100, help="number of unique questions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default="outputs/agents_test.jsonl")
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--questions-per-chunk", type=int, default=2, help="x5 chats per batch")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--no-4bit", action="store_true", help="use fp16 instead of 4-bit")
    parser.add_argument("--plain", action="store_true",
                        help="run the model with no moral value (the paper's 'PT model' baseline)")
    args = parser.parse_args()

    questions = unique_questions(load_split(args.split, args.data_dir))[: args.n]
    print(f"{len(questions)} questions selected from split={args.split}")

    model, tokenizer = load_model(args.model, four_bit=not args.no_4bit)
    start = time.time()

    def gen(chats):
        return generate_batch(model, tokenizer, chats, args.max_new_tokens)

    def progress(done: int, total: int) -> None:
        print(f"  {done}/{total} new questions  ({time.time() - start:.0f}s)")

    builders = {"plain": build_plain_messages} if args.plain else None
    per_chunk = args.questions_per_chunk * (5 if args.plain else 1)  # keep batch size ~equal
    n = run_agents(questions, gen, args.out, per_chunk, progress, builders)
    print(f"Done. {n} new questions written to {args.out}")


if __name__ == "__main__":
    main()
