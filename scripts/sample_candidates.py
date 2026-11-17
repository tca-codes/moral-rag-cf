"""Reward training, stage 1: sample candidate answers and score them with the judges.

    python scripts/sample_candidates.py --n 4000 --no-4bit

For each moral value and each of the first N training questions, the
value-prompted model samples several answers; that value's judge scores them.
Everything is saved to outputs/reward/<value>.jsonl. Safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mvrag.agents import VALUES
from mvrag.data import load_split, unique_questions
from mvrag.generate import load_model, sample_batch
from mvrag.judges import load_judge, predict_probs
from mvrag.reward import load_selections, reward_summary, sample_and_score


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4000, help="number of questions per value")
    parser.add_argument("--samples", type=int, default=4, help="candidate answers per question")
    parser.add_argument("--values", nargs="+", default=list(VALUES), choices=list(VALUES))
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judges-dir", default="judges/full")
    parser.add_argument("--out-dir", default="outputs/reward")
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--batch-size", type=int, default=16, help="questions per batch (x samples sequences)")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-4bit", action="store_true", help="use fp16 instead of 4-bit")
    args = parser.parse_args()

    questions = unique_questions(load_split(args.split, args.data_dir))[: args.n]
    print(f"{len(questions)} questions from split={args.split}, {args.samples} candidates each, values={args.values}")
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, four_bit=not args.no_4bit)

    def sample_fn(chats):
        return sample_batch(model, tokenizer, chats, args.samples, args.max_new_tokens, args.temperature, args.top_p)

    summary = {}
    for value in args.values:
        judge, judge_tok = load_judge(Path(args.judges_dir) / value)
        start = time.time()

        def score_fn(texts):
            return predict_probs(judge, judge_tok, texts)

        def progress(done: int, total: int) -> None:
            if done % (args.batch_size * 10) == 0 or done == total:
                print(f"  [{value}] {done}/{total} new questions  ({time.time() - start:.0f}s)")

        out_path = Path(args.out_dir) / f"{value}.jsonl"
        sample_and_score(questions, value, sample_fn, score_fn, out_path, args.batch_size, progress)
        summary[value] = reward_summary(load_selections(out_path))
        del judge

    print(f"\n{'value':<11}{'questions':>10}{'avg reward, all candidates':>29}{'avg reward, best candidate':>29}")
    for value, s in summary.items():
        print(f"{value:<11}{s['n']:>10}{s['mean_candidate_reward']:>29.3f}{s['mean_best_reward']:>29.3f}")
    (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
