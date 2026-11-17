"""Reward training, stage 2: fine-tune one LoRA adapter per moral value.

    python scripts/train_agents.py --no-4bit

Each agent is trained to produce the candidate answers its judge scored highest
(see scripts/sample_candidates.py). Adapters are saved to
outputs/adapters/agents/<value>. Safe to re-run: finished values are skipped
and an interrupted value resumes from its last save.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from mvrag.agents import VALUES
from mvrag.agg_train import load_for_training, read_state, train
from mvrag.reward import build_agent_training_examples, load_selections, select_first_questions


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-dir", default="outputs/reward")
    parser.add_argument("--out-dir", default="outputs/adapters/agents")
    parser.add_argument("--values", nargs="+", default=list(VALUES), choices=list(VALUES))
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None, help="stop each value after this many steps (trial)")
    parser.add_argument("--limit-questions", type=int, default=None,
                        help="train on the first N questions only (same order as sampling), for the data-size comparison")
    parser.add_argument("--no-4bit", action="store_true", help="train on the uncompressed model")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="slower, but uses much less GPU memory (needed for the uncompressed 7B model on an L4)")
    args = parser.parse_args()

    for value in args.values:
        out_dir = Path(args.out_dir) / value
        state = read_state(out_dir)
        if state.get("total_steps") and state["step"] >= state["total_steps"]:
            print(f"[{value}] already trained ({state['step']} steps), skipping")
            continue
        selections = load_selections(Path(args.reward_dir) / f"{value}.jsonl")
        if args.limit_questions:
            selections = select_first_questions(selections, args.limit_questions)
        examples = build_agent_training_examples(selections, value)
        if not examples:
            raise SystemExit(f"[{value}] no candidates found in {args.reward_dir}; run sample_candidates.py first")
        print(f"\n[{value}] training on {len(examples)} winning answers")
        model, tokenizer = load_for_training(args.model, out_dir, four_bit=not args.no_4bit, rank=args.rank,
                                             grad_checkpointing=args.grad_checkpointing)
        state = train(model, tokenizer, examples, out_dir, epochs=args.epochs, batch_size=args.batch_size,
                      grad_accum=args.grad_accum, lr=args.lr, max_len=args.max_len,
                      save_every=args.save_every, max_steps=args.max_steps)
        print(f"[{value}] saved to {out_dir}: step {state['step']}/{state['total_steps']}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
