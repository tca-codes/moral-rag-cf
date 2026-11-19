"""Train the aggregator (moderator) with LoRA on MIC: the paper's CMVA-GS.

Timing trial first (40 steps, a few minutes), then the full run:

    python scripts/train_aggregator.py --max-steps 40 --out-dir outputs/adapters/trial
    python scripts/train_aggregator.py

Needs the agents' answers for the training questions (scripts/run_agents.py
--split train). Safe to re-run: it resumes from the last saved step.
"""

from __future__ import annotations

import argparse
import json

from mvrag.agg_train import build_training_examples, load_for_training, train
from mvrag.data import load_split
from mvrag.jobs import read_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", default="outputs/agents_train_7B_full.jsonl")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--target", default="worker_answer", choices=["worker_answer", "answer"],
                        help="which MIC reply is the ground truth: human-written (worker_answer) or chatbot (answer)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out-dir", default="outputs/adapters/aggregator")
    parser.add_argument("--data-dir", default="data/MIC")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (timing trial)")
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true", help="train on the uncompressed model (needs more memory)")
    parser.add_argument("--grad-checkpointing", action="store_true", help="slower, but uses less GPU memory")
    parser.add_argument("--full-logits", action="store_true",
                        help="compute logits for every position (default: only the target reply, which saves memory)")
    parser.add_argument("--retrieved", default=None,
                        help="retrieved examples for the training split (RAG variant); omit for plain CMVA-GS")
    args = parser.parse_args()

    agents = {}
    with open(args.agents, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                agents[rec["uid"]] = rec
    rows = load_split(args.split, args.data_dir)
    retrieved = read_outputs(args.retrieved) if args.retrieved else None
    examples = build_training_examples(rows, agents, args.target, retrieved)
    if args.limit_examples:
        examples = examples[: args.limit_examples]
    print(f"{len(rows)} MIC rows in split={args.split}; {len(examples)} usable training examples "
          f"(target = {args.target}); agent answers for {len(agents)} questions")
    if not examples:
        raise SystemExit("No training examples. Has run_agents.py finished for this split?")

    model, tokenizer = load_for_training(args.model, args.out_dir, four_bit=not args.no_4bit,
                                         rank=args.rank, grad_checkpointing=args.grad_checkpointing)
    model.print_trainable_parameters()
    state = train(model, tokenizer, examples, args.out_dir, epochs=args.epochs, batch_size=args.batch_size,
                  grad_accum=args.grad_accum, lr=args.lr, max_len=args.max_len,
                  save_every=args.save_every, max_steps=args.max_steps,
                  target_logits_only=not args.full_logits)
    if state.get("skipped_too_long"):
        print(f"Note: {state['skipped_too_long']} examples were longer than --max-len {args.max_len} and skipped")
    print(f"Saved adapter to {args.out_dir}: {state}")


if __name__ == "__main__":
    main()
