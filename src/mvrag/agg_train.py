"""Training the Contextual Aggregator with LoRA (the paper's CMVA-GS system).

Paper: each sample is a MIC question + moral profile + the five agents' answers;
the aggregator is trained with cross-entropy to produce the ground-truth answer,
by supervised LoRA fine-tuning (rank 128, lr 5e-7, batch 256, 21.6K steps on
8 A100s). Ours: same data and objective, one pass, on a single Colab GPU; see
docs/DECISIONS.md for the settings that differ.

The loss is computed on the target reply only (prompt tokens are masked out).
Training can be interrupted: the adapter and the step counter are saved every
`save_every` steps and a re-run continues from the last save.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from mvrag.aggregator import build_aggregator_messages, profile_from_morals, profile_key

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_training_examples(examples, agents: dict, target_field: str = "worker_answer",
                            retrieved: dict | None = None) -> list[dict]:
    """One training sample per MIC row that has agent answers and a target reply.

    retrieved: optional {profile_key: {"retrieved": [...]}} for the RAG variant.
    """
    out = []
    for ex in sorted(examples, key=lambda e: (e.uid, e.row)):
        target = getattr(ex, target_field)
        if ex.uid not in agents or not target:
            continue
        profile = profile_from_morals(ex.morals)
        context = None
        if retrieved is not None:
            rec = retrieved.get(profile_key(ex.uid, profile))
            if rec is None:
                raise ValueError(f"no retrieved examples for {ex.uid}; run build_retrieval.py for this split")
            context = rec["retrieved"]
        messages = build_aggregator_messages(ex.question, agents[ex.uid]["answers"], profile, context)
        out.append({"messages": messages, "target": target})
    return out


def encode_example(tokenizer, messages: list[dict], target: str, max_len: int):
    """(input_ids, labels) with labels = -100 on the prompt; None if too long."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(target_ids) > max_len:
        return None
    return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids


def collate(encoded: list, pad_id: int, left: bool = False):
    """Pad a list of (input_ids, labels) into tensors (right-padded unless left=True)."""
    import torch

    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    labels = torch.full((len(encoded), width), -100, dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(encoded):
        sl = slice(width - len(ids), width) if left else slice(0, len(ids))
        input_ids[i, sl] = torch.tensor(ids)
        labels[i, sl] = torch.tensor(lab)
        mask[i, sl] = 1
    return input_ids, mask, labels


def target_only_loss(model, input_ids, mask, labels):
    """Causal-LM loss computed from the logits of the target positions only.

    Expects LEFT-padded inputs, so every example's target reply sits at the end.
    Only the last (longest target + 1) positions are projected onto the
    vocabulary, instead of the whole prompt. With a 150K-word vocabulary and
    prompts of several hundred tokens this saves most of the memory spent on
    logits; the loss value is the same as the standard one (mean token
    cross-entropy over the target tokens).
    """
    import torch

    keep = int((labels != -100).sum(dim=1).max().item()) + 1
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
    logits = model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                   logits_to_keep=keep).logits
    pred = logits[:, :-1, :].float()
    gold = labels[:, -(keep - 1):]
    return torch.nn.functional.cross_entropy(pred.reshape(-1, pred.size(-1)), gold.reshape(-1), ignore_index=-100)


def lr_at(step: int, total_steps: int, base_lr: float, warmup_frac: float = 0.03) -> float:
    """Linear warmup then linear decay to zero."""
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr * max(0.0, (total_steps - step) / max(1, total_steps - warmup))


def example_order(n: int, epochs: int, seed: int) -> list[int]:
    """Deterministic shuffled order over all epochs, so a resumed run continues exactly."""
    order: list[int] = []
    for epoch in range(epochs):
        idx = list(range(n))
        random.Random(seed + epoch).shuffle(idx)
        order.extend(idx)
    return order


def read_state(out_dir: str | Path) -> dict:
    path = Path(out_dir) / "train_state.json"
    return json.loads(path.read_text()) if path.exists() else {"step": 0}


def load_for_training(model_name: str, out_dir: str | Path, four_bit: bool = True, rank: int = 64,
                      grad_checkpointing: bool = False):
    """Base model + LoRA adapter (resumed from `out_dir` if a saved adapter exists)."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else (torch.float16 if cuda else torch.float32)

    kwargs: dict = {"device_map": "auto"} if cuda else {}
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4"
        )
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if four_bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=grad_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif grad_checkpointing:
        # Recompute activations in the backward pass to save memory. The base
        # weights are frozen, so the inputs must require grad for checkpointing
        # to pass gradients through to the LoRA layers.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if grad_checkpointing:
        model.config.use_cache = False

    out_dir = Path(out_dir)
    if (out_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(out_dir), is_trainable=True)
        print(f"Resuming adapter from {out_dir} at step {read_state(out_dir)['step']}")
    else:
        config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                            target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    return model, tokenizer


OPTIMIZER_FILE = "optimizer.pt"


def save_optimizer(optimizer, out_dir: str | Path) -> None:
    """Save AdamW's running averages, so a resumed run continues exactly where it stopped."""
    import torch

    tmp = Path(out_dir) / (OPTIMIZER_FILE + ".tmp")
    torch.save(optimizer.state_dict(), tmp)
    tmp.replace(Path(out_dir) / OPTIMIZER_FILE)


def restore_optimizer(optimizer, out_dir: str | Path) -> bool:
    """Load the optimizer state saved by `save_optimizer`; False if there is none."""
    import torch

    path = Path(out_dir) / OPTIMIZER_FILE
    if not path.exists():
        return False
    device = optimizer.param_groups[0]["params"][0].device
    optimizer.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return True


def _save(model, out_dir: Path, state: dict, optimizer=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    if optimizer is not None:
        save_optimizer(optimizer, out_dir)
    (out_dir / "train_state.json").write_text(json.dumps(state, indent=2))


def train(
    model,
    tokenizer,
    examples: list[dict],
    out_dir: str | Path,
    epochs: int = 1,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 1e-4,
    max_len: int = 640,
    seed: int = 0,
    save_every: int = 500,
    log_every: int = 25,
    max_steps: int | None = None,
    target_logits_only: bool = False,
) -> dict:
    """Train the LoRA adapter; returns the final state dict.

    A "step" is one optimizer update = batch_size * grad_accum examples.
    `max_steps` stops early (for a timing trial) without changing the schedule.
    `target_logits_only` computes logits only for the target reply (much less
    memory for long prompts; same loss).
    """
    import torch

    out_dir = Path(out_dir)
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    use_bf16 = cuda and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and not use_bf16)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    order = example_order(len(examples), epochs, seed)
    per_step = batch_size * grad_accum
    total_steps = math.ceil(len(order) / per_step)
    state = read_state(out_dir)
    start_step = state["step"]
    stop_step = total_steps if max_steps is None else min(total_steps, start_step + max_steps)
    print(f"{len(examples)} examples, {total_steps} steps in total; running steps {start_step} to {stop_step}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if start_step > 0:
        if restore_optimizer(optimizer, out_dir):
            print("Resumed the optimizer state")
        else:
            print("Note: no saved optimizer state; the optimizer restarts (older checkpoint)")
    model.train()
    skipped, losses, t0 = 0, [], time.time()

    for step in range(start_step, stop_step):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, total_steps, lr)
        optimizer.zero_grad()
        step_loss = 0.0
        chunk = order[step * per_step : (step + 1) * per_step]
        for b in range(0, len(chunk), batch_size):
            encoded = []
            for i in chunk[b : b + batch_size]:
                enc = encode_example(tokenizer, examples[i]["messages"], examples[i]["target"], max_len)
                if enc is None:
                    skipped += 1
                else:
                    encoded.append(enc)
            if not encoded:
                continue
            input_ids, mask, labels = (t.to(device) for t in
                                       collate(encoded, tokenizer.pad_token_id, left=target_logits_only))
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cuda):
                if target_logits_only:
                    loss = target_only_loss(model, input_ids, mask, labels) / grad_accum
                else:
                    loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss / grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(step_loss)

        done = step + 1
        if log_every and (done % log_every == 0 or done == stop_step):
            elapsed = time.time() - t0
            rate = elapsed / (done - start_step)
            mem = f"  peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if cuda else ""
            recent = sum(losses[-log_every:]) / len(losses[-log_every:])
            print(f"  step {done}/{total_steps}  loss {recent:.4f}  {rate:.2f}s/step  "
                  f"about {rate * (total_steps - done) / 3600:.1f}h left for the full run{mem}")
        if done % save_every == 0 or done == stop_step:
            state = {"step": done, "total_steps": total_steps, "examples": len(examples),
                     "skipped_too_long": skipped, "last_loss": losses[-1]}
            _save(model, out_dir, state, optimizer)
    return state
