"""Value judges: one binary classifier per moral value (the paper's reward models).

A judge reads a text and returns P(text follows the value) in [0, 1], which is
what the paper uses as the reward for that value.
"""

from __future__ import annotations

import json
from pathlib import Path


def binary_metrics(labels: list[int], preds: list[int]) -> dict:
    """Accuracy, macro-F1, and the majority-class baseline accuracy."""
    n = len(labels)
    if n == 0:
        raise ValueError("no examples")
    f1s = []
    for cls in (0, 1):
        tp = sum(1 for y, p in zip(labels, preds) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(labels, preds) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(labels, preds) if y == cls and p != cls)
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    pos = sum(labels)
    return {
        "n": n,
        "accuracy": sum(1 for y, p in zip(labels, preds) if y == p) / n,
        "macro_f1": sum(f1s) / 2,
        "majority_baseline": max(pos, n - pos) / n,
        "positive_rate": pos / n,
    }


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def predict_probs(model, tokenizer, texts: list[str], batch_size: int = 64, max_len: int = 64) -> list[float]:
    """P(label = 1) for each text."""
    import torch

    model.eval()
    probs: list[float] = []
    with torch.no_grad():
        for chunk in _batches(texts, batch_size):
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            logits = model(**enc).logits.float()
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return probs


def train_judge(
    train: list[tuple[str, int]],
    dev: list[tuple[str, int]],
    out_dir: str | Path,
    base_model: str = "distilroberta-base",
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_len: int = 64,
    seed: int = 0,
    log_every: int = 100,
) -> dict:
    """Fine-tune `base_model` as a binary classifier; save it and return dev metrics."""
    import random

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rng = random.Random(seed)
    step = 0
    for _ in range(epochs):
        order = list(train)
        rng.shuffle(order)
        model.train()
        for chunk in _batches(order, batch_size):
            texts = [t for t, _ in chunk]
            labels = torch.tensor([y for _, y in chunk], device=device)
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = model(**enc, labels=labels).loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if log_every and step % log_every == 0:
                print(f"    step {step}  loss {loss.item():.4f}")

    probs = predict_probs(model, tokenizer, [t for t, _ in dev], max_len=max_len)
    metrics = binary_metrics([y for _, y in dev], [1 if p >= 0.5 else 0 for p in probs])
    metrics.update({"base_model": base_model, "n_train": len(train), "epochs": epochs})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_judge(judge_dir: str | Path):
    """Load a saved judge; returns (model, tokenizer) on GPU if available."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(judge_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(judge_dir)).to(device)
    model.eval()
    return model, tokenizer
