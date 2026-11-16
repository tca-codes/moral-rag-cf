"""Thin wrappers around Hugging Face generation (greedy decoding only).

Greedy decoding is deliberate: the counterfactual experiments compare outputs
before/after removing an input, so sampling noise would be indistinguishable
from a real effect.
"""

from __future__ import annotations


def load_model(model_name: str, four_bit: bool = True):
    """Load a causal LM + tokenizer. Defaults suit a 16GB Colab T4 (fp16, 4-bit)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"device_map": "auto"}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, messages_list: list[list[dict]], max_new_tokens: int = 160) -> list[str]:
    """Greedy-generate one reply per chat (a list of {role, content} messages)."""
    import torch

    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[:, enc["input_ids"].shape[1] :]
    return [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
