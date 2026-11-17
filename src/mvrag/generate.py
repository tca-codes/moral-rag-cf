"""Thin wrappers around Hugging Face generation (greedy decoding only).

Greedy decoding is deliberate: the counterfactual experiments compare outputs
before/after removing an input, so sampling noise would be indistinguishable
from a real effect.
"""

from __future__ import annotations

import logging


class _DropLengthWarning(logging.Filter):
    """Llama ships max_length=4096 in its generation config; we always pass
    max_new_tokens, which takes precedence, so this warning is noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "seem to have been set" not in record.getMessage()


logging.getLogger("transformers.generation.utils").addFilter(_DropLengthWarning())


def load_model(model_name: str, four_bit: bool = True, adapter: str | None = None):
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
    if adapter:  # a trained LoRA adapter (e.g. the trained aggregator)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
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


def sample_batch(model, tokenizer, messages_list: list[list[dict]], n: int, max_new_tokens: int = 60,
                 temperature: float = 1.0, top_p: float = 0.95) -> list[list[str]]:
    """Sample `n` different replies per chat (used to get candidates for reward training)."""
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
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[:, enc["input_ids"].shape[1] :]
    texts = [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
    return [texts[i * n : (i + 1) * n] for i in range(len(messages_list))]


def load_model_with_adapters(model_name: str, adapters: dict[str, str], four_bit: bool = True):
    """Base model with several named LoRA adapters; switch with model.set_adapter(name)."""
    from peft import PeftModel

    model, tokenizer = load_model(model_name, four_bit=four_bit)
    names = list(adapters)
    model = PeftModel.from_pretrained(model, adapters[names[0]], adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(adapters[name], adapter_name=name)
    model.eval()
    return model, tokenizer
