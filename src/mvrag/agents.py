"""Prompt-based stand-ins for the paper's five PPO-tuned Moral Agents.

The CMVA-GS paper RL-fine-tunes one LLM per moral value. That is not feasible on
Colab, so each "agent" here is the same small model with a value-specific system
prompt. The descriptions are our paraphrases of Moral Foundations Theory; the
paper's actual prompts/rewards are not published. See docs/DECISIONS.md.

v2 (2026-11-13): the first prompts produced five near-identical, long answers
that opened with "As an AI language model". MIC replies are short, casual
chatbot lines, so the agents now answer as a person, in 1-2 sentences, with a
sharper per-value lens.
"""

from __future__ import annotations

from mvrag.data import PAPER_VALUES

VALUES = PAPER_VALUES
PROMPT_VERSION = "v2"

VALUE_DESCRIPTIONS = {
    "care": "care and protection from harm: compassion, kindness, preventing suffering",
    "fairness": "fairness and justice: equal treatment, honesty, not cheating or exploiting others",
    "loyalty": "loyalty: faithfulness to one's group, family, friends and commitments, not betraying them",
    "authority": "authority and respect: respecting legitimate leaders, rules, traditions and social order",
    "sanctity": "sanctity and purity: respecting what is sacred or dignified, avoiding degradation",
}

VALUE_LENS = {
    "care": "You judge every situation by who could get hurt and how to look after them.",
    "fairness": "You judge every situation by whether everyone is treated equally and honestly.",
    "loyalty": "You judge every situation by whether people stand by their family, friends and group.",
    "authority": "You judge every situation by whether rules, elders, traditions and proper roles are respected.",
    "sanctity": "You judge every situation by whether it is pure, decent and dignified, or degrading.",
}


def build_agent_messages(value: str, question: str) -> list[dict]:
    """Chat messages for the moral agent that prioritizes `value`."""
    if value not in VALUE_DESCRIPTIONS:
        raise ValueError(f"unknown value {value!r}; expected one of {list(VALUE_DESCRIPTIONS)}")
    system = (
        "You are a person replying to a question on an online forum. "
        "You are not an AI and you never mention being one. "
        f"Your strongest moral conviction is {VALUE_DESCRIPTIONS[value]}. "
        f"{VALUE_LENS[value]} "
        "Reply in the first person, in one or two short, casual sentences. "
        "Take a clear stance that follows from this conviction, and do not hedge "
        "or list other viewpoints."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def build_plain_messages(question: str) -> list[dict]:
    """Same style instructions as the agents but with no moral value.

    Stand-in for the paper's "PT model" (the starting model before any value
    alignment), so that agent-vs-plain differences are due to the value prompt.
    """
    system = (
        "You are a person replying to a question on an online forum. "
        "You are not an AI and you never mention being one. "
        "Reply in the first person, in one or two short, casual sentences."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
