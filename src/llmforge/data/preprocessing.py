from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizerBase

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def format_example(example: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
    """Render a {"messages": [...]} example into a single training string
    using the tokenizer's chat template (falls back to ChatML)."""
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHATML_TEMPLATE
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def tokenize_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> dict[str, list[int]]:
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
