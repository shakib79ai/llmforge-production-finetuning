from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llmforge.evaluation.metrics import exact_match_score, rouge_scores


@dataclass
class BenchmarkResult:
    model_name: str
    predictions: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@torch.inference_mode()
def generate_predictions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int = 512,
) -> list[str]:
    predictions = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        predictions.append(tokenizer.decode(generated, skip_special_tokens=True))
    return predictions


def run_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    references: list[str],
    model_name: str = "model",
    max_new_tokens: int = 512,
) -> BenchmarkResult:
    predictions = generate_predictions(model, tokenizer, prompts, max_new_tokens)
    metrics = {
        "exact_match": exact_match_score(predictions, references),
        **rouge_scores(predictions, references),
    }
    return BenchmarkResult(model_name=model_name, predictions=predictions, metrics=metrics)
