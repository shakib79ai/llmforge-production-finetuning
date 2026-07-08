from __future__ import annotations

import math

import evaluate
from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_exact_match = evaluate.load("exact_match")


def rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = _rouge.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = max(len(predictions), 1)
    return {k: v / n for k, v in totals.items()}


def exact_match_score(predictions: list[str], references: list[str]) -> float:
    result = _exact_match.compute(predictions=predictions, references=references)
    return result["exact_match"]


def perplexity(loss: float) -> float:
    return math.exp(loss)
