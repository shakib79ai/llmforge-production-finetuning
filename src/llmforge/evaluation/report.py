from __future__ import annotations

import json
from pathlib import Path

from llmforge.evaluation.benchmark import BenchmarkResult


def compare_results(before: BenchmarkResult, after: BenchmarkResult) -> dict:
    deltas = {
        metric: after.metrics[metric] - before.metrics.get(metric, 0.0)
        for metric in after.metrics
    }
    return {
        "before": {"model": before.model_name, "metrics": before.metrics},
        "after": {"model": after.model_name, "metrics": after.metrics},
        "delta": deltas,
    }


def write_report(comparison: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
