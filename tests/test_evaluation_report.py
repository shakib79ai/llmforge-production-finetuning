from llmforge.evaluation.benchmark import BenchmarkResult
from llmforge.evaluation.report import compare_results


def test_compare_results_computes_deltas():
    before = BenchmarkResult(model_name="base", metrics={"exact_match": 0.2, "rouge1": 0.4})
    after = BenchmarkResult(model_name="fine-tuned", metrics={"exact_match": 0.5, "rouge1": 0.6})

    comparison = compare_results(before, after)

    assert comparison["delta"]["exact_match"] == 0.3
    assert round(comparison["delta"]["rouge1"], 2) == 0.2
    assert comparison["before"]["model"] == "base"
    assert comparison["after"]["model"] == "fine-tuned"
