#!/usr/bin/env python
"""Benchmark a base model against a fine-tuned adapter and write a before/after report."""
from __future__ import annotations

import argparse
import json

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmforge.evaluation.benchmark import run_benchmark
from llmforge.evaluation.report import compare_results, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--eval-file", required=True, help="JSON file with 'prompts' and 'references'")
    parser.add_argument("--output", default="outputs/eval_report.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    with open(args.eval_file, encoding="utf-8") as f:
        eval_data = json.load(f)
    prompts, references = eval_data["prompts"], eval_data["references"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="bfloat16")

    before = run_benchmark(
        base_model, tokenizer, prompts, references, "base", args.max_new_tokens
    )

    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    after = run_benchmark(
        tuned_model, tokenizer, prompts, references, "fine-tuned", args.max_new_tokens
    )

    comparison = compare_results(before, after)
    write_report(comparison, args.output)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
