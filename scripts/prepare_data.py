#!/usr/bin/env python
"""Validate and split a raw SFT dataset into train/eval JSONL files."""
from __future__ import annotations

import argparse
from pathlib import Path

from llmforge.data.loader import load_sft_dataset, train_eval_split
from llmforge.data.validation import validate_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to raw JSONL or HF dataset id")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_sft_dataset(args.input)
    validate_sft_dataset(dataset)

    split = train_eval_split(dataset, eval_ratio=args.eval_ratio, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split["train"].to_json(output_dir / "train.jsonl")
    split["eval"].to_json(output_dir / "eval.jsonl")

    print(f"train={len(split['train'])} eval={len(split['eval'])} -> {output_dir}")


if __name__ == "__main__":
    main()
