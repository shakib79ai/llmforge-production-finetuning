#!/usr/bin/env python
"""Run LoRA/QLoRA supervised fine-tuning from config files."""
from __future__ import annotations

import argparse
from functools import partial

from llmforge.config import load_lora_config, load_model_config, load_training_config
from llmforge.data.loader import load_sft_dataset
from llmforge.data.preprocessing import format_example
from llmforge.training.trainer import build_trainer, load_base_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--lora-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--eval-file", default="data/processed/eval.jsonl")
    args = parser.parse_args()

    model_cfg = load_model_config(args.model_config)
    lora_cfg = load_lora_config(args.lora_config)
    training_cfg = load_training_config(args.training_config)

    _, tokenizer = load_base_model_and_tokenizer(model_cfg, lora_cfg)

    train_dataset = load_sft_dataset(args.train_file).map(
        partial(format_example, tokenizer=tokenizer)
    )
    eval_dataset = load_sft_dataset(args.eval_file).map(
        partial(format_example, tokenizer=tokenizer)
    )

    trainer = build_trainer(model_cfg, lora_cfg, training_cfg, train_dataset, eval_dataset)
    trainer.train()
    trainer.save_model(training_cfg.output_dir)


if __name__ == "__main__":
    main()
