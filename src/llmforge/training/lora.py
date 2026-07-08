from __future__ import annotations

import torch
from peft import LoraConfig as PeftLoraConfig
from transformers import BitsAndBytesConfig

from llmforge.config import LoraConfig

DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def build_peft_config(cfg: LoraConfig) -> PeftLoraConfig:
    return PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        task_type=cfg.task_type,
    )


def build_quantization_config(cfg: LoraConfig) -> BitsAndBytesConfig | None:
    if not cfg.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=DTYPE_MAP[cfg.bnb_4bit_compute_dtype],
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )
