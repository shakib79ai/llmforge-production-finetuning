from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    trust_remote_code: bool = False
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    max_seq_length: int = 4096


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str]
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


class TrainingConfig(BaseModel):
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    bf16: bool = True
    packing: bool = False
    seed: int = 42
    report_to: list[str] = []


class DeploymentConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    device: str = "cuda"


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig(**load_yaml(path))


def load_lora_config(path: str | Path) -> LoraConfig:
    return LoraConfig(**load_yaml(path))


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig(**load_yaml(path))


def load_deployment_config(path: str | Path) -> DeploymentConfig:
    return DeploymentConfig(**load_yaml(path))
