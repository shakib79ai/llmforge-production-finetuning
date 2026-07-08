from llmforge.config import (
    load_deployment_config,
    load_lora_config,
    load_model_config,
    load_training_config,
)


def test_load_model_config():
    cfg = load_model_config("configs/model/llama3-8b.yaml")
    assert cfg.name == "meta-llama/Meta-Llama-3-8B"
    assert cfg.max_seq_length == 4096


def test_load_lora_config_qlora():
    cfg = load_lora_config("configs/lora/qlora.yaml")
    assert cfg.load_in_4bit is True
    assert cfg.r == 64


def test_load_training_config():
    cfg = load_training_config("configs/training/sft.yaml")
    assert cfg.num_train_epochs == 3
    assert cfg.report_to == ["wandb"]


def test_load_deployment_config():
    cfg = load_deployment_config("configs/deployment/api.yaml")
    assert cfg.port == 8000
