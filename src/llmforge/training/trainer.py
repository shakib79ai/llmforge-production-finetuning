from __future__ import annotations

from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from llmforge.config import LoraConfig, ModelConfig, TrainingConfig
from llmforge.training.callbacks import EvalLoggingCallback
from llmforge.training.lora import build_peft_config, build_quantization_config


def load_base_model_and_tokenizer(
    model_cfg: ModelConfig, lora_cfg: LoraConfig
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    quant_config = build_quantization_config(lora_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        trust_remote_code=model_cfg.trust_remote_code,
        torch_dtype=model_cfg.torch_dtype,
        quantization_config=quant_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name, trust_remote_code=model_cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def build_trainer(
    model_cfg: ModelConfig,
    lora_cfg: LoraConfig,
    training_cfg: TrainingConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
) -> SFTTrainer:
    model, tokenizer = load_base_model_and_tokenizer(model_cfg, lora_cfg)
    peft_config = build_peft_config(lora_cfg)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=training_cfg.output_dir,
        num_train_epochs=training_cfg.num_train_epochs,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        warmup_ratio=training_cfg.warmup_ratio,
        weight_decay=training_cfg.weight_decay,
        logging_steps=training_cfg.logging_steps,
        eval_strategy=training_cfg.eval_strategy,
        eval_steps=training_cfg.eval_steps,
        save_strategy=training_cfg.save_strategy,
        save_steps=training_cfg.save_steps,
        save_total_limit=training_cfg.save_total_limit,
        bf16=training_cfg.bf16,
        packing=training_cfg.packing,
        seed=training_cfg.seed,
        report_to=training_cfg.report_to,
        max_seq_length=model_cfg.max_seq_length,
        dataset_text_field="text",
    )

    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[EvalLoggingCallback()],
    )
