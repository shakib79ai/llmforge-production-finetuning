from __future__ import annotations

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(base_model_name: str, adapter_path: str, output_dir: str | Path) -> Path:
    """Merge a trained LoRA adapter into the base model weights and save a
    standalone model directory ready for quantization or serving."""
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="bfloat16")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_dir)

    return output_dir
