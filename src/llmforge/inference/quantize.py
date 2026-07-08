from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def quantize_for_serving(
    model_path: str,
    output_dir: str | Path,
    quant_type: str = "nf4",
) -> Path:
    """Load a merged model in 4-bit and re-save it so deployment only ever
    loads the smaller quantized weights, not the full-precision checkpoint."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quant_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir
