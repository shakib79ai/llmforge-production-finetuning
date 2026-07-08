# llmforge

Production-ready parameter-efficient fine-tuning (LoRA/QLoRA) framework for open-source LLMs, covering the full lifecycle: dataset preparation, supervised fine-tuning, automated evaluation, experiment tracking, inference optimization, and deployment.

## Layout

```
configs/            YAML configs: model, LoRA/QLoRA, training, deployment
src/llmforge/
  config.py          Pydantic schemas + YAML loaders
  data/              Loading, chat-template formatting, schema validation
  training/          LoRA/QLoRA setup and the SFT trainer wrapper
  evaluation/         Metrics (ROUGE, exact match, perplexity), before/after benchmarking
  tracking/           Experiment tracker wrapper (wandb)
  inference/         Adapter merging, quantization, generation engine
  deployment/         FastAPI serving app
scripts/             CLI entry points tying the modules together
tests/               Unit tests
docker/              Training and serving container images
```

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Validate and split a raw dataset of {"messages": [...]} rows
python scripts/prepare_data.py --input data/raw/my_dataset.jsonl

# 2. Fine-tune with LoRA or QLoRA
python scripts/train.py \
  --model-config configs/model/llama3-8b.yaml \
  --lora-config configs/lora/qlora.yaml \
  --training-config configs/training/sft.yaml

# 3. Compare base vs. fine-tuned on held-out prompts
python scripts/evaluate.py \
  --base-model meta-llama/Meta-Llama-3-8B \
  --adapter-path outputs/sft-run \
  --eval-file data/processed/eval_prompts.json

# 4. Merge the adapter and (optionally) quantize for serving
python scripts/merge_and_quantize.py \
  --base-model meta-llama/Meta-Llama-3-8B \
  --adapter-path outputs/sft-run \
  --merged-dir outputs/merged

# 5. Serve
python scripts/serve.py --config configs/deployment/api.yaml
```

## Testing

```bash
pytest
```
