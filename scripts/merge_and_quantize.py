#!/usr/bin/env python
"""Merge a LoRA adapter into the base model and optionally quantize for serving."""
from __future__ import annotations

import argparse

from llmforge.inference.merge import merge_adapter
from llmforge.inference.quantize import quantize_for_serving


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--merged-dir", default="outputs/merged")
    parser.add_argument("--quantized-dir", default=None, help="If set, also produce a 4-bit copy here")
    args = parser.parse_args()

    merged_path = merge_adapter(args.base_model, args.adapter_path, args.merged_dir)
    print(f"Merged model saved to {merged_path}")

    if args.quantized_dir:
        quantized_path = quantize_for_serving(str(merged_path), args.quantized_dir)
        print(f"Quantized model saved to {quantized_path}")


if __name__ == "__main__":
    main()
