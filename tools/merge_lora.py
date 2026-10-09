# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Merge a LoRA adapter from train/finetune_lora.py into a full model directory.

The output directory mirrors the base model directory (configs, tokenizer, VAE)
with a merged `ema.safetensors`, so it can be passed as MODEL_PATH / --model_path
to every inference entrypoint, including the NF4/INT8 quantized loaders.

    python tools/merge_lora.py \
        --adapter results/lora_demo/final \
        --output_dir results/lora_demo/merged
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="Directory with adapter_model.safetensors + adapter_config.json.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_path", default=None, help="Defaults to base_model_path in adapter_config.json.")
    args = parser.parse_args()

    with open(os.path.join(args.adapter, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    base_model_path = args.base_model_path or adapter_config["base_model_path"]
    scale = adapter_config["lora_alpha"] / adapter_config["r"]

    adapter = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
    state_dict = load_file(os.path.join(base_model_path, "ema.safetensors"))

    merged = 0
    for key in adapter:
        if ".lora_A." not in key:
            continue
        module = key.split(".lora_A.")[0]
        lora_a = adapter[key]
        lora_b = adapter[key.replace(".lora_A.", ".lora_B.")]
        weight_key = f"{module}.weight"
        weight = state_dict[weight_key]
        state_dict[weight_key] = (weight.float() + scale * (lora_b @ lora_a)).to(weight.dtype)
        merged += 1
    print(f"Merged {merged} LoRA modules (scale={scale}) into {base_model_path}/ema.safetensors")

    os.makedirs(args.output_dir, exist_ok=True)
    for name in os.listdir(base_model_path):
        src = os.path.join(base_model_path, name)
        if name != "ema.safetensors" and os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.output_dir, name))
    save_file(state_dict, os.path.join(args.output_dir, "ema.safetensors"))
    shutil.copy2(os.path.join(args.adapter, "adapter_config.json"), os.path.join(args.output_dir, "merged_adapter_config.json"))
    print(f"Wrote merged model to {args.output_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
