# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Export a full fine-tuning checkpoint (scripts/finetune_full.sh) as an inference model directory.

A checkpoint (<run_dir>/checkpoint/<step>/) holds FP32 `ema.safetensors` and `model.safetensors`
(~58 GB each). The export takes one of them (EMA by default, as for the released model), casts it to
BF16 and writes it as `ema.safetensors` next to the configs, tokenizer and VAE of the base model, so the
result works as MODEL_PATH / --model_path for every inference and evaluation entry point.

    python tools/export_full_checkpoint.py --checkpoint results/full_celebamask_img2palette_seg2img/checkpoint/0005000 \\
        --output_dir results/full_celebamask_img2palette_seg2img/export_0005000
"""

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint step dir with ema.safetensors / model.safetensors.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_path", default=os.environ.get("MODEL_PATH", "/models/SenseNova-Vision-7B-MoT"),
                        help="Model dir providing configs, tokenizer and ae.safetensors (the one training started from).")
    parser.add_argument("--weights", default="ema", choices=["ema", "model"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = parser.parse_args()

    src = os.path.join(args.checkpoint, f"{args.weights}.safetensors")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    state = {}
    with safe_open(src, framework="pt") as f:  # tensor by tensor keeps peak RAM at ~1x the output size
        for key in f.keys():
            t = f.get_tensor(key)
            state[key] = t.to(dtype) if t.is_floating_point() else t

    # Same keys as the base checkpoint, so the inference loader finds every weight.
    with safe_open(os.path.join(args.base_model_path, "ema.safetensors"), framework="pt") as f:
        base_keys = set(f.keys())
    missing, extra = sorted(base_keys - set(state)), sorted(set(state) - base_keys)
    if missing:
        raise SystemExit(f"checkpoint lacks {len(missing)} base weights, e.g. {missing[:5]}")
    if extra:
        print(f"[export] dropping {len(extra)} keys not in the base checkpoint, e.g. {extra[:5]}")
        for key in extra:
            del state[key]

    os.makedirs(args.output_dir, exist_ok=True)
    for name in os.listdir(args.base_model_path):
        path = os.path.join(args.base_model_path, name)
        if os.path.isfile(path) and name != "ema.safetensors":
            shutil.copy2(path, os.path.join(args.output_dir, name))
    save_file(state, os.path.join(args.output_dir, "ema.safetensors"))
    with open(os.path.join(args.output_dir, "export_info.json"), "w") as f:
        json.dump({"checkpoint": os.path.abspath(args.checkpoint), "weights": args.weights, "dtype": args.dtype,
                   "base_model_path": args.base_model_path, "num_tensors": len(state)}, f, indent=2)
    print(f"[export] {len(state)} tensors ({args.weights}, {args.dtype}) -> {args.output_dir}")


if __name__ == "__main__":
    main()
