# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for evaluating CelebAMask-HQ fine-tuned modes (img2palette, seg2img).

The model is the base checkpoint, optionally with a LoRA adapter from train/finetune_lora.py
applied in place (W += alpha/r * B @ A, the same arithmetic as tools/merge_lora.py), so a
training checkpoint can be evaluated without writing a merged 29 GB copy. Evaluation runs in
BF16 with the whole model on the GPU (30 GiB budget on a 32 GB card).
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from data.celebamask import IMG2PALETTE_PROMPT, PALETTE, load_label_map, load_photo, load_split_ids
from inference.sensenova_vision import BASE_PARAMS, SenseNovaVisionModel

DATASET_ROOT = "/datasets/CelebAMask-HQ/CelebAMask-HQ"
_COLOURS = PALETTE.astype(np.int32)


def add_model_args(parser):
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "/models/SenseNova-Vision-7B-MoT"))
    parser.add_argument("--adapter", default=None,
                        help="LoRA checkpoint dir (adapter_model.safetensors + adapter_config.json); omit for the base model.")
    parser.add_argument("--max_mem_per_gpu", default="30GiB")
    parser.add_argument("--dataset_root", default=DATASET_ROOT)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="First N images of the split.")
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument("--cfg_text_scale", type=float, default=None)
    parser.add_argument("--cfg_img_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)


def overrides(args):
    return {k: getattr(args, k) for k in ("num_timesteps", "cfg_text_scale", "cfg_img_scale")
            if getattr(args, k) is not None}


@torch.no_grad()
def apply_lora_inplace(bagel, adapter_dir):
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    scale = cfg["lora_alpha"] / cfg["r"]
    adapter = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    modules = dict(bagel.named_modules())
    merged = 0
    for key in adapter:
        if ".lora_A." not in key:
            continue
        name = key.split(".lora_A.")[0]
        weight = modules[name].weight
        if weight.device.type == "meta":
            raise RuntimeError(f"{name} is offloaded; raise --max_mem_per_gpu or merge with tools/merge_lora.py")
        delta = scale * (adapter[key.replace(".lora_A.", ".lora_B.")].float() @ adapter[key].float())
        weight.add_(delta.to(weight.device, weight.dtype))
        merged += 1
    print(f"[eval] applied {merged} LoRA modules from {adapter_dir} (scale={scale}, step={cfg.get('step')})")
    return cfg


def load_model(args):
    model = SenseNovaVisionModel(model_path=args.model_path, dtype="bf16", max_mem_per_gpu=args.max_mem_per_gpu)
    info = {"model_path": args.model_path, "adapter": args.adapter}
    if args.adapter:
        info["adapter_config"] = apply_lora_inplace(model.model, args.adapter)
    return model, info


def split_ids(args):
    return load_split_ids(args.dataset_root, args.split)[: args.limit]


def palette_to_label(image):
    """Snap every pixel to the nearest official CelebAMask-HQ colour; also return the share of pixels
    farther than 30 (RGB Euclidean) from every palette colour, i.e. how far the output strays from the format."""
    rgb = np.asarray(image.convert("RGB").resize((512, 512), Image.NEAREST), dtype=np.int32)
    dist = ((rgb[:, :, None, :] - _COLOURS[None, None]) ** 2).sum(-1)
    return dist.argmin(-1).astype(np.uint8), float((np.sqrt(dist.min(-1)) > 30).mean())


def segment_to_palette(model, photo, seed, params):
    """Run the img2palette mode: returns (raw generated map, label map, off-palette share)."""
    raw = model.generate(contents=[{"type": "image", "value": photo}, {"type": "text", "value": IMG2PALETTE_PROMPT}],
                         mode="dense_perception", noise_seed=seed, **params).convert("RGB")
    label, off = palette_to_label(raw)
    return raw, label, off


def confusion(gt, pred, n=19):
    return np.bincount(gt.astype(np.int64).ravel() * n + pred.astype(np.int64).ravel(), minlength=n * n).reshape(n, n)


def miou(conf):
    tp = np.diag(conf).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (conf.sum(0) + conf.sum(1) - tp)
    return float(np.nanmean(iou[1:])), iou


__all__ = ["add_model_args", "overrides", "load_model", "split_ids", "palette_to_label", "segment_to_palette",
           "confusion", "miou", "load_photo", "load_label_map", "BASE_PARAMS"]
