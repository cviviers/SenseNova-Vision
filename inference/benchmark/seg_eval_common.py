# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for evaluating the fine-tuned img2palette / seg2img modes on CelebAMask-HQ and COCO-Stuff.

* Model: the base checkpoint, optionally with a LoRA adapter from train/finetune_lora.py applied in place
  (W += alpha/r * B @ A, the arithmetic of tools/merge_lora.py), so a training checkpoint is evaluated
  without writing a merged 29 GB copy. BF16 with the whole model on the GPU (30 GiB budget on 32 GB).
* Datasets: an adapter per dataset gives the evaluation images, labels at the training-time size,
  the palette, the training prompts and the metric protocol, and the image transforms used in training
  (the model's default inference transform would e.g. upscale a 512x384 COCO image to 683x512).
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from data.transforms import ImageTransform
from inference.sensenova_vision import SenseNovaVisionModel


# ----------------------------------------------------------------------------- datasets
class CelebAMaskEval:
    name = "celebamask"
    default_root = "/datasets/CelebAMask-HQ/CelebAMask-HQ"
    ignore_unlabeled = False   # class 0 (background) is a real class; mIoU averages the 18 face classes
    photo_question = "Is this a realistic photograph of a human face? Answer yes or no."
    # Training transforms (data/configs/celebamask_hq/celebamask_img2palette_seg2img.yaml).
    vae_transform = ImageTransform(512, 512, 16)
    vit_transform = ImageTransform(518, 224, 14)

    def __init__(self, root, palette_key="none"):
        # The CelebAMask-HQ img2palette prompt always lists the full colour key, so palette_key changes nothing.
        from data import celebamask as m
        self.m, self.root = m, root
        self.palette = m.PALETTE
        self.num_ids = len(m.PALETTE)
        self.eval_ids = list(m.FACE_CLASS_IDS)
        self.class_names = list(m.CLASS_NAMES)
        self.snap_ids = list(range(self.num_ids))

    def keys(self, split, limit):
        return [str(i) for i in self.m.load_split_ids(self.root, split)[:limit]]

    def load(self, key):
        return self.m.load_photo(self.root, int(key)), self.m.load_label_map(self.root, int(key))

    def img2palette_prompt(self):
        return self.m.IMG2PALETTE_PROMPT

    def seg2img_prompt(self, label):
        return self.m.SEG2IMG_PROMPT


class CocoStuffEval:
    name = "cocostuff"
    default_root = "/datasets/segmentation"
    ignore_unlabeled = True    # standard COCO-Stuff protocol: unlabeled ground-truth pixels are ignored
    photo_question = ("Is this a realistic photograph (not a segmentation map, drawing or flat-colour image)? "
                      "Answer yes or no.")
    # Training transforms (data/configs/cocostuff/cocostuff_img2palette_seg2img.yaml).
    vae_transform = ImageTransform(512, 256, 16)
    vit_transform = ImageTransform(518, 224, 14)

    def __init__(self, root, palette_key="none"):
        from data import cocostuff as m
        self.m, self.root = m, root
        self.palette_key = palette_key
        self.images = {}  # split -> _Source, opened once (the train2017 zip index is large)
        self.labels = m._Source(os.path.join(root, "stuffthingmaps_trainval2017.zip"))
        self.palette = m.PALETTE
        self.num_ids = 183
        self.eval_ids = list(m.USED_IDS)
        self.class_names = list(m.CLASS_NAMES)
        self.snap_ids = [0] + list(m.USED_IDS)

    def keys(self, split, limit):
        split = {"val": "val2017", "train": "train2017"}.get(split, split)
        self.split = split
        return self.labels.names(split, ".png")[:limit]

    def load(self, key):
        if self.split not in self.images:
            self.images[self.split] = self.m._Source(os.path.join(self.root, f"{self.split}.zip"))
        photo = self.images[self.split].open(self.split, f"{key}.jpg").convert("RGB")
        label = self.m.load_label(self.labels.open(self.split, f"{key}.png"))
        # Bring both to the training-time size (photo bicubic, label nearest), as SegMultiTaskDataset.fit does.
        size = self.vae_transform.resize_transform(photo).size
        return photo.resize(size, Image.BICUBIC), np.asarray(Image.fromarray(label).resize(size, Image.NEAREST))

    def img2palette_prompt(self):
        return self.m.IMG2PALETTE_FULL_KEY_PROMPT if self.palette_key == "full" else self.m.IMG2PALETTE_PROMPT

    def seg2img_prompt(self, label):
        return self.m.seg2img_prompt(label)


DATASETS = {"celebamask": CelebAMaskEval, "cocostuff": CocoStuffEval}


def get_dataset(args):
    cls = DATASETS[args.dataset]
    return cls(args.dataset_root or cls.default_root, palette_key=args.palette_key)


# ----------------------------------------------------------------------------- model
def add_model_args(parser):
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--dataset_root", default=None, help="Defaults per dataset (container paths).")
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "/models/SenseNova-Vision-7B-MoT"))
    parser.add_argument("--adapter", default=None,
                        help="LoRA checkpoint dir (adapter_model.safetensors + adapter_config.json); omit for the base model.")
    parser.add_argument("--max_mem_per_gpu", default="30GiB")
    parser.add_argument("--palette_key", default="none", choices=["none", "full"],
                        help="img2palette prompt (also the seg2img round trip): none = as trained; full = list "
                             "the colour of every class, for zero-shot runs on COCO-Stuff, whose training prompt "
                             "has no key. CelebAMask-HQ always lists its 18 classes.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None, help="First N images of the split.")
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument("--cfg_text_scale", type=float, default=None)
    parser.add_argument("--cfg_img_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)


def generation_params(args, ds):
    params = {k: getattr(args, k) for k in ("num_timesteps", "cfg_text_scale", "cfg_img_scale")
              if getattr(args, k) is not None}
    return params, {"vae_transform": ds.vae_transform, "vit_transform": ds.vit_transform}


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


# ----------------------------------------------------------------------------- generation / metrics
def palette_to_label(ds, image, size):
    """Snap every pixel to the nearest palette colour of the dataset; also return the share of pixels farther
    than 20 (RGB Euclidean) from every palette colour, i.e. how far the output strays from the format."""
    colours = ds.palette[ds.snap_ids].astype(np.int32)
    rgb = np.asarray(image.convert("RGB").resize(size, Image.NEAREST), dtype=np.int32)
    dist = ((rgb[:, :, None, :] - colours[None, None]) ** 2).sum(-1)
    label = np.asarray(ds.snap_ids, dtype=np.uint8)[dist.argmin(-1)]
    return label, float((np.sqrt(dist.min(-1)) > 20).mean())


def segment_to_palette(model, ds, photo, seed, params, transforms):
    """Run the img2palette mode on a photo: returns (raw generated map, label map at the photo's size, off-palette share)."""
    raw = model.generate(contents=[{"type": "image", "value": photo},
                                   {"type": "text", "value": ds.img2palette_prompt()}],
                         mode="dense_perception", noise_seed=seed, **params, **transforms).convert("RGB")
    label, off = palette_to_label(ds, raw, photo.size)
    return raw, label, off


def confusion(ds, gt, pred):
    gt, pred = gt.astype(np.int64).ravel(), pred.astype(np.int64).ravel()
    if ds.ignore_unlabeled:
        keep = gt != 0
        gt, pred = gt[keep], pred[keep]
    n = ds.num_ids
    return np.bincount(gt * n + pred, minlength=n * n).reshape(n, n)


def summarize(ds, conf):
    """mIoU over the dataset's evaluated classes (classes absent from both GT and prediction are skipped)."""
    tp = np.diag(conf).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (conf.sum(0) + conf.sum(1) - tp)
    ids = ds.eval_ids
    return {
        "mIoU": float(np.nanmean(iou[ids])),
        "classes_scored": int(np.sum(~np.isnan(iou[ids]))),
        "pixel_accuracy": float(tp.sum() / conf.sum()) if conf.sum() else float("nan"),
        "per_class_IoU": {ds.class_names[i]: (None if np.isnan(iou[i]) else float(iou[i])) for i in ids},
    }
