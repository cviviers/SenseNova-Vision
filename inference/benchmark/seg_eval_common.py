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
    symm_beta = 10.0  # map dequantization of the symm configs

    def __init__(self, root):
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

    def symm_prompt(self):
        return self.m.SYMM_PROMPT

    def symm_legend_prompt(self):
        return self.m.SYMM_LEGEND_PROMPT

    def legend_map(self, label):
        """(map in the legend's colours, legend text, class ids in legend order) for a label map, as in training."""
        _, _, image, info = self.m.legend_task(label)
        return image, info["answer"], info["classes"]

    def seg2img_prompt(self, label):
        return self.m.seg2img_prompt(label)

    def seg2img_region_prompt(self, label):
        # Zero-shot prompt of the earlier mask2face sweep: region names, no colours (colour keys make the
        # base model copy the map).
        from inference.benchmark.batch_celebamask_mask2face import make_prompt
        return make_prompt("celeb_dynamic", label)

    def legend_prompt(self):
        return self.m.LEGEND_PROMPT

    def legend_class_id(self, phrase):
        """Class id of a legend phrase ('left eye', 'earring-1', 'skin'), or None if it names no face class."""
        if not hasattr(self, "_legend_ids"):
            from inference.benchmark.batch_celebamask_segment import GENERIC_ALIASES
            self._legend_ids = {**GENERIC_ALIASES, **{p: c for c, p in enumerate(self.m.PHRASES) if c}}
        base, _, idx = phrase.rpartition("-")
        name = base if base and idx.isdigit() else phrase  # drop the instance number
        return self._legend_ids.get(" ".join(name.lower().replace("-", " ").split()))


# Region-name seg2img prompt for COCO-Stuff, worded like celeb_dynamic (batch_celebamask_mask2face.py).
COCO_REGION_PROMPT = (
    "Convert this segmentation map into a photorealistic COCO-style photograph. Strictly preserve the geometry "
    "and boundaries of all regions while generating natural textures, materials, and soft photographic lighting. "
    "Do not alter the scene layout or retain segmentation colors. The segmentation map contains these regions: "
)


class CocoStuffEval:
    name = "cocostuff"
    default_root = "/datasets/segmentation"
    ignore_unlabeled = True    # standard COCO-Stuff protocol: unlabeled ground-truth pixels are ignored
    photo_question = ("Is this a realistic photograph (not a segmentation map, drawing or flat-colour image)? "
                      "Answer yes or no.")
    # Training transforms (data/configs/cocostuff/cocostuff_img2palette_seg2img.yaml).
    vae_transform = ImageTransform(512, 256, 16)
    vit_transform = ImageTransform(518, 224, 14)
    symm_beta = 6.0  # map dequantization of the symm configs

    def __init__(self, root):
        from data import cocostuff as m
        self.m, self.root = m, root
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
        return self.m.IMG2PALETTE_PROMPT

    def symm_prompt(self):
        return self.m.SYMM_PROMPT

    def symm_legend_prompt(self):
        return self.m.SYMM_LEGEND_PROMPT

    def legend_map(self, label):
        """(map in the legend's colours, legend text, class ids in legend order) for a label map, as in training."""
        _, _, image, info = self.m.legend_task(label)
        return image, info["answer"], info["classes"]

    def seg2img_prompt(self, label):
        return self.m.seg2img_prompt(label)

    def seg2img_region_prompt(self, label):
        return COCO_REGION_PROMPT + ", ".join(self.m.phrase(int(c)) for c in np.unique(label) if c) + "."

    def legend_prompt(self):
        return self.m.LEGEND_PROMPT

    def legend_class_id(self, phrase):
        """Class id of a legend phrase ('person-0', 'sky', 'wall-other-merged'), or None if it names no class."""
        if not hasattr(self, "_legend_ids"):
            self._legend_ids = {self.m.legend_name(c): c for c in self.m.USED_IDS}
        base, _, idx = phrase.rpartition("-")
        name = base if base and idx.isdigit() else phrase  # drop the instance number
        name = name.lower().replace("-merged", "").replace("-other", "").replace("-stuff", "").replace("-", " ")
        return self._legend_ids.get(" ".join(name.split()))


DATASETS ={"celebamask": CelebAMaskEval, "cocostuff": CocoStuffEval}


def get_dataset(args):
    cls = DATASETS[args.dataset]
    return cls(args.dataset_root or cls.default_root)


# ----------------------------------------------------------------------------- model
SYMM_EMBEDDING_FILE = "symm_embedding.safetensors"


def add_model_args(parser):
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--dataset_root", default=None, help="Defaults per dataset (container paths).")
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "/models/SenseNova-Vision-7B-MoT"))
    parser.add_argument("--adapter", default=None,
                        help="LoRA checkpoint dir (adapter_model.safetensors + adapter_config.json); omit for the base model.")
    parser.add_argument("--max_mem_per_gpu", default="30GiB")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None, help="First N images of the split.")
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument("--cfg_text_scale", type=float, default=None)
    parser.add_argument("--cfg_img_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symm", action="store_true", help="Model trained with the symmetric flow (mode symm).")
    parser.add_argument("--symm_seg_steps", type=int, default=1, help="--symm: segmentation steps.")
    parser.add_argument("--symm_gen_steps", type=int, default=25, help="--symm: synthesis steps.")
    parser.add_argument("--symm_no_vit", action="store_true", help="--symm: segment without the photo's ViT prefix.")
    parser.add_argument("--symm_beta", type=float, default=None,
                        help="--symm: map dequantization in synthesis, 0-255 scale (default: as in training).")
    parser.add_argument("--symm_recompute_known", action="store_true",
                        help="--symm: compute the known stream at every step instead of integrating it.")
    parser.add_argument("--decoder", default=None,
                        help="--symm or one-step eval: segmentation decoder checkpoint (train/train_seg_decoder.py).")
    parser.add_argument("--symm_cfg_scale", type=float, default=1.0,
                        help="--symm: text guidance in both directions (1 = none); needs a model trained with "
                             "symm_text_dropout > 0.")
    parser.add_argument("--oracle_legend", action="store_true",
                        help="--symm --format legend: use the ground-truth colour legend instead of generating one "
                             "(diagnostic: separates map quality from legend errors).")
    parser.add_argument("--legend_cache", default=None,
                        help="--symm --format legend: directory of generated legends (<key>.txt), reused by later "
                             "evaluations of the same model on the same photos.")


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
    symm = {k[len("symm_embedding."):]: v for k, v in adapter.items() if k.startswith("symm_embedding.")}
    if symm:
        bagel.enable_symm_embedding(symm)
    print(f"[eval] applied {merged} LoRA modules from {adapter_dir} (scale={scale}, step={cfg.get('step')})")
    return cfg


def load_model(args):
    model = SenseNovaVisionModel(model_path=args.model_path, dtype="bf16", max_mem_per_gpu=args.max_mem_per_gpu)
    info = {"model_path": args.model_path, "adapter": args.adapter}
    if args.adapter:
        info["adapter_config"] = apply_lora_inplace(model.model, args.adapter)
    symm_path = os.path.join(args.model_path, SYMM_EMBEDDING_FILE)  # written by tools/export_full_checkpoint.py
    if os.path.exists(symm_path):
        model.model.enable_symm_embedding(load_file(symm_path))
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


def symm_flow(model, ds, args):
    """SymmetricFlow on a loaded model, with the dataset's training transforms and the decoder if one is given."""
    from inference.symmetric import SymmetricFlow

    flow = SymmetricFlow(model, ds.vae_transform, ds.vit_transform)
    flow.decoder = None
    if getattr(args, "decoder", None):
        from modeling.seg_decoder import load_seg_decoder
        flow.decoder = load_seg_decoder(args.decoder, flow.device)
    return flow


def segment_symm(flow, ds, photo, seed, args):
    """Segment through SymmetricFlow: returns (raw map, label map at the photo's size, off-palette share).

    With --symm this is the symmetric flow. Without it (--decoder on a model trained on img2palette) it is the
    one-step conditional estimate, the control for the symmetric flow."""
    kwargs = dict(seed=seed, num_steps=args.symm_seg_steps, use_vit=not args.symm_no_vit,
                  recompute_known=args.symm_recompute_known, cfg_scale=args.symm_cfg_scale)
    if flow.decoder is None:
        raw = flow.segment(photo, ds.symm_prompt(), **kwargs).convert("RGB")
        label, off = palette_to_label(ds, raw, photo.size)
        return raw, label, off
    if args.symm:
        latent, shape = flow.segment_latent(photo, ds.symm_prompt(), **kwargs)
    else:
        latent, shape = flow.conditional_latent(photo, ds.img2palette_prompt(), seed)
    label = decode_label(flow, ds, latent, shape, photo.size)
    return Image.fromarray(ds.palette[label]), label, 0.0


def decoder_logits(flow, latent, shape, size):
    """Decoder logits (1, classes, H, W) for latent tokens, upsampled to PIL size (W, H)."""
    from modeling.seg_decoder import unpack_latent

    z = unpack_latent(flow.model, latent, shape)
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        logits = flow.decoder(z.to(next(flow.decoder.parameters()).dtype))
    return torch.nn.functional.interpolate(logits.float(), size=size[::-1], mode="bilinear", align_corners=False)


def decode_label(flow, ds, latent, shape, size):
    """Class ids from latent tokens through the segmentation decoder, at PIL size (W, H)."""
    logits = decoder_logits(flow, latent, shape, size)
    allowed = torch.zeros(ds.num_ids, dtype=torch.bool, device=logits.device)
    allowed[ds.eval_ids if ds.ignore_unlabeled else list(range(ds.num_ids))] = True
    logits[:, ~allowed] = float("-inf")
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def legend_to_label(ds, raw, text, size):
    """Read a map drawn in a colour legend's colours through the legend, at PIL size (W, H). Returns (label map,
    {"legend_entries", "unmatched": phrases that name no dataset class, "legend_ids": class id or None per entry})."""
    from utils.parsing_output import class_mask_from_raw_output
    label = np.zeros(size[::-1], dtype=np.uint8)
    parsed = class_mask_from_raw_output(raw.convert("RGB").resize(size, Image.NEAREST), text)
    ids = [ds.legend_class_id(phrase) for phrase in parsed["gcg_phrases"]]
    for index, cid in enumerate(ids):
        if cid is not None:
            label[parsed["class_mask"] == index] = cid
    unmatched = [p for p, c in zip(parsed["gcg_phrases"], ids) if c is None]
    return label, {"legend_entries": len(ids), "unmatched": unmatched, "legend_ids": ids}


def legend_presence(ds, ids, gt_label):
    """(true positives, false positives, false negatives) of the classes a legend names, against the evaluated
    classes present in the ground truth."""
    pred = {c for c in ids if c is not None}
    gt = {int(c) for c in np.unique(gt_label)} & set(ds.eval_ids)
    return len(pred & gt), len(pred - gt), len(gt - pred)


def segment_with_legend(model, ds, photo, seed, params, transforms, max_text_tokens=1024, gt_label=None):
    """Run the released model's generic-segmentation format (ds.legend_prompt(): colour legend text, then a mask in
    those colours) and decode the mask through the legend. Returns (raw map, label map at the photo's size, info)
    with info = {"text", "legend_entries", "unmatched": legend phrases that name no dataset class}, plus the legend's
    class presence counts (legend_tp/fp/fn) when gt_label is given."""
    out = model.generate(contents=[{"type": "image", "value": photo}, {"type": "text", "value": ds.legend_prompt()}],
                         mode="caption_generate", noise_seed=seed, return_intermediate_outputs=True,
                         max_think_token_n=max_text_tokens, **params, **transforms)
    text, raw = out.get("text") or "", out.get("image")
    if raw is None:
        label = np.zeros(photo.size[::-1], dtype=np.uint8)
        return Image.new("RGB", photo.size), label, {"text": text, "legend_entries": 0, "unmatched": []}
    raw = raw.convert("RGB")
    label, info = legend_to_label(ds, raw, text, photo.size)
    if gt_label is not None:
        info.update(zip(("legend_tp", "legend_fp", "legend_fn"), legend_presence(ds, info["legend_ids"], gt_label)))
    del info["legend_ids"]
    return raw, label, {"text": text, **info}


def symm_legend_text(flow, ds, photo, args, key=None, gt_label=None, conditional=False):
    """Colour legend for segmenting a photo with a symm_legend model: the ground truth with --oracle_legend, else
    generated from the ViT prefix (conditional: from the photo's VAE + ViT prefix, as an img2palette_legend model
    writes it). With --legend_cache and a key, a generated legend is stored and reused."""
    if args.oracle_legend:
        return ds.legend_map(gt_label)[1]
    path = os.path.join(args.legend_cache, f"{key}.txt") if args.legend_cache and key is not None else None
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read()
    max_tokens = getattr(args, "max_text_tokens", 1024)
    text = (flow.generate_conditional_legend(photo, ds.legend_prompt(), max_tokens) if conditional else
            flow.generate_legend(photo, ds.symm_legend_prompt(), max_tokens))
    if path:
        os.makedirs(args.legend_cache, exist_ok=True)
        with open(path + ".tmp", "w") as f:
            f.write(text)
        os.replace(path + ".tmp", path)
    return text


def segment_symm_legend(flow, ds, photo, seed, args, key=None, gt_label=None):
    """Segment with a symm_legend model: the colour legend first, then the joint flow, read through the legend (or
    through a slot decoder, train_seg_decoder.py --legend). Returns (raw map, label map at the photo's size, info) as
    segment_with_legend. key: the photo's dataset key, for --legend_cache (None for photos that are not the
    dataset's, e.g. generated ones)."""
    if args.symm_no_vit and not args.oracle_legend:
        raise ValueError("--symm_no_vit with --format legend needs --oracle_legend: the legend is read from the ViT prefix")
    text = symm_legend_text(flow, ds, photo, args, key, gt_label)
    kwargs = dict(seed=seed, num_steps=args.symm_seg_steps, use_vit=not args.symm_no_vit,
                  recompute_known=args.symm_recompute_known, legend=text, cfg_scale=args.symm_cfg_scale)
    if flow.decoder is None:
        raw = flow.segment(photo, ds.symm_legend_prompt(), **kwargs).convert("RGB")
        label, info = legend_to_label(ds, raw, text, photo.size)
    else:
        latent, shape = flow.segment_latent(photo, ds.symm_legend_prompt(), **kwargs)
        label, info = decode_legend_slots(flow, ds, latent, shape, photo.size, text)
        raw = Image.fromarray(ds.palette[label])
    if gt_label is not None:
        info.update(zip(("legend_tp", "legend_fp", "legend_fn"), legend_presence(ds, info["legend_ids"], gt_label)))
    del info["legend_ids"]
    return raw, label, {"text": text, **info}


def segment_legend_conditional(flow, ds, photo, seed, args, key=None, gt_label=None):
    """The one-step control for a slot decoder on a model trained on img2palette_legend (v4): its own legend from the
    photo prefix, then one step from noise with no guidance, read by the decoder through the legend."""
    text = symm_legend_text(flow, ds, photo, args, key, gt_label, conditional=True)
    latent, shape = flow.conditional_latent(photo, ds.legend_prompt(), seed, legend=text)
    label, info = decode_legend_slots(flow, ds, latent, shape, photo.size, text)
    if gt_label is not None:
        info.update(zip(("legend_tp", "legend_fp", "legend_fn"), legend_presence(ds, info["legend_ids"], gt_label)))
    del info["legend_ids"]
    return Image.fromarray(ds.palette[label]), label, {"text": text, **info}


def decode_legend_slots(flow, ds, latent, shape, size, text):
    """Class ids through a slot decoder (train_seg_decoder.py --legend): slot 0 is background / unlabeled and slot k
    the k-th legend entry. Slots past the legend's length are excluded."""
    from utils import extract_ordered_caption_instances

    phrases = [phrase for phrase, _ in extract_ordered_caption_instances(text)]
    ids = [ds.legend_class_id(p) for p in phrases]
    logits = decoder_logits(flow, latent, shape, size)
    n = min(len(ids), logits.shape[1] - 1)
    logits[:, n + 1:] = float("-inf")
    lut = np.zeros(logits.shape[1], dtype=np.uint8)
    lut[1: n + 1] = [c or 0 for c in ids[:n]]
    label = lut[logits.argmax(1)[0].cpu().numpy()]
    return label, {"legend_entries": len(ids), "unmatched": [p for p, c in zip(phrases, ids) if c is None],
                   "legend_ids": ids}


def presence_f1(logs):
    """Micro precision, recall and F1 of legend class presence over records with legend_tp/fp/fn, or {}."""
    logs = [l for l in logs if "legend_tp" in l]
    if not logs:
        return {}
    tp, fp, fn = (sum(l[k] for l in logs) for k in ("legend_tp", "legend_fp", "legend_fn"))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"legend_precision": precision, "legend_recall": recall,
            "legend_f1": 2 * precision * recall / max(precision + recall, 1e-9)}


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
