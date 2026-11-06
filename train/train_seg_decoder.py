# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Train the segmentation decoder (modeling/seg_decoder.py) on one-step map latents of a frozen model.

Half of the samples use ground-truth map latents, the other half latents predicted in one function evaluation at a
uniform t, so that the decoder sees predictions of varying quality. Loss: cross-entropy + Dice. One GPU.

--flow picks the model that predicts the latents:
    symm      a model trained with the symmetric flow (docs/SYMMETRIC.md)
    existing  a model trained on the img2palette mode, which is the control: the same decoder on the same data
              without the symmetric flow, so a gain can be attributed to the decoder or to the flow

    python -m train.train_seg_decoder --dataset celebamask --flow symm --adapter <dir> --output_dir results/symm/dec \\
        --size S --epochs 30
Evaluate with eval_seg_img2palette --decoder <output_dir>/decoder.pt (with --symm for --flow symm). Re-running with
the same --output_dir resumes.

--legend: maps in the colour-legend format (mode symm_legend with --flow symm; img2palette_legend with --flow existing).
Legend colours belong to positions, not classes, so the decoder predicts the legend slot (0 = background / unlabeled,
k = the k-th legend entry); evaluation maps slots to classes through the generated legend. Training uses the
ground-truth legend.

--cache_dir: predicting a latent costs a forward pass of the whole model, which dominates training. The cache pass
stores, per image, the ground-truth latent and --cache_predicted latents predicted at uniform t, once; training then
reads latents from memory. Fewer distinct predictions per image, much faster updates.

--cache_steps N (--flow symm --legend): the predicted latents are the sampler's output after N steps, with the eval's
settings (--cache_cfg, --cache_hold_known, ViT prefix, ground-truth legend), instead of one-step estimates: a decoder
for multi-step segmentation must read the sampler's sharp output, not the blurry average of a single step. That
costs N (or 2N with guidance) forward passes per image, so the cache can be built by several jobs at once
(--cache_shard i --cache_shards n --cache_only, each writing its own parts), and training starts once every part
exists.

--size vae starts from the VAE's own decoder (modeling/seg_decoder.py), its RGB output layer replaced.

--val_latents DIR scores the decoder after training on latents stored by eval_seg_img2palette --save_latents (each with
its generated legend), so the decoder and the VAE readout are compared on the very same latents: <output_dir>/val/
pred/<key>.png and metrics.json. --eval_only scores the saved decoder.pt without training.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image

from inference.benchmark.seg_eval_common import (add_model_args, confusion, decode_legend_slots, get_dataset,
                                                  load_model, summarize, symm_flow)
from modeling.seg_decoder import SIZES, SegDecoder, init_from_vae, unpack_latent


class Pairs(Dataset):
    def __init__(self, ds, keys):
        self.ds, self.keys = ds, keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self.ds.load(self.keys[index])


def segmentation_loss(logits, label, ignore_unlabeled):
    """Cross-entropy + Dice over the classes present in the label, as (ce, dice)."""
    logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
    valid = label != 0 if ignore_unlabeled else torch.ones_like(label, dtype=torch.bool)
    if not valid.any():  # an image with no labelled pixel: cross-entropy would be nan
        return logits.sum() * 0, logits.sum() * 0
    ce = F.cross_entropy(logits, label, ignore_index=0 if ignore_unlabeled else -100)
    probs = logits.softmax(1)[0]
    scores = []
    for cid in label[valid].unique():
        truth = ((label[0] == cid) & valid[0]).float()
        prob = probs[cid] * valid[0]
        scores.append(1 - (2 * (prob * truth).sum() + 1) / (prob.sum() + truth.sum() + 1))
    dice = torch.stack(scores).mean() if scores else logits.sum() * 0
    return ce, dice


DEFAULT_MAX_SLOTS = {"celebamask": 18, "cocostuff": 40}


def slot_target(label, order, max_slots):
    """Legend slot per pixel: k for the k-th class of the legend order (k <= max_slots), else 0."""
    lut = np.zeros(256, dtype=np.int64)
    for slot, cid in enumerate(order[:max_slots], 1):
        lut[cid] = slot
    return lut[np.asarray(label)]


class Targets(Dataset):
    """(index, per-pixel target) per key; the target is computed in the data-loader workers."""

    def __init__(self, ds, keys, target_fn):
        self.ds, self.keys, self.target_fn = ds, keys, target_fn

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return index, self.target_fn(self.ds.load(self.keys[index])[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.set_defaults(split="train")
    parser.add_argument("--flow", required=True, choices=("symm", "existing"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--size", default="S", choices=list(SIZES))
    parser.add_argument("--epochs", type=float, default=30)
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many updates.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--predicted_ratio", type=float, default=0.5, help="Share of predicted latents.")
    parser.add_argument("--symm_vit_prob", type=float, default=0.5, help="--flow symm: share with the ViT prefix.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--legend", action="store_true", help="Colour-legend maps; the decoder predicts legend slots.")
    parser.add_argument("--max_slots", type=int, default=None,
                        help="--legend: legend entries the decoder tells apart (default 18 CelebAMask-HQ, 40 COCO-Stuff); "
                             "pixels of later entries are background (CelebAMask-HQ) or ignored (COCO-Stuff).")
    parser.add_argument("--cache_dir", default=None, help="Predict latents once into this directory, then train from it.")
    parser.add_argument("--cache_predicted", type=int, default=1, help="--cache_dir: predicted latents per image.")
    parser.add_argument("--cache_steps", type=int, default=0,
                        help="--cache_dir with --flow symm --legend: predicted latents from the sampler after this many "
                             "steps (as the eval runs it) instead of one step at uniform t (0).")
    parser.add_argument("--cache_cfg", type=float, default=1.0, help="--cache_steps: text guidance scale.")
    parser.add_argument("--cache_hold_known", action="store_true", help="--cache_steps: keep the photo clean.")
    parser.add_argument("--cache_shard", type=int, default=0, help="Build only the cache parts p with p %% n == i.")
    parser.add_argument("--cache_shards", type=int, default=1)
    parser.add_argument("--cache_only", action="store_true", help="Build this shard's cache parts, then exit.")
    parser.add_argument("--val_latents", default=None,
                        help="Directory of latents saved by eval_seg_img2palette --save_latents: score the decoder on "
                             "them after training (--legend only).")
    parser.add_argument("--eval_only", action="store_true", help="Score the saved decoder.pt on --val_latents.")
    args = parser.parse_args()
    args.symm = args.flow == "symm"
    if args.max_slots is None:
        args.max_slots = DEFAULT_MAX_SLOTS[args.dataset]
    if args.cache_steps and not (args.cache_dir and args.symm and args.legend):
        parser.error("--cache_steps needs --cache_dir, --flow symm and --legend")
    if (args.cache_only or args.cache_shards > 1) and not args.cache_dir:
        parser.error("--cache_only and --cache_shards need --cache_dir")
    if (args.val_latents or args.eval_only) and not args.legend:
        parser.error("--val_latents scores slot decoders (--legend)")
    if args.eval_only and not args.val_latents:
        parser.error("--eval_only needs --val_latents")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    ds = get_dataset(args)
    keys = ds.keys(args.split, args.limit)
    model, info = load_model(args)
    flow = symm_flow(model, ds, args)
    device = flow.device
    if args.legend:
        prompt = ds.symm_legend_prompt() if args.symm else ds.legend_prompt()
    else:
        prompt = ds.symm_prompt() if args.symm else ds.img2palette_prompt()
    num_classes = args.max_slots + 1 if args.legend else ds.num_ids

    def target_and_map(label):
        """(target per pixel, map image for the VAE, legend text or None) for a label map."""
        if not args.legend:
            return np.asarray(label, dtype=np.int64), ds.m.palette_image(np.asarray(label)), None
        legend_map, legend, order = ds.legend_map(np.asarray(label))
        return slot_target(label, order, args.max_slots), legend_map, legend

    def predict(photo, mask_latent, legend, t, seed, use_vit):
        if args.cache_steps:  # the eval's sampler: ViT prefix, ground-truth legend, guidance, photo optionally held
            latent, _ = flow.segment_latent(photo, prompt, seed, num_steps=args.cache_steps, use_vit=True,
                                            legend=legend, cfg_scale=args.cache_cfg, hold_known=args.cache_hold_known)
            return latent
        if args.symm:
            return flow.one_step_from_pair(photo, mask_latent, t, prompt, seed, photo if use_vit else None, legend)
        latent, _ = flow.conditional_latent(photo, prompt, seed, noise_level=t, mask_latent=mask_latent, legend=legend)
        return latent

    cache = None
    if args.cache_dir and not args.eval_only:
        cache = build_cache(args, ds, keys, flow, target_and_map, predict)
        if args.cache_only:
            print(f"[decoder] cache shard {args.cache_shard}/{args.cache_shards} done", flush=True)
            os._exit(0)

    decoder = SegDecoder(model.model.latent_channel, num_classes, args.size).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.0)
    ckpt_path = os.path.join(args.output_dir, "decoder.pt")
    step = 0
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        print(f"[decoder] resuming at step {step}", flush=True)
    elif args.eval_only:
        raise SystemExit(f"--eval_only: no {ckpt_path}")
    elif args.size == "vae":
        copied = init_from_vae(decoder, os.path.join(args.model_path, "ae.safetensors"))
        print(f"[decoder] initialised from the VAE decoder ({copied} tensors)", flush=True)
    if args.eval_only:
        score_val(args, flow, decoder)
        os._exit(0)
    total_steps = math.ceil(args.epochs * len(keys) / args.batch_size)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"[decoder] flow={args.flow} legend={args.legend} classes={num_classes} cache={bool(cache)} "
          f"size={args.size} images={len(keys)} steps={total_steps} "
          f"params={sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M", flush=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({**vars(args), **info}, f, indent=2, default=str)

    def save():
        tmp = ckpt_path + ".tmp"
        torch.save({"decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "config": decoder.config}, tmp)
        os.replace(tmp, ckpt_path)

    def batches():
        epoch = step
        while True:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(keys), generator=generator).tolist()
            source = Targets(ds, keys, lambda label: target_and_map(label)[0]) if cache else Pairs(ds, keys)
            loader = DataLoader(torch.utils.data.Subset(source, order), batch_size=None, num_workers=args.num_workers,
                                prefetch_factor=4 if args.num_workers else None)
            yield from loader
            epoch += 1

    decoder.train()
    stream = batches()
    window = {"ce": [], "dice": [], "predicted": []}
    start = time.time()
    while step < total_steps:
        for _ in range(args.batch_size):
            predicted = rng.random() < args.predicted_ratio
            if cache:
                index, target = next(stream)
                entry = cache[index]
                latent = rng.choice(entry["pred"]) if predicted else entry["gt"]
                latent, shape = latent.to(device).float(), entry["shape"]
            else:
                photo, label = next(stream)
                target, mask_image, legend = target_and_map(label)
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    mask_latent, shape = flow.encode(mask_image)
                    if predicted:
                        latent = predict(photo, mask_latent, legend, rng.random(), rng.getrandbits(32),
                                         rng.random() < args.symm_vit_prob)
                    else:
                        latent = mask_latent
            target = torch.as_tensor(target, device=device)[None]
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                logits = decoder(unpack_latent(model.model, latent, shape))
                ce, dice = segmentation_loss(logits.float(), target, ds.ignore_unlabeled)
                ((ce + dice) / args.batch_size).backward()
            window["ce"].append(ce.item())
            window["dice"].append(dice.item())
            window["predicted"].append(float(predicted))
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0 or step == 1:
            record = {"step": step, "ce": float(np.mean(window["ce"])), "dice": float(np.mean(window["dice"])),
                      "predicted_share": float(np.mean(window["predicted"])), "grad_norm": float(grad_norm),
                      "seconds": round(time.time() - start, 1)}
            print(f"[decoder] {json.dumps(record)}", flush=True)
            with open(os.path.join(args.output_dir, "train_log.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
            window = {"ce": [], "dice": [], "predicted": []}
        if step % args.save_every == 0:
            save()
    save()
    print(f"[decoder] saved {ckpt_path}", flush=True)
    if args.val_latents:
        score_val(args, flow, decoder)
    os._exit(0)  # data-loader workers iterate a generator that never ends


@torch.no_grad()
def score_val(args, flow, decoder):
    """mIoU of the slot decoder on latents stored by eval_seg_img2palette --save_latents, read through the legend each
    was drawn with; writes <output_dir>/val/pred/<key>.png and metrics.json. Its own dataset instance: the COCO-Stuff
    loader follows the split of its last keys() call, and training uses the train split."""
    ds = get_dataset(args)
    split_keys = ds.keys("val", None)
    present = {f[:-3] for f in os.listdir(args.val_latents) if f.endswith(".pt")}
    keys = [k for k in split_keys if k in present]
    out = os.path.join(args.output_dir, "val")
    os.makedirs(os.path.join(out, "pred"), exist_ok=True)
    flow.decoder = decoder.eval()
    conf = np.zeros((ds.num_ids, ds.num_ids), np.int64)
    start = time.time()
    for n, key in enumerate(keys, 1):
        entry = torch.load(os.path.join(args.val_latents, f"{key}.pt"), map_location="cpu", weights_only=False)
        photo, gt = ds.load(key)
        label, _ = decode_legend_slots(flow, ds, entry["latent"].to(flow.device).float(), entry["shape"], photo.size,
                                       entry["legend"])
        Image.fromarray(label).save(os.path.join(out, "pred", f"{key}.png"))
        conf += confusion(ds, gt, label)
        if n % 200 == 0:
            print(f"[decoder] scored {n}/{len(keys)} mIoU so far {summarize(ds, conf)['mIoU']:.4f} "
                  f"({time.time() - start:.0f}s)", flush=True)
    metrics = {"num_images": len(keys), "val_latents": args.val_latents, **summarize(ds, conf)}
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"[decoder] val: mIoU {metrics['mIoU']:.4f} pixel acc {metrics['pixel_accuracy']:.4f} on {len(keys)} "
          f"images -> {out}/metrics.json", flush=True)


def build_cache(args, ds, keys, flow, target_and_map, predict, part_size=256):
    """Ground-truth and predicted latents for every key, in parts of part_size images under args.cache_dir (resumes
    by part). Returns the list of {"gt", "pred": [...], "shape"} in key order, on the CPU in BF16."""
    os.makedirs(args.cache_dir, exist_ok=True)
    parts = math.ceil(len(keys) / part_size)
    start = time.time()
    mine = [part for part in range(parts) if part % args.cache_shards == args.cache_shard]
    for done, part in enumerate(mine, 1):
        path = os.path.join(args.cache_dir, f"part_{part:05d}.pt")
        if os.path.exists(path):
            continue
        part_rng = random.Random(f"{args.seed}-{part}")
        entries = []
        for key in keys[part * part_size : (part + 1) * part_size]:
            photo, label = ds.load(key)
            _, mask_image, legend = target_and_map(label)
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                mask_latent, shape = flow.encode(mask_image)
                pred = [predict(photo, mask_latent, legend, part_rng.random(), part_rng.getrandbits(32),
                                part_rng.random() < args.symm_vit_prob).to("cpu", torch.bfloat16)
                        for _ in range(args.cache_predicted)]
            entries.append({"key": key, "gt": mask_latent.to("cpu", torch.bfloat16), "pred": pred, "shape": shape})
        tmp = f"{path}.{os.getpid()}.tmp"  # per process: shards share the directory
        torch.save(entries, tmp)
        os.replace(tmp, path)
        eta = (time.time() - start) / done * (len(mine) - done) / 60
        print(f"[decoder] cache part {part} ({done}/{len(mine)} of shard {args.cache_shard}) eta={eta:.0f}min",
              flush=True)
    if args.cache_only:
        return None
    cache = []
    for part in range(parts):
        entries = torch.load(os.path.join(args.cache_dir, f"part_{part:05d}.pt"), weights_only=False)
        cache.extend(entries)
    assert [e["key"] for e in cache] == list(keys), "the cache was built for other keys; use a new --cache_dir"
    return cache


if __name__ == "__main__":
    main()
