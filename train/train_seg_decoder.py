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

from inference.benchmark.seg_eval_common import add_model_args, get_dataset, load_model, symm_flow
from modeling.seg_decoder import SIZES, SegDecoder, unpack_latent


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
    args = parser.parse_args()
    args.symm = args.flow == "symm"

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    ds = get_dataset(args)
    keys = ds.keys(args.split, args.limit)
    model, info = load_model(args)
    flow = symm_flow(model, ds, args)
    device = flow.device
    prompt = ds.symm_prompt() if args.symm else ds.img2palette_prompt()

    decoder = SegDecoder(model.model.latent_channel, ds.num_ids, args.size).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.0)
    ckpt_path = os.path.join(args.output_dir, "decoder.pt")
    step = 0
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        print(f"[decoder] resuming at step {step}", flush=True)
    total_steps = math.ceil(args.epochs * len(keys) / args.batch_size)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"[decoder] flow={args.flow} size={args.size} images={len(keys)} steps={total_steps} "
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
            loader = DataLoader(Pairs(ds, [keys[i] for i in order]), batch_size=None, num_workers=args.num_workers,
                                prefetch_factor=4 if args.num_workers else None)
            yield from loader
            epoch += 1

    decoder.train()
    stream = batches()
    window = {"ce": [], "dice": [], "predicted": []}
    start = time.time()
    while step < total_steps:
        for _ in range(args.batch_size):
            photo, label = next(stream)
            predicted = rng.random() < args.predicted_ratio
            mask_image = ds.m.palette_image(np.asarray(label))
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                mask_latent, shape = flow.encode(mask_image)
                if predicted:
                    t = rng.random()
                    seed = rng.getrandbits(32)
                    if args.symm:
                        vit_image = photo if rng.random() < args.symm_vit_prob else None
                        latent = flow.one_step_from_pair(photo, mask_latent, t, prompt, seed, vit_image)
                    else:
                        latent, _ = flow.conditional_latent(photo, prompt, seed, noise_level=t, mask_latent=mask_latent)
                else:
                    latent = mask_latent
            target = torch.as_tensor(np.asarray(label, dtype=np.int64), device=device)[None]
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
    os._exit(0)  # data-loader workers iterate a generator that never ends


if __name__ == "__main__":
    main()
