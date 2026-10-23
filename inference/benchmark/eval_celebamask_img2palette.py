# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the img2palette mode (face photo -> CelebAMask-HQ palette map) on a CelebAMask-HQ split.

For every photo: generate with IMG2PALETTE_PROMPT (the training prompt, data/celebamask.py) in
dense-perception mode, snap the output to the official palette, and score it against the
ground-truth label map. The per-class report comes from tools/evaluation/segment/evaluate_celebamask.py,
so numbers are comparable with the zero-shot segmentation experiments.

Outputs (resumable; existing pred/<id>.png are skipped):
    raw/<id>.png        generated map as produced
    pred/<id>.png       label map (ids 0-18) after snapping to the palette
    log.jsonl           per image: seconds, off-palette pixel share
    metrics.json        mIoU (face classes, +background, L/R merged), F1, per class, confusion matrix,
                        plus mean off-palette share and seconds per image

    python -m inference.benchmark.eval_celebamask_img2palette \\
        --adapter results/celebamask_lora_i2p_s2i/final --output_dir results/celebamask_eval/final/img2palette --limit 500
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from inference.benchmark.celebamask_eval_common import (
    add_model_args,
    load_model,
    load_photo,
    overrides,
    segment_to_palette,
    split_ids,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    for sub in ("raw", "pred"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)
    ids = split_ids(args)
    todo = [i for i in ids if not os.path.exists(os.path.join(args.output_dir, "pred", f"{i}.png"))]
    print(f"[img2palette] split={args.split} images={len(ids)} remaining={len(todo)} adapter={args.adapter}", flush=True)

    if todo:
        model, info = load_model(args)
        params = overrides(args)
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump({**vars(args), **info, "inference_overrides": params}, f, indent=2, default=str)
        start = time.time()
        for n, image_id in enumerate(todo, 1):
            t0 = time.time()
            raw, label, off = segment_to_palette(model, load_photo(args.dataset_root, image_id), args.seed + image_id, params)
            raw.save(os.path.join(args.output_dir, "raw", f"{image_id}.png"))
            Image.fromarray(label).save(os.path.join(args.output_dir, "pred", f"{image_id}.png"))
            with open(os.path.join(args.output_dir, "log.jsonl"), "a") as f:
                f.write(json.dumps({"id": image_id, "seconds": round(time.time() - t0, 2), "off_palette": off}) + "\n")
            eta = (time.time() - start) / n * (len(todo) - n) / 60
            print(f"[img2palette] {n}/{len(todo)} id={image_id} {time.time() - t0:.1f}s off_palette={off:.3f} "
                  f"eta={eta:.0f}min", flush=True)

    # Full per-class report with the repository's CelebAMask-HQ evaluator, then add the format-adherence stats.
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    subprocess.run([sys.executable, "tools/evaluation/segment/evaluate_celebamask.py", "--dataset_root",
                    args.dataset_root, "--pred_dir", os.path.join(args.output_dir, "pred"),
                    "--output_json", metrics_path], check=True)
    logs = [json.loads(l) for l in open(os.path.join(args.output_dir, "log.jsonl"))]
    metrics = json.load(open(metrics_path))
    metrics["mean_off_palette_share"] = float(np.mean([l["off_palette"] for l in logs]))
    metrics["seconds_per_image"] = float(np.mean([l["seconds"] for l in logs]))
    json.dump(metrics, open(metrics_path, "w"), indent=2)
    print(f"[img2palette] off-palette share {metrics['mean_off_palette_share']:.4f}, "
          f"{metrics['seconds_per_image']:.1f}s/image -> {metrics_path}")


if __name__ == "__main__":
    main()
