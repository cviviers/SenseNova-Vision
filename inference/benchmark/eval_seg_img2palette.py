# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the img2palette mode (photo -> palette segmentation map) on CelebAMask-HQ or COCO-Stuff.

For every image: generate with the training prompt (data/celebamask.py / data/cocostuff.py) in
dense-perception mode at the training image size, snap the output to the dataset palette, and score
it against the ground truth.

* celebamask  mIoU over the 18 face classes (background counted as a class in the confusion matrix);
              the full report of tools/evaluation/segment/evaluate_celebamask.py is added (F1, L/R merged).
* cocostuff   mIoU over the 171 COCO-Stuff classes, unlabeled ground-truth pixels ignored (standard protocol).

Outputs (resumable; existing pred/<key>.png are skipped):
    raw/<key>.png     generated map as produced
    pred/<key>.png    label map (class ids) after snapping to the palette
    log.jsonl         per image: seconds, off-palette pixel share
    metrics.json      mIoU, pixel accuracy, per-class IoU, confusion matrix, off-palette share, seconds/image

    python -m inference.benchmark.eval_seg_img2palette --dataset cocostuff \\
        --adapter results/cocostuff_lora_i2p_s2i/final --output_dir results/cocostuff_eval/final/img2palette --limit 500
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from inference.benchmark.seg_eval_common import (
    add_model_args,
    confusion,
    generation_params,
    get_dataset,
    load_model,
    segment_to_palette,
    summarize,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    ds = get_dataset(args)
    out = args.output_dir
    for sub in ("raw", "pred"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    keys = ds.keys(args.split, args.limit)
    todo = [k for k in keys if not os.path.exists(os.path.join(out, "pred", f"{k}.png"))]
    print(f"[img2palette] dataset={ds.name} split={args.split} images={len(keys)} remaining={len(todo)} "
          f"adapter={args.adapter}", flush=True)

    if todo:
        model, info = load_model(args)
        params, transforms = generation_params(args, ds)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump({**vars(args), **info, "inference_overrides": params}, f, indent=2, default=str)
        start = time.time()
        for n, key in enumerate(todo, 1):
            t0 = time.time()
            photo, _ = ds.load(key)
            raw, label, off = segment_to_palette(model, ds, photo, args.seed + int(key) % 1_000_000, params, transforms)
            raw.save(os.path.join(out, "raw", f"{key}.png"))
            Image.fromarray(label).save(os.path.join(out, "pred", f"{key}.png"))
            with open(os.path.join(out, "log.jsonl"), "a") as f:
                f.write(json.dumps({"key": key, "seconds": round(time.time() - t0, 2), "off_palette": off}) + "\n")
            eta = (time.time() - start) / n * (len(todo) - n) / 60
            print(f"[img2palette] {n}/{len(todo)} {key} {time.time() - t0:.1f}s off_palette={off:.3f} eta={eta:.0f}min",
                  flush=True)

    conf = np.zeros((ds.num_ids, ds.num_ids), np.int64)
    for key in keys:
        conf += confusion(ds, ds.load(key)[1], np.asarray(Image.open(os.path.join(out, "pred", f"{key}.png"))))
    logs = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]
    metrics = {"dataset": ds.name, "num_images": len(keys), **summarize(ds, conf),
               "mean_off_palette_share": float(np.mean([l["off_palette"] for l in logs])),
               "seconds_per_image": float(np.mean([l["seconds"] for l in logs])),
               "confusion_matrix": conf.tolist()}
    if ds.name == "celebamask":
        # Full CelebAMask-HQ report (F1, L/R merged, mIoU with background) from the repository's evaluator.
        report = os.path.join(out, "celebamask_report.json")
        subprocess.run([sys.executable, "tools/evaluation/segment/evaluate_celebamask.py", "--dataset_root", ds.root,
                        "--pred_dir", os.path.join(out, "pred"), "--output_json", report], check=True,
                       stdout=subprocess.DEVNULL)
        metrics["celebamask_report"] = {k: v for k, v in json.load(open(report)).items()
                                        if k not in ("confusion_matrix", "per_class", "class_names")}
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"[img2palette] {ds.name}: mIoU {metrics['mIoU']:.4f} over {metrics['classes_scored']} classes, "
          f"pixel acc {metrics['pixel_accuracy']:.4f}, off-palette {metrics['mean_off_palette_share']:.4f}, "
          f"{metrics['seconds_per_image']:.1f}s/image -> {out}/metrics.json")


if __name__ == "__main__":
    main()
