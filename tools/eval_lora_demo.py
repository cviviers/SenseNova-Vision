# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Score a model directory on the held-out "glorp" split (see make_lora_demo_data.py).

Reports box F1@0.5 / mean best IoU for bbox detection and mask IoU for binary
segmentation, so the base model and a LoRA-merged model can be compared:

    python tools/eval_lora_demo.py --model_path /models/SenseNova-Vision-7B-MoT --output_dir results/lora_demo/eval_base
    python tools/eval_lora_demo.py --model_path results/lora_demo/merged       --output_dir results/lora_demo/eval_lora
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from inference.sensenova_vision import SenseNovaVisionModel
from tools.make_lora_demo_data import BBOX_PROMPT, GLORP, SEG_PROMPT
from utils.parsing_output import parse_detection_output


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(preds, gts, thr=0.5):
    used, tp, best = set(), 0, []
    for gt in gts:
        ious = [(box_iou(gt, p), j) for j, p in enumerate(preds) if j not in used]
        iou, j = max(ious, default=(0.0, None))
        best.append(iou)
        if iou >= thr:
            used.add(j)
            tp += 1
    return tp, best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default="datas/lora_demo/test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", default=os.environ.get("SENSENOVA_DTYPE", "bf16"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "annotations.jsonl")) as f:
        samples = [json.loads(line) for line in f][: args.limit]
    os.makedirs(args.output_dir, exist_ok=True)
    model = SenseNovaVisionModel(model_path=args.model_path, dtype=args.dtype)

    tp = n_pred = n_gt = 0
    best_ious, mask_ious, records = [], [], []
    for sample in samples:
        image_path = os.path.join(args.data_dir, sample["image"])
        image = Image.open(image_path).convert("RGB")
        stem = os.path.splitext(os.path.basename(image_path))[0]

        text = model.generate(
            question="<image>" + BBOX_PROMPT.format(name=GLORP), images=[image_path], mode="dense_detection"
        )
        preds = [p["bbox"] for p in parse_detection_output(text, image.size) if p.get("kind") == "bbox"]
        hits, best = match_boxes(preds, sample["boxes"])
        tp, n_pred, n_gt = tp + hits, n_pred + len(preds), n_gt + len(sample["boxes"])
        best_ious.extend(best)

        mask_pred = model.generate(
            question="<image>" + SEG_PROMPT.format(name=GLORP), images=[image_path], mode="dense_perception"
        )
        mask_pred.save(os.path.join(args.output_dir, f"{stem}_mask.png"))
        pred = np.asarray(mask_pred.convert("L").resize(image.size, Image.NEAREST)) > 127
        gt = np.asarray(Image.open(os.path.join(args.data_dir, sample["mask"])).convert("L")) > 127
        union = np.logical_or(pred, gt).sum()
        mask_iou = float(np.logical_and(pred, gt).sum() / union) if union else 1.0
        mask_ious.append(mask_iou)

        records.append({"image": sample["image"], "text": text, "num_pred": len(preds), "box_hits": hits,
                        "num_gt": len(sample["boxes"]), "mask_iou": mask_iou})
        print(f"{stem}: boxes {hits}/{len(sample['boxes'])} (pred {len(preds)}) mask IoU {mask_iou:.3f}", flush=True)

    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    summary = {
        "model_path": args.model_path,
        "num_images": len(samples),
        "box_precision@0.5": precision,
        "box_recall@0.5": recall,
        "box_f1@0.5": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "box_mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "mask_miou": float(np.mean(mask_ious)) if mask_ious else 0.0,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "samples": records}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
