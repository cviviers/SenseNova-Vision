# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""IoU evaluation for CelebAMask-HQ face parsing predictions.

Ground-truth label maps are rebuilt from the per-part annotation PNGs the same
way CelebAMask-HQ's g_mask.py does it (class ids 1..18 painted in order, later
classes overwrite earlier ones, 0 = background). Predictions are uint8 label
PNGs with the same ids, as written by inference/benchmark/batch_celebamask_segment.py.

Reports, from a dataset-level confusion matrix:
  * per-class IoU and F1 (F1 is the usual face-parsing metric),
  * mIoU over the 18 face classes, and with background (19 classes),
  * mIoU with left/right classes merged (eye, brow, ear), which separates
    side confusions from segmentation quality.

Classes that were not prompted (``--classes`` at inference time) are excluded
from the means. Classes absent from both GT and prediction are reported as NaN.

    python tools/evaluation/segment/evaluate_celebamask.py --pred_dir results/celebamask_hq/val_binary/pred
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

CLASS_NAMES = ["background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear",
               "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth"]
LR_MERGE = {"l_eye": "eye", "r_eye": "eye", "l_brow": "brow", "r_brow": "brow", "l_ear": "ear", "r_ear": "ear"}


def gt_label_map(dataset_root, image_id):
    anno_dir = os.path.join(dataset_root, "CelebAMask-HQ-mask-anno", str(image_id // 2000))
    label = np.zeros((512, 512), dtype=np.uint8)
    for class_id, name in enumerate(CLASS_NAMES[1:], start=1):
        path = os.path.join(anno_dir, f"{image_id:05d}_{name}.png")
        if os.path.exists(path):
            label[np.asarray(Image.open(path).convert("L")) > 127] = class_id
    return label


def scores(conf):
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return iou, f1


def merge_lr(conf):
    names = [LR_MERGE.get(n, n) for n in CLASS_NAMES]
    merged_names = list(dict.fromkeys(names))
    index = np.array([merged_names.index(n) for n in names])
    merged = np.zeros((len(merged_names), len(merged_names)), dtype=np.int64)
    np.add.at(merged, (index[:, None], index[None, :]), conf)
    return merged, merged_names


def mean(values, keep):
    vals = [v for v, k in zip(values, keep) if k and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred_dir", required=True, help="Directory of <id>.png label maps.")
    parser.add_argument("--dataset_root", default="/datasets/CelebAMask-HQ/CelebAMask-HQ")
    parser.add_argument("--output_json", default=None, help="Defaults to <pred_dir>/../metrics.json.")
    parser.add_argument("--save_gt_dir", default=None, help="Optionally write the rebuilt GT label maps here.")
    args = parser.parse_args()

    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, "*.png")), key=lambda p: int(os.path.basename(p)[:-4]))
    if not pred_paths:
        raise SystemExit(f"No predictions in {args.pred_dir}")

    prompted = set(range(len(CLASS_NAMES)))
    run_dir = os.path.dirname(os.path.normpath(args.pred_dir))
    configs = glob.glob(os.path.join(run_dir, "config_shard*.json"))
    if configs:
        with open(configs[0]) as f:
            prompted = {0} | {int(k) for k in json.load(f)["prompted_classes"]}

    n = len(CLASS_NAMES)
    conf = np.zeros((n, n), dtype=np.int64)
    if args.save_gt_dir:
        os.makedirs(args.save_gt_dir, exist_ok=True)
    for path in pred_paths:
        image_id = int(os.path.basename(path)[:-4])
        pred = np.asarray(Image.open(path))
        gt = gt_label_map(args.dataset_root, image_id)
        if args.save_gt_dir:
            Image.fromarray(gt).save(os.path.join(args.save_gt_dir, f"{image_id}.png"))
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred).resize(gt.shape[::-1], Image.NEAREST))
        conf += np.bincount(gt.astype(np.int64).ravel() * n + pred.ravel(), minlength=n * n).reshape(n, n)

    iou, f1 = scores(conf)
    keep = [i in prompted for i in range(n)]
    face_keep = [k and i > 0 for i, k in enumerate(keep)]
    merged_conf, merged_names = merge_lr(conf)
    merged_iou, _ = scores(merged_conf)
    merged_keep = [name != "background" and any(
        k for orig, k in zip(CLASS_NAMES, keep) if LR_MERGE.get(orig, orig) == name) for name in merged_names]

    result = {
        "pred_dir": args.pred_dir,
        "num_images": len(pred_paths),
        "mIoU_face_classes": mean(iou, face_keep),
        "mIoU_with_background": mean(iou, keep),
        "mean_F1_face_classes": mean(f1, face_keep),
        "pixel_accuracy": float(np.diag(conf).sum() / conf.sum()),
        "mIoU_face_classes_lr_merged": mean(merged_iou, merged_keep),
        "per_class": {
            name: {"IoU": float(iou[i]), "F1": float(f1[i]), "gt_pixels": int(conf[i].sum()),
                   "prompted": keep[i]}
            for i, name in enumerate(CLASS_NAMES)
        },
        "per_class_lr_merged_IoU": {name: float(merged_iou[i]) for i, name in enumerate(merged_names)},
        "confusion_matrix": conf.tolist(),
        "class_names": CLASS_NAMES,
    }
    output_json = args.output_json or os.path.join(run_dir, "metrics.json")
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"images: {len(pred_paths)}")
    print(f"{'class':12s} {'IoU':>7s} {'F1':>7s}")
    for i, name in enumerate(CLASS_NAMES):
        mark = "" if keep[i] else "  (not prompted)"
        print(f"{name:12s} {iou[i]:7.4f} {f1[i]:7.4f}{mark}")
    for key in ("mIoU_face_classes", "mIoU_with_background", "mean_F1_face_classes",
                "mIoU_face_classes_lr_merged", "pixel_accuracy"):
        print(f"{key}: {result[key]:.4f}")
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
