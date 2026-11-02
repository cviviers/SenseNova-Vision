# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Measure how much of a segmentation score comes from thin regions, by removing them and scoring again.

Palette-format outputs are decoded by snapping every pixel to the nearest palette colour. Soft edges between two
large regions then become thin bands of whatever class has a colour in between (on CelebAMask-HQ, l_lip's navy
between background and hair, l_ear's brown between background and skin). This script applies one rule to every run:

  1. Thin pixels: for each class c, the pixels of c that lie in no (2r+1) x (2r+1) square made only of c, i.e.
     mask minus its binary opening with that square. Every class is tested on the original map, so the order does
     not matter; --never_remove classes are kept.
  2. Refill: every removed pixel takes the label of the nearest kept pixel (Euclidean distance transform).
  3. Score: one confusion matrix pooled over all images, per-class IoU = TP / (TP + FP + FN), mIoU = unweighted
     mean over the evaluated classes, skipping classes absent from both GT and prediction.

Radius 0 is the unchanged prediction, so the first row of every run should reproduce its official score. The
clean-up runs at the prediction's own resolution, before any resize. It also removes genuinely thin true
structures (necklaces, glasses frames, poles), so use it to size the effect, not as a replacement metric.

Predictions are <key>.png files: single-channel label maps (L, P or I;16 PNGs, value = class id), or RGB palette
images together with --palette. Ground truth is one of:

  --gt_dir DIR             <key>.png label maps with the same ids (generic; set --num_classes and friends)
  --celebamask_root ROOT   rebuilt from CelebAMask-HQ's part PNGs; 19 ids, mIoU over the 18 face classes
  --cocostuff PATH         stuffthingmaps_trainval2017.zip or its extracted folder (<split>/<key>.png);
                           255 -> 0 (unlabeled, ignored), other ids + 1; mIoU over the 171 used classes

The presets match tools/evaluation/segment/evaluate_celebamask.py and inference/benchmark/seg_eval_common.py.
Without ground truth the script only writes the cleaned maps (--save_dir).

    # Reproduce the study's CelebA comparison (r0 = official scores)
    python tools/evaluation/segment/rescore_thin_regions.py \\
        --pred "LoRA v3=results/spike/lora-celeba-i2p/img2palette/pred" \\
        --pred "LoRA v4=results/spike/lora-celeba-v4-legend/img2legend/pred" \\
        --celebamask_root /mnt/big_data/datasets/CelebAMask-HQ/CelebAMask-HQ --radius 0 1 2 --output_json rescore.json

    # Any other label maps
    python tools/evaluation/segment/rescore_thin_regions.py --pred my_model/pred --gt_dir my_data/labels \\
        --num_classes 21 --ignore_index 255 --never_remove 0

    # Only clean, no scoring
    python tools/evaluation/segment/rescore_thin_regions.py --pred my_model/pred --save_dir cleaned --radius 1
"""

import argparse
import io
import json
import os
import zipfile
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CELEBAMASK_NAMES = ["background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear",
                    "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth"]
COCOSTUFF_UNUSED_IDS = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91)


def remove_thin(label, radius, never_remove=()):
    """Return (cleaned label map, number of relabelled pixels) for one map; radius 0 returns it unchanged."""
    if radius == 0:
        return label, 0
    square = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    keep = np.ones(label.shape, dtype=bool)
    for c in np.unique(label):
        if c in never_remove:
            continue
        mask = label == c
        keep[mask & ~ndi.binary_opening(mask, square)] = False
    if keep.all() or not keep.any():
        return label, 0
    index = ndi.distance_transform_edt(~keep, return_distances=False, return_indices=True)
    cleaned = label[tuple(index)]
    return cleaned, int((cleaned != label).sum())


def decode_palette(rgb, colours, ids):
    """Snap every pixel to the nearest palette colour (RGB Euclidean), as seg_eval_common.palette_to_label does."""
    label = np.empty(rgb.shape[:2], dtype=ids.dtype)
    for y in range(0, rgb.shape[0], 64):  # in row blocks: H x W x classes distances get large for 180+ classes
        block = rgb[y:y + 64].astype(np.int32)
        dist = ((block[:, :, None, :] - colours[None, None]) ** 2).sum(-1)
        label[y:y + 64] = ids[dist.argmin(-1)]
    return label


def load_palette(path):
    """JSON list indexed by class id, each entry [r, g, b] or null for ids that are never predicted."""
    with open(path) as f:
        entries = json.load(f)
    ids = np.array([i for i, c in enumerate(entries) if c is not None])
    colours = np.array([entries[i] for i in ids], dtype=np.int32)
    return colours, ids.astype(np.uint16 if ids.max() > 255 else np.uint8)


def load_pred(path, palette):
    image = Image.open(path)
    if image.mode in ("L", "P", "I", "I;16"):
        return np.asarray(image)
    if palette is None:
        raise ValueError(f"{path} is a {image.mode} image; pass --palette to decode colour maps")
    return decode_palette(np.asarray(image.convert("RGB")), *palette)


def resize_nearest(label, shape):
    if label.shape == shape:
        return label
    return np.asarray(Image.fromarray(label).resize(shape[::-1], Image.NEAREST))


class GroundTruth:
    """Loads the GT label map for a key. Workers each open their own zip handle."""

    def __init__(self, args):
        self.gt_dir, self.celebamask_root = args.gt_dir, args.celebamask_root
        self.cocostuff, self.split = args.cocostuff, args.cocostuff_split
        self._zip = None

    def __call__(self, key):
        if self.gt_dir:
            return np.asarray(Image.open(os.path.join(self.gt_dir, f"{key}.png")))
        if self.celebamask_root:
            image_id = int(key)
            anno_dir = os.path.join(self.celebamask_root, "CelebAMask-HQ-mask-anno", str(image_id // 2000))
            label = np.zeros((512, 512), dtype=np.uint8)
            for class_id, name in enumerate(CELEBAMASK_NAMES[1:], start=1):
                path = os.path.join(anno_dir, f"{image_id:05d}_{name}.png")
                if os.path.exists(path):
                    label[np.asarray(Image.open(path).convert("L")) > 127] = class_id
            return label
        if os.path.isdir(self.cocostuff):
            raw = np.asarray(Image.open(os.path.join(self.cocostuff, self.split, f"{key}.png")))
        else:
            if self._zip is None:
                self._zip = zipfile.ZipFile(self.cocostuff)
            raw = np.asarray(Image.open(io.BytesIO(self._zip.read(f"{self.split}/{key}.png"))))
        return np.where(raw == 255, 0, raw.astype(np.int64) + 1).astype(np.uint8)


_worker = {}


def init_worker(args, runs, palette):
    _worker.update(args=args, runs=runs, palette=palette, gt=GroundTruth(args) if args.has_gt else None)


def process(key):
    """Confusion matrix and relabelled pixel count for every (run, radius) on one image."""
    args, runs, palette, gt_loader = _worker["args"], _worker["runs"], _worker["palette"], _worker["gt"]
    n = args.num_classes
    gt = gt_loader(key) if gt_loader else None
    out = {}
    for name, pred_dir in runs:
        pred = load_pred(os.path.join(pred_dir, f"{key}.png"), palette)
        for radius in args.radius:
            cleaned, changed = remove_thin(pred, radius, args.never_remove)
            if args.save_dir and radius > 0:
                path = os.path.join(args.save_dir, safe_name(name), f"r{radius}", f"{key}.png")
                Image.fromarray(cleaned).save(path)
            if gt is None:
                out[name, radius] = (None, changed, cleaned.size)
                continue
            if args.resize == "gt_to_pred":
                g, p = resize_nearest(gt, cleaned.shape), cleaned
            else:
                g, p = gt, resize_nearest(cleaned, gt.shape)
            g, p = g.astype(np.int64).ravel(), p.astype(np.int64).ravel()
            valid = ~np.isin(g, args.ignore_index)
            if p.max(initial=0) >= n or g[valid].max(initial=0) >= n:
                raise ValueError(f"{key}: class id >= --num_classes {n}")
            conf = np.bincount(g[valid] * n + p[valid], minlength=n * n).reshape(n, n)
            out[name, radius] = (conf, changed, cleaned.size)
    return out


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def class_iou(conf):
    tp = np.diag(conf).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (conf.sum(0) + conf.sum(1) - tp)


def parse_ids(values):
    """'1-18' and '3' style tokens to a sorted list of ints."""
    ids = set()
    for value in values:
        for token in str(value).split(","):
            lo, _, hi = token.partition("-")
            ids.update(range(int(lo), int(hi or lo) + 1))
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", action="append", required=True, metavar="[NAME=]DIR",
                        help="Directory of <key>.png predictions; repeat to compare runs on the same keys. "
                             "NAME defaults to the folder name.")
    gt = parser.add_mutually_exclusive_group()
    gt.add_argument("--gt_dir", help="Directory of <key>.png ground-truth label maps.")
    gt.add_argument("--celebamask_root", help="CelebAMask-HQ root (holds CelebAMask-HQ-mask-anno/).")
    gt.add_argument("--cocostuff", help="stuffthingmaps_trainval2017.zip or its extracted folder.")
    parser.add_argument("--cocostuff_split", default="val2017")
    parser.add_argument("--num_classes", type=int, help="Label ids are 0..num_classes-1 (preset: 19 / 183).")
    parser.add_argument("--eval_classes", nargs="+",
                        help="Ids averaged into mIoU, e.g. 1-18 (default: all ids not in --ignore_index; "
                             "preset: the 18 face classes / the 171 used COCO-Stuff classes).")
    parser.add_argument("--ignore_index", nargs="+", default=None,
                        help="GT ids whose pixels are not scored, e.g. 255 (preset COCO-Stuff: 0).")
    parser.add_argument("--never_remove", nargs="+", default=None,
                        help="Ids the clean-up never removes, e.g. a background class (preset: 0).")
    parser.add_argument("--radius", type=int, nargs="+", default=[0, 1, 2],
                        help="Square half-widths to test; 1 = 3x3 square, 0 = unchanged prediction.")
    parser.add_argument("--resize", choices=["pred_to_gt", "gt_to_pred"],
                        help="Size match when shapes differ (nearest neighbour). Default pred_to_gt; the "
                             "COCO-Stuff preset uses gt_to_pred, as the study's evaluator does.")
    parser.add_argument("--palette", help="JSON list indexed by class id of [r, g, b] (or null): decode RGB "
                                          "predictions by nearest colour.")
    parser.add_argument("--class_names", help="JSON list of names indexed by class id, for the report.")
    parser.add_argument("--save_dir", help="Write cleaned maps to <save_dir>/<run>/r<radius>/<key>.png.")
    parser.add_argument("--output_json")
    parser.add_argument("--top_classes", type=int, default=5,
                        help="Print the classes whose IoU changes most at the first radius above 0.")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()

    args.has_gt = bool(args.gt_dir or args.celebamask_root or args.cocostuff)
    args.radius = sorted(set(args.radius))
    names = None
    if args.celebamask_root:
        defaults = dict(num_classes=19, eval_classes=list(range(1, 19)), ignore_index=[], never_remove=[0],
                        resize="pred_to_gt")
        names = CELEBAMASK_NAMES
    elif args.cocostuff:
        defaults = dict(num_classes=183, eval_classes=[i for i in range(1, 183) if i not in COCOSTUFF_UNUSED_IDS],
                        ignore_index=[0], never_remove=[0], resize="gt_to_pred")
    else:
        defaults = dict(num_classes=None, eval_classes=None, ignore_index=[], never_remove=[], resize="pred_to_gt")
    for field in ("ignore_index", "never_remove", "eval_classes"):
        value = getattr(args, field)
        setattr(args, field, parse_ids(value) if value is not None else defaults[field])
    args.num_classes = args.num_classes or defaults["num_classes"]
    args.resize = args.resize or defaults["resize"]
    if args.has_gt and not args.num_classes:
        raise SystemExit("--gt_dir needs --num_classes")
    if args.has_gt and args.eval_classes is None:
        args.eval_classes = [i for i in range(args.num_classes) if i not in args.ignore_index]
    if not args.has_gt and not args.save_dir:
        raise SystemExit("Without ground truth there is nothing to score; pass --save_dir to write cleaned maps")
    if args.class_names:
        with open(args.class_names) as f:
            names = json.load(f)
    label = (lambda i: names[i]) if names else str
    palette = load_palette(args.palette) if args.palette else None

    runs = []
    for spec in args.pred:
        name, sep, pred_dir = spec.partition("=")
        runs.append((name, pred_dir) if sep else (os.path.basename(os.path.normpath(spec)), spec))
    if len({name for name, _ in runs}) < len(runs):
        raise SystemExit("Run names must be unique; name them with NAME=DIR")
    key_sets = [{f[:-4] for f in os.listdir(d) if f.endswith(".png")} for _, d in runs]
    keys = sorted(key_sets[0], key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k))
    if not keys:
        raise SystemExit(f"No .png predictions in {runs[0][1]}")
    for (name, pred_dir), key_set in zip(runs, key_sets):
        if key_set != key_sets[0]:
            raise SystemExit(f"{name} ({pred_dir}) has {len(key_set)} maps, {len(key_set ^ key_sets[0])} keys "
                             f"differ from {runs[0][0]}; runs must cover the same images")
    if args.save_dir:
        for name, _ in runs:
            for radius in args.radius:
                if radius > 0:
                    os.makedirs(os.path.join(args.save_dir, safe_name(name), f"r{radius}"), exist_ok=True)

    totals = {}
    with Pool(args.workers, initializer=init_worker, initargs=(args, runs, palette)) as pool:
        try:
            for i, result in enumerate(pool.imap(process, keys, chunksize=8)):
                for k, (conf, changed, pixels) in result.items():
                    t = totals.setdefault(k, [0, 0, 0])
                    t[0] = t[0] + conf if conf is not None else 0
                    t[1] += changed
                    t[2] += pixels
                if (i + 1) % 500 == 0:
                    print(f"{i + 1}/{len(keys)} images", flush=True)
        except ValueError as e:  # raised in a worker (SystemExit there would kill it and hang the pool)
            raise SystemExit(str(e))

    report = {"num_images": len(keys), "radius": args.radius, "never_remove": args.never_remove,
              "num_classes": args.num_classes, "eval_classes": args.eval_classes, "ignore_index": args.ignore_index,
              "resize": args.resize, "runs": {}}
    print(f"\n{len(keys)} images; relabelled = share of prediction pixels the clean-up changed")
    header = f"{'run':24s} {'radius':>6s} {'relabelled':>10s}"
    print(header + (f" {'mIoU':>7s} {'change':>7s} {'pix_acc':>7s}" if args.has_gt else ""))
    for name, pred_dir in runs:
        entry = report["runs"][name] = {"pred_dir": pred_dir}
        base_iou = None
        for radius in args.radius:
            conf, changed, pixels = totals[name, radius]
            row = {"relabelled_share": changed / pixels}
            line = f"{name[:24]:24s} {radius:6d} {changed / pixels:10.4f}"
            if args.has_gt:
                iou = class_iou(conf)
                miou = float(np.nanmean(iou[args.eval_classes]))
                base_iou = iou if radius == 0 else base_iou
                row.update(mIoU=miou, pixel_accuracy=float(np.diag(conf).sum() / conf.sum()),
                           per_class_IoU={label(i): float(iou[i]) for i in args.eval_classes})
                change = f"{miou - float(np.nanmean(base_iou[args.eval_classes])):+7.4f}" if base_iou is not None \
                    else f"{'':>7s}"
                line += f" {miou:7.4f} {change} {row['pixel_accuracy']:7.4f}"
            entry[f"r{radius}"] = row
            print(line)
        positive = [r for r in args.radius if r > 0]
        if args.has_gt and base_iou is not None and positive and args.top_classes:
            iou = class_iou(totals[name, positive[0]][0])
            delta = [(iou[i] - base_iou[i], i) for i in args.eval_classes if not np.isnan(iou[i] - base_iou[i])]
            top = sorted(delta, key=lambda d: -abs(d[0]))[:args.top_classes]
            print(f"{'':24s} largest IoU changes at r{positive[0]}: "
                  + ", ".join(f"{label(i)} {base_iou[i]:.3f}->{iou[i]:.3f}" for _, i in top))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.output_json}")
    if args.save_dir:
        print(f"cleaned maps in {args.save_dir}/<run>/r<radius>/")


if __name__ == "__main__":
    main()
