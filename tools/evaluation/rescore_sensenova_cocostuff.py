"""Re-score SenseNova-Vision COCO-Stuff segmentations with the void fill, or score its CelebAMask-HQ ones.

SenseNova-Vision's img2palette eval (inference/benchmark/seg_eval_common.py, symmflow branch) snaps every
generated pixel to the palette *including black*, which decodes to class 0 (unlabeled). Ground-truth
unlabeled pixels are ignored, but a black *prediction* on a labelled pixel only costs a miss for the true
class and never a false positive for any class: a free abstention that a forced-choice model does not get.

COCO-Stuff (--dataset cocostuff): the saved label maps are scored two ways, one confusion matrix pooled
over all images:

  as evaluated   the predictions unchanged: reproduces SenseNova's metrics.json (black = miss only)
  void filled    every black pixel takes the class of the nearest non-black predicted pixel, so every
                 pixel is a real class, as in the standard forced-choice protocol (all-black maps stay 0)

The fill runs at the prediction's own resolution, before any resize. mIoU averages the 171 used
classes; unlabeled ground truth is ignored. The black-prediction rate (labelled pixels predicted black)
is reported as well; it is the size of the effect.

CelebAMask-HQ (--dataset celeba): black is background, a real class, so there is nothing to fill. The maps
are scored two ways, mIoU over the 18 face classes:

  background scored    background is a class in the matrix, so a face class predicted on background is
                       a false positive: reproduces SenseNova's metrics.json
  background ignored   ground-truth background pixels are dropped, as 2Symm2Flow's evaluate_segmentation
                       does: a face class predicted on background costs nothing

Here "black" is the share of face pixels predicted background.

--thin_radius r [r ...] also scores every variant after removing thin regions (utils/thin_regions.py's
rule, as in SenseNova's tools/rescore_thin_regions.py): pixels of a class in no (2r+1) x (2r+1) square of
that class are refilled from the nearest kept pixel, after the void fill and before any resize. It sizes
the effect of thin regions; it is not a replacement metric.

Predictions:  <key>.png class-id maps from eval_seg_img2palette (pred/) or <key>_s<seed>.png round-trip
              maps from eval_seg_seg2img (reseg/). COCO-Stuff: ids 0 = unlabeled, 1-182 = COCO-Stuff ids.
              CelebA: ids 0-18 in CelebAMask-HQ order, 0 = background.
Ground truth: COCO-Stuff: stuffthingmaps_trainval2017 (folder or zip), value v = id - 1, 255 = unlabeled.
              CelebA: the Hugging Face eurecom-ds/celeba_hq_mask split 2Symm2Flow evaluates on (RGB masks in
              the CelebAMask-HQ colours). It has no image ids: its rows are the split's images in ascending
              CelebA-HQ id, so the folder must hold the whole split and row i is the i-th key in numeric order.

Run from src/symm2flow:
    uv run python rescore_sensenova_cocostuff.py --pred /path/to/cocostuff_eval/final/img2palette/pred
    uv run python rescore_sensenova_cocostuff.py --pred .../seg2img/reseg --photo_log .../seg2img/log.jsonl \\
        --output_json rescore.json
    uv run python rescore_sensenova_cocostuff.py --dataset celeba --pred .../celebamask/lora-v4_legend/pred
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

from config import data_raw_dir

UNUSED_IDS = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91)
# CelebAMask-HQ class colours, indexed by class id (as in data/dataloaders.py's CelebHQMaskedDataset).
CELEBA_COLORS = np.array([
    [0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
    [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
    [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0],
])
# Per dataset: ids in the confusion matrix, ids in the mean (171 used COCO-Stuff classes, 18 face classes).
DATASETS = {
    "cocostuff": (183, [i for i in range(1, 183) if i not in UNUSED_IDS]),
    "celeba": (19, list(range(1, 19))),
}


# ----------------------------------------------------------------------------- correction
def fill_void(pred):
    """Give every 0 (black) pixel the class of the nearest non-black predicted pixel; (map, filled pixels)."""
    void = pred == 0
    if not void.any() or void.all():
        return pred, 0  # all-black map (e.g. a failed seg2img photo): nothing to propagate, stays a miss
    y, x = ndi.distance_transform_edt(void, return_distances=False, return_indices=True)
    return pred[y, x], int(void.sum())


def remove_thin(pred, radius):
    """Refill every pixel that lies in no (2r+1)-square of its own class from the nearest kept pixel.

    A window holds one class exactly when its max equals its min; the union of those windows is the
    union of the per-class openings (see utils/thin_regions.py). Windows must lie inside the map.
    """
    size, (h, w) = 2 * radius + 1, pred.shape
    uniform = np.zeros(pred.shape, bool)
    inner = ndi.maximum_filter(pred, size) == ndi.minimum_filter(pred, size)
    uniform[radius:h - radius, radius:w - radius] = inner[radius:h - radius, radius:w - radius]
    keep = ndi.binary_dilation(uniform, np.ones((size, size), bool))
    if keep.all() or not keep.any():
        return pred
    y, x = ndi.distance_transform_edt(~keep, return_distances=False, return_indices=True)
    return pred[y, x]


# ----------------------------------------------------------------------------- data
class GroundTruth:
    """COCO-Stuff label maps as class ids (0 = unlabeled); each worker opens its own zip handle."""

    def __init__(self, labels, split):
        self.labels, self.split, self._zip = labels, split, None

    def __call__(self, key):
        if self.labels.endswith(".zip"):
            if self._zip is None:
                self._zip = zipfile.ZipFile(self.labels)
            raw = Image.open(io.BytesIO(self._zip.read(f"{self.split}/{key}.png")))
        else:
            raw = Image.open(os.path.join(self.labels, self.split, f"{key}.png"))
        v = np.asarray(raw, dtype=np.uint8)
        return np.where(v == 255, 0, v.astype(np.uint16) + 1).astype(np.uint8)


class CelebAGroundTruth:
    """CelebAMask-HQ RGB masks from the Hugging Face split as class ids; colours off the palette become 0."""

    def __init__(self, labels, split, keys):
        from datasets import load_dataset

        self.dataset = load_dataset(labels, split=split)
        if len(self.dataset) != len(keys):
            raise SystemExit(f"{len(keys)} predictions but {len(self.dataset)} rows in {labels} ({split}): "
                             "the rows are matched by position, so the folder must hold the whole split")
        self.row = {key: i for i, key in enumerate(sorted(keys, key=int))}
        self.codes = CELEBA_COLORS[:, 0] << 16 | CELEBA_COLORS[:, 1] << 8 | CELEBA_COLORS[:, 2]

    def __call__(self, key):
        rgb = np.asarray(self.dataset[self.row[key]]["mask"].convert("RGB")).astype(np.int64)
        codes = rgb[..., 0] << 16 | rgb[..., 1] << 8 | rgb[..., 2]
        label = np.zeros(codes.shape, np.uint8)
        for class_id, code in enumerate(self.codes):
            label[codes == code] = class_id
        return label


def resize_nearest(label, shape):
    if label.shape == shape:
        return label
    return np.asarray(Image.fromarray(label).resize(shape[::-1], Image.NEAREST))


def confusion(gt, pred, num_ids, ignore_zero=True):
    gt, pred = gt.astype(np.int64).ravel(), pred.astype(np.int64).ravel()
    if ignore_zero:  # unlabeled (COCO-Stuff) or background (CelebA) ground truth is ignored
        keep = gt != 0
        gt, pred = gt[keep], pred[keep]
    return np.bincount(gt * num_ids + pred, minlength=num_ids * num_ids).reshape(num_ids, num_ids)


def summarize(conf, used_ids, average):
    tp = np.diag(conf).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (conf.sum(0) + conf.sum(1) - tp)
    iou = iou[used_ids]
    if average == "gt":  # only classes present in the ground truth
        iou = np.where(conf.sum(1)[used_ids] > 0, iou, np.nan)
    labelled = conf[1:].sum()  # ground-truth pixels of a real (non-0) class
    return {
        "mIoU": float(np.nanmean(iou)),
        "classes_scored": int(np.sum(~np.isnan(iou))),
        "pixel_accuracy": float(tp.sum() / conf.sum()) if conf.sum() else float("nan"),
        "black_prediction_rate": float(conf[1:, 0].sum() / labelled) if labelled else float("nan"),
    }


# ----------------------------------------------------------------------------- workers
_worker = {}


def init_worker(args, failed, keys):
    if args.dataset == "celeba":
        gt = CelebAGroundTruth(args.labels, args.split, keys)
    else:
        gt = GroundTruth(args.labels, args.split)
    _worker.update(args=args, failed=failed, gt=gt)


def process(name):
    """Confusion matrices of every variant for one prediction file."""
    args, failed = _worker["args"], _worker["failed"]
    num_ids = DATASETS[args.dataset][0]
    key = name.split("_s")[0]  # reseg/<key>_s<seed>.png -> <key>
    pred = np.asarray(Image.open(os.path.join(args.pred, f"{name}.png")))
    if pred.ndim != 2:
        raise ValueError(f"{name}.png is not a single-channel class-id map (got shape {pred.shape})")
    if name in failed:
        pred = np.zeros_like(pred)  # failed photo check: scored as unlabeled, as eval_seg_seg2img does
    gt = _worker["gt"](key)

    if args.dataset == "celeba":
        # (prediction, ignore background ground truth)
        variants = {"background scored": (pred, False), "background ignored": (pred, True)}
    else:
        variants = {"as evaluated": (pred, True), "void filled": (fill_void(pred)[0], True)}
    for r in args.thin_radius or ():
        variants.update({f"{v} | thin r={r}": (remove_thin(p, r), z) for v, (p, z) in list(variants.items())
                         if "| thin" not in v})

    out = {}
    for variant, (p, ignore_zero) in variants.items():
        if args.resize == "gt_to_pred":
            g, q = resize_nearest(gt, p.shape), p
        else:
            g, q = gt, resize_nearest(p, gt.shape)
        out[variant] = (confusion(g, q, num_ids, ignore_zero), int((p != pred).sum()), p.size)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", required=True, help="Folder of SenseNova class-id maps (pred/ or reseg/).")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="cocostuff")
    parser.add_argument("--labels", default=None,
                        help="cocostuff: stuffthingmaps_trainval2017 folder or zip (default: the repo's data/raw); "
                             "celeba: Hugging Face dataset (default: eurecom-ds/celeba_hq_mask).")
    parser.add_argument("--split", default=None, help="Default: val2017 (cocostuff), validation (celeba).")
    parser.add_argument("--photo_log", default=None,
                        help="seg2img log.jsonl: outputs with is_photo=false are scored as all unlabeled.")
    parser.add_argument("--resize", choices=["gt_to_pred", "pred_to_gt"], default="gt_to_pred",
                        help="gt_to_pred (default) matches SenseNova's protocol (GT nearest-resized to the "
                             "training size); pred_to_gt scores at the original GT resolution (standard).")
    parser.add_argument("--average", choices=["union", "gt"], default="union",
                        help="union (default, mmseg and SenseNova): classes in GT or prediction; "
                             "gt: only classes present in the ground truth (2Symm2Flow's _miou_dice).")
    parser.add_argument("--thin_radius", type=int, nargs="+", default=None,
                        help="Also score every variant with thin regions removed, one row per radius >= 1 "
                             "(e.g. --thin_radius 1 2). Diagnostic only.")
    parser.add_argument("--limit", type=int, default=None, help="First N prediction files.")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    if args.dataset == "celeba":
        args.labels = args.labels or "eurecom-ds/celeba_hq_mask"
        args.split = args.split or "validation"
    else:
        args.labels = args.labels or os.path.join(data_raw_dir, "cocostuff", "stuffthingmaps_trainval2017")
        args.split = args.split or "val2017"

    all_names = sorted(f[:-4] for f in os.listdir(args.pred) if f.endswith(".png"))
    names = all_names[: args.limit]
    if not names:
        raise SystemExit(f"no .png predictions in {args.pred}")
    keys = sorted({name.split("_s")[0] for name in all_names})  # CelebA rows are matched against all keys
    failed = set()
    if args.photo_log:
        for line in open(args.photo_log):
            rec = json.loads(line)
            if not rec.get("is_photo", True):
                failed.add(f"{rec['key']}_s{rec['seed']}")
    print(f"[rescore] {len(names)} predictions from {args.pred} | {args.dataset} GT {args.labels} ({args.split}) | "
          f"resize {args.resize} | average {args.average}" + (f" | {len(failed)} failed photos" if failed else ""))

    num_ids, used_ids = DATASETS[args.dataset]
    totals = {}
    with Pool(args.workers, initializer=init_worker, initargs=(args, failed, keys)) as pool:
        for out in pool.imap_unordered(process, names, chunksize=8):
            for variant, (conf, changed, pixels) in out.items():
                t = totals.setdefault(variant, [np.zeros((num_ids, num_ids), np.int64), 0, 0])
                t[0] += conf
                t[1] += changed
                t[2] += pixels

    results = {}
    print(f"\n{'variant':32s} {'mIoU':>7s} {'classes':>7s} {'pix acc':>7s} {'black':>7s} {'changed':>8s}")
    for variant, (conf, changed, pixels) in totals.items():
        s = summarize(conf, used_ids, args.average)
        s["changed_pixel_share"] = changed / pixels
        results[variant] = s
        print(f"{variant:32s} {s['mIoU']:7.4f} {s['classes_scored']:7d} {s['pixel_accuracy']:7.4f} "
              f"{s['black_prediction_rate']:7.4f} {s['changed_pixel_share']:8.4f}")
    if args.dataset == "celeba":
        print("\nblack = share of face pixels predicted background; changed = share of predicted pixels "
              "relabelled by the thin-region clean-up.")
    else:
        print("\nblack = share of labelled pixels predicted unlabeled (black); changed = share of predicted pixels "
              "relabelled by the fill and the thin-region clean-up.")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"pred": args.pred, "dataset": args.dataset, "num_predictions": len(names),
                       "split": args.split, "resize": args.resize, "average": args.average,
                       "thin_radius": args.thin_radius, "results": results},
                      f, indent=2)
        print(f"[rescore] wrote {args.output_json}")


if __name__ == "__main__":
    main()
