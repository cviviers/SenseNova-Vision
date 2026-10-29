# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the img2palette mode (photo -> palette segmentation map) on CelebAMask-HQ or COCO-Stuff.

For every image: generate with the training prompt (data/celebamask.py / data/cocostuff.py) in
dense-perception mode at the training image size, snap the output to the dataset palette, and score
it against the ground truth.

* celebamask  mIoU over the 18 face classes (background counted as a class in the confusion matrix);
              the full report of tools/evaluation/segment/evaluate_celebamask.py is added (F1, L/R merged).
* cocostuff   mIoU over the 171 COCO-Stuff classes, unlabeled ground-truth pixels ignored (standard protocol).

--format legend (cocostuff) asks instead in the released model's own generic-segmentation format
(data/cocostuff.py LEGEND_PROMPT, mode caption_generate): the model writes a colour legend
<p>class<color>(R, G, B)</color></p> and then a mask in those colours, decoded through the legend. This is the
zero-shot baseline in the model's native format, and the evaluation of LoRAs trained on img2palette_legend.
log.jsonl then records the legend text, its number of entries and the phrases that name no COCO-Stuff class.

Outputs (resumable; existing pred/<key>.png are skipped):
    raw/<key>.png     generated map as produced
    pred/<key>.png    label map (class ids) after snapping to the palette
    log.jsonl         per image: seconds, off-palette pixel share (log.shard<i>.jsonl when sharded)
    metrics.json      mIoU, pixel accuracy, per-class IoU, confusion matrix, off-palette share, seconds/image

    python -m inference.benchmark.eval_seg_img2palette --dataset cocostuff \\
        --adapter results/cocostuff_lora_i2p_s2i/final --output_dir results/cocostuff_eval/final/img2palette --limit 500

--num_shards/--shard_id --generate_only split the images over GPUs; a final run without them computes the metrics
from the files (scripts/eval_seg_tasks.sh does this for img2legend on all visible GPUs).
"""

import argparse
import glob
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
    segment_with_legend,
    summarize,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", default="palette", choices=["palette", "legend"],
                        help="palette: the fine-tuning prompt and fixed palette; legend: colour legend + mask (cocostuff).")
    parser.add_argument("--max_text_tokens", type=int, default=1024, help="--format legend: legend length limit.")
    parser.add_argument("--num_shards", type=int, default=1, help="Split the images over N processes, e.g. one per GPU.")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--generate_only", action="store_true",
                        help="Write this shard's files and skip the metrics; a final run without sharding computes them.")
    args = parser.parse_args()

    ds = get_dataset(args)
    if args.format == "legend" and not hasattr(ds, "legend_prompt"):
        parser.error(f"--format legend is not implemented for {ds.name}")
    out = args.output_dir
    for sub in ("raw", "pred"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    keys = ds.keys(args.split, args.limit)
    todo = [k for k in keys[args.shard_id::args.num_shards] if not os.path.exists(os.path.join(out, "pred", f"{k}.png"))]
    print(f"[img2palette] dataset={ds.name} format={args.format} split={args.split} images={len(keys)} "
          f"shard={args.shard_id}/{args.num_shards} remaining={len(todo)} adapter={args.adapter}", flush=True)
    log_path = os.path.join(out, "log.jsonl" if args.num_shards == 1 else f"log.shard{args.shard_id}.jsonl")

    if todo:
        model, info = load_model(args)
        params, transforms = generation_params(args, ds)
        if args.shard_id == 0:
            with open(os.path.join(out, "config.json"), "w") as f:
                json.dump({**vars(args), **info, "inference_overrides": params}, f, indent=2, default=str)
        start = time.time()
        for n, key in enumerate(todo, 1):
            t0 = time.time()
            photo, _ = ds.load(key)
            seed = args.seed + int(key) % 1_000_000
            if args.format == "legend":
                raw, label, rec = segment_with_legend(model, ds, photo, seed, params, transforms, args.max_text_tokens)
                note = f"legend={rec['legend_entries']} unmatched={len(rec['unmatched'])}"
            else:
                raw, label, off = segment_to_palette(model, ds, photo, seed, params, transforms)
                rec, note = {"off_palette": off}, f"off_palette={off:.3f}"
            raw.save(os.path.join(out, "raw", f"{key}.png"))
            Image.fromarray(label).save(os.path.join(out, "pred", f"{key}.png"))
            with open(log_path, "a") as f:
                f.write(json.dumps({"key": key, "seconds": round(time.time() - t0, 2), **rec}) + "\n")
            eta = (time.time() - start) / n * (len(todo) - n) / 60
            print(f"[img2palette] {n}/{len(todo)} {key} {time.time() - t0:.1f}s {note} eta={eta:.0f}min", flush=True)

    if args.generate_only:
        print(f"[img2palette] shard {args.shard_id}/{args.num_shards} done; metrics come from a final unsharded run")
        return

    conf = np.zeros((ds.num_ids, ds.num_ids), np.int64)
    for key in keys:
        conf += confusion(ds, ds.load(key)[1], np.asarray(Image.open(os.path.join(out, "pred", f"{key}.png"))))
    records = {}  # one record per image from every shard's log (the last one wins on a re-run)
    for path in sorted(glob.glob(os.path.join(out, "log*.jsonl"))):
        for line in open(path):
            rec = json.loads(line)
            records[rec["key"]] = rec
    logs = [records[k] for k in keys if k in records]
    metrics = {"dataset": ds.name, "format": args.format, "num_images": len(keys), **summarize(ds, conf),
               "seconds_per_image": float(np.mean([l["seconds"] for l in logs]))}
    if args.format == "legend":
        entries = sum(l["legend_entries"] for l in logs)
        metrics.update(mean_legend_entries=entries / len(logs),
                       unmatched_phrase_share=sum(len(l["unmatched"]) for l in logs) / max(entries, 1),
                       empty_legend_rate=float(np.mean([l["legend_entries"] == 0 for l in logs])))
        note = (f"legend entries {metrics['mean_legend_entries']:.1f}/image, unmatched "
                f"{metrics['unmatched_phrase_share']:.3f}, empty {metrics['empty_legend_rate']:.3f}")
    else:
        metrics["mean_off_palette_share"] = float(np.mean([l["off_palette"] for l in logs]))
        note = f"off-palette {metrics['mean_off_palette_share']:.4f}"
    metrics["confusion_matrix"] = conf.tolist()
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
          f"pixel acc {metrics['pixel_accuracy']:.4f}, {note}, {metrics['seconds_per_image']:.1f}s/image -> "
          f"{out}/metrics.json")


if __name__ == "__main__":
    main()
