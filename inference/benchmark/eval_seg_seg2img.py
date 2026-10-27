# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the seg2img mode (palette segmentation map -> photo) on CelebAMask-HQ or COCO-Stuff.

For every ground-truth map (and seed): render it in the dataset palette at the training image size,
generate a photo with the training prompt in edit mode, and measure

* photo rate   the model's yes/no check whether the output is a realistic photograph (of a face, for
               CelebAMask-HQ); copied maps and silhouettes fail it.
* layout       round trip: the generated photo is segmented with the img2palette mode of the SAME model and
               scored (mIoU, dataset protocol) against the input map; outputs failing the photo check count
               as unlabeled/background (a copied map would otherwise score near-perfectly). The same
               segmenter on the real photos gives the ceiling; layout_ratio = round trip / ceiling.
               Generator and judge share weights, so read it as a consistency check alongside FID.
* FID / KID    clean-fid ("clean" resize) between the generated photos and the real photos of the same
               images at the same size. FID is strongly biased below ~2k images: compare runs at equal N,
               or rely on KID for small sets.

Outputs (resumable per image/seed):
    gen/<key>_s<seed>.png    generated photos            ref/<key>.png    real photos (training size)
    reseg/<key>_s<seed>.png  round-trip label maps       log.jsonl        per image/seed records
    sheet.jpg                map | generated | real | round trip, first 8 images
    metrics.json

    python -m inference.benchmark.eval_seg_seg2img --dataset cocostuff \\
        --adapter results/cocostuff_lora_i2p_s2i/final --output_dir results/cocostuff_eval/final/seg2img --limit 500
"""

import argparse
import json
import os
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


def is_photo(model, ds, image):
    answer = model.generate(contents=[{"type": "image", "value": image}, {"type": "text", "value": ds.photo_question}],
                            mode="understanding")
    return answer.strip().lower().startswith("yes")


def fid_kid(gen_dir, ref_dir):
    from cleanfid import fid

    return {"fid": float(fid.compute_fid(gen_dir, ref_dir, mode="clean", verbose=False)),
            "kid": float(fid.compute_kid(gen_dir, ref_dir, mode="clean", verbose=False))}


def write_sheet(out, ds, keys, labels, size=192):
    rows = [k for k in keys if os.path.exists(f"{out}/gen/{k}_s0.png")][:8]
    if not rows:
        return
    sheet = Image.new("RGB", (4 * size, len(rows) * size), "white")
    for r, k in enumerate(rows):
        reseg = f"{out}/reseg/{k}_s0.png"
        tiles = [Image.fromarray(ds.palette[labels[k]]), Image.open(f"{out}/gen/{k}_s0.png"),
                 Image.open(f"{out}/ref/{k}.png"),
                 Image.fromarray(ds.palette[np.asarray(Image.open(reseg))]) if os.path.exists(reseg)
                 else Image.new("RGB", (size, size), "white")]
        for c, t in enumerate(tiles):
            t = t.convert("RGB")
            t.thumbnail((size, size))
            sheet.paste(t, (c * size + (size - t.width) // 2, r * size + (size - t.height) // 2))
    sheet.save(f"{out}/sheet.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", type=int, default=1, help="Photos per map (noise seeds 0..N-1).")
    parser.add_argument("--no_roundtrip", action="store_true", help="Skip the round-trip layout check.")
    parser.add_argument("--no_photo_check", action="store_true")
    parser.add_argument("--no_fid", action="store_true")
    args = parser.parse_args()

    ds = get_dataset(args)
    out = args.output_dir
    for sub in ("gen", "ref", "reseg"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    keys = ds.keys(args.split, args.limit)
    jobs = [(k, s) for k in keys for s in range(args.seeds) if not os.path.exists(f"{out}/gen/{k}_s{s}.png")]
    print(f"[seg2img] dataset={ds.name} split={args.split} maps={len(keys)} seeds={args.seeds} remaining={len(jobs)} "
          f"adapter={args.adapter}", flush=True)

    model = None
    need_ceiling = not args.no_roundtrip and not os.path.exists(f"{out}/ceiling.json")
    if jobs or need_ceiling:
        model, info = load_model(args)
        params, transforms = generation_params(args, ds)
        with open(f"{out}/config.json", "w") as f:
            json.dump({**vars(args), **info, "inference_overrides": params}, f, indent=2, default=str)

    labels = {}
    start = time.time()
    for n, (k, s) in enumerate(jobs, 1):
        photo, label = ds.load(k)
        if not os.path.exists(f"{out}/ref/{k}.png"):
            photo.save(f"{out}/ref/{k}.png")
        t0 = time.time()
        gen = model.generate(contents=[{"type": "image", "value": Image.fromarray(ds.palette[label])},
                                       {"type": "text", "value": ds.seg2img_prompt(label)}],
                             mode="edit", noise_seed=1000 * s + int(k) % 1_000_000, **params, **transforms)
        gen = gen.convert("RGB").resize(photo.size, Image.BICUBIC)
        rec = {"key": k, "seed": s, "seconds": round(time.time() - t0, 2)}
        if not args.no_photo_check:
            rec["is_photo"] = is_photo(model, ds, gen)
        if not args.no_roundtrip:
            _, pred, off = segment_to_palette(model, ds, gen, args.seed + int(k) % 1_000_000, {}, transforms)
            Image.fromarray(pred).save(f"{out}/reseg/{k}_s{s}.png")
            rec["roundtrip_off_palette"] = off
        gen.save(f"{out}/gen/{k}_s{s}.png")  # written last: marks the job done for resume
        with open(f"{out}/log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        eta = (time.time() - start) / n * (len(jobs) - n) / 60
        print(f"[seg2img] {n}/{len(jobs)} {k} seed={s} {rec['seconds']:.1f}s eta={eta:.0f}min", flush=True)

    for k in keys:
        labels[k] = ds.load(k)[1]
    # Ceiling: the same img2palette segmenter on the real photos of these maps.
    if need_ceiling:
        conf = np.zeros((ds.num_ids, ds.num_ids), np.int64)
        for k in keys:
            _, pred, _ = segment_to_palette(model, ds, ds.load(k)[0], args.seed + int(k) % 1_000_000, {}, transforms)
            conf += confusion(ds, labels[k], pred)
        json.dump({"mIoU": summarize(ds, conf)["mIoU"], "num_images": len(keys)}, open(f"{out}/ceiling.json", "w"))

    logs = [json.loads(l) for l in open(f"{out}/log.jsonl")]
    metrics = {"dataset": ds.name, "num_maps": len(keys), "num_generated": len(logs),
               "seconds_per_image": float(np.mean([l["seconds"] for l in logs]))}
    if not args.no_photo_check:
        metrics["photo_rate"] = float(np.mean([l["is_photo"] for l in logs]))
    if not args.no_roundtrip:
        conf = np.zeros((ds.num_ids, ds.num_ids), np.int64)
        for l in logs:
            pred = np.asarray(Image.open(f"{out}/reseg/{l['key']}_s{l['seed']}.png"))
            if not l.get("is_photo", True):
                pred = np.zeros_like(pred)  # copied maps / silhouettes count as unlabeled
            conf += confusion(ds, labels[l["key"]], pred)
        rt = summarize(ds, conf)
        ceiling = json.load(open(f"{out}/ceiling.json"))["mIoU"]
        metrics.update(roundtrip_mIoU=rt["mIoU"], ceiling_mIoU=ceiling,
                       layout_ratio=rt["mIoU"] / ceiling if ceiling else None,
                       roundtrip_per_class_IoU=rt["per_class_IoU"])
    if not args.no_fid:
        metrics.update(fid_kid(f"{out}/gen", f"{out}/ref"))
        metrics["fid_note"] = f"{len(logs)} generated vs {len(os.listdir(out + '/ref'))} real; FID is biased below ~2k"
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)
    write_sheet(out, ds, keys, labels)
    print("[seg2img] " + json.dumps({k: v for k, v in metrics.items() if k != "roundtrip_per_class_IoU"}))


if __name__ == "__main__":
    main()
