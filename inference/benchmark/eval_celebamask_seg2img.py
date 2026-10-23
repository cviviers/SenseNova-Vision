# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the seg2img mode (CelebAMask-HQ palette map -> face photo) on a CelebAMask-HQ split.

For every ground-truth mask (and seed): render it in the official palette, generate a face with
SEG2IMG_PROMPT (the training prompt, data/celebamask.py) in edit mode, and measure

* face rate     the model's yes/no check "is this a realistic photograph of a human face?"
* layout        round trip: the generated face is segmented with the img2palette mode of the SAME
                model and scored (mIoU, 18 face classes) against the input mask. The same segmenter
                on the real photos gives the ceiling; ``layout_ratio`` = round trip / ceiling.
                Caveat: generator and judge share weights, so read it as a consistency check.
* FID / KID     clean-fid (Inception, "clean" resize) between the generated faces and the real
                photos of the same images. FID is strongly biased below ~2k images: compare runs at
                equal N, or rely on KID for small sets.

Outputs (resumable per image/seed):
    gen/<id>_s<seed>.png    generated faces            ref/<id>.png      real photos (512x512)
    reseg/<id>_s<seed>.png  round-trip label maps      log.jsonl         per image/seed records
    sheet.jpg               mask | generated | real | round trip, first 8 images
    metrics.json

    python -m inference.benchmark.eval_celebamask_seg2img \\
        --adapter results/celebamask_lora_i2p_s2i/final --output_dir results/celebamask_eval/final/seg2img --limit 500
"""

import argparse
import json
import os
import time

import numpy as np
from PIL import Image

from data.celebamask import PALETTE, SEG2IMG_PROMPT, palette_image
from inference.benchmark.batch_celebamask_mask2face import FACE_QUESTION
from inference.benchmark.celebamask_eval_common import (
    add_model_args,
    confusion,
    load_label_map,
    load_model,
    load_photo,
    miou,
    overrides,
    segment_to_palette,
    split_ids,
)


def is_face(model, image):
    answer = model.generate(contents=[{"type": "image", "value": image}, {"type": "text", "value": FACE_QUESTION}],
                            mode="understanding")
    return answer.strip().lower().startswith("yes")


def fid_kid(gen_dir, ref_dir):
    from cleanfid import fid

    return {"fid": float(fid.compute_fid(gen_dir, ref_dir, mode="clean", verbose=False)),
            "kid": float(fid.compute_kid(gen_dir, ref_dir, mode="clean", verbose=False))}


def write_sheet(out, ids, seed, dataset_root):
    tiles_per_row, size = 4, 192
    rows = [i for i in ids if os.path.exists(f"{out}/gen/{i}_s{seed}.png")][:8]
    if not rows:
        return
    sheet = Image.new("RGB", (tiles_per_row * size, len(rows) * size), "white")
    for r, i in enumerate(rows):
        tiles = [palette_image(load_label_map(dataset_root, i)), Image.open(f"{out}/gen/{i}_s{seed}.png"),
                 Image.open(f"{out}/ref/{i}.png")]
        reseg = f"{out}/reseg/{i}_s{seed}.png"
        tiles.append(Image.fromarray(PALETTE[np.asarray(Image.open(reseg))]) if os.path.exists(reseg)
                     else Image.new("RGB", (size, size), "white"))
        for c, t in enumerate(tiles):
            sheet.paste(t.convert("RGB").resize((size, size)), (c * size, r * size))
    sheet.save(f"{out}/sheet.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", type=int, default=1, help="Faces per mask (noise seeds 0..N-1).")
    parser.add_argument("--no_roundtrip", action="store_true", help="Skip the round-trip layout check.")
    parser.add_argument("--no_face_check", action="store_true")
    parser.add_argument("--no_fid", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    for sub in ("gen", "ref", "reseg"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    ids = split_ids(args)
    jobs = [(i, s) for i in ids for s in range(args.seeds) if not os.path.exists(f"{out}/gen/{i}_s{s}.png")]
    print(f"[seg2img] split={args.split} masks={len(ids)} seeds={args.seeds} remaining={len(jobs)} "
          f"adapter={args.adapter}", flush=True)

    model = None
    if jobs or (not args.no_roundtrip and not os.path.exists(f"{out}/ceiling.json")):
        model, info = load_model(args)
        params = overrides(args)
        with open(f"{out}/config.json", "w") as f:
            json.dump({**vars(args), **info, "inference_overrides": params}, f, indent=2, default=str)

    start = time.time()
    for n, (i, s) in enumerate(jobs, 1):
        label = load_label_map(args.dataset_root, i)
        if not os.path.exists(f"{out}/ref/{i}.png"):
            load_photo(args.dataset_root, i).save(f"{out}/ref/{i}.png")
        t0 = time.time()
        face = model.generate(contents=[{"type": "image", "value": palette_image(label)},
                                        {"type": "text", "value": SEG2IMG_PROMPT}],
                              mode="edit", noise_seed=1000 * s + i, **params).convert("RGB").resize((512, 512))
        rec = {"id": i, "seed": s, "seconds": round(time.time() - t0, 2)}
        if not args.no_face_check:
            rec["is_face"] = is_face(model, face)
        if not args.no_roundtrip:
            _, pred, off = segment_to_palette(model, face, args.seed + i, {})
            Image.fromarray(pred).save(f"{out}/reseg/{i}_s{s}.png")
            rec["roundtrip_off_palette"] = off
        face.save(f"{out}/gen/{i}_s{s}.png")  # written last: marks the job done for resume
        with open(f"{out}/log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        eta = (time.time() - start) / n * (len(jobs) - n) / 60
        print(f"[seg2img] {n}/{len(jobs)} id={i} seed={s} {rec['seconds']:.1f}s eta={eta:.0f}min", flush=True)

    # Ceiling: the same img2palette segmenter on the real photos of these masks.
    if not args.no_roundtrip and not os.path.exists(f"{out}/ceiling.json"):
        conf = np.zeros((19, 19), np.int64)
        for i in ids:
            _, pred, _ = segment_to_palette(model, load_photo(args.dataset_root, i), args.seed + i, {})
            conf += confusion(load_label_map(args.dataset_root, i), pred)
        json.dump({"mIoU": miou(conf)[0], "num_images": len(ids)}, open(f"{out}/ceiling.json", "w"))

    logs = [json.loads(l) for l in open(f"{out}/log.jsonl")]
    metrics = {"num_masks": len(ids), "num_generated": len(logs), "seconds_per_face": float(np.mean([l["seconds"] for l in logs]))}
    if not args.no_face_check:
        metrics["face_rate"] = float(np.mean([l["is_face"] for l in logs]))
    if not args.no_roundtrip:
        conf = np.zeros((19, 19), np.int64)
        for l in logs:
            pred = np.asarray(Image.open(f"{out}/reseg/{l['id']}_s{l['seed']}.png"))
            if not l.get("is_face", True):
                # A copied mask or silhouette segments back almost perfectly; score non-faces as background.
                pred = np.zeros_like(pred)
            conf += confusion(load_label_map(args.dataset_root, l["id"]), pred)
        rt, iou = miou(conf)
        ceiling = json.load(open(f"{out}/ceiling.json"))["mIoU"]
        metrics.update(roundtrip_mIoU=rt, ceiling_mIoU=ceiling, layout_ratio=rt / ceiling if ceiling else None,
                       roundtrip_per_class_IoU={k: float(v) for k, v in zip(range(19), iou)})
    if not args.no_fid:
        metrics.update(fid_kid(f"{out}/gen", f"{out}/ref"))
        metrics["fid_note"] = f"{len(logs)} generated vs {len(os.listdir(out + '/ref'))} real; FID is biased below ~2k"
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)
    write_sheet(out, ids, 0, args.dataset_root)
    print("[seg2img] " + json.dumps({k: v for k, v in metrics.items() if k != "roundtrip_per_class_IoU"}))


if __name__ == "__main__":
    main()
