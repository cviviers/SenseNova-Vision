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
               *_ungated scores every output regardless of the photo check (stylised but well-laid-out
               outputs the check rejects); read it together with photo_rate.
               Generator and judge share weights, so read it as a consistency check alongside FID.
* FID / KID    clean-fid ("clean" resize) between the generated photos and the real photos of the same
               images at the same size. FID is strongly biased below ~2k images: compare runs at equal N,
               or rely on KID for small sets.

Large FID runs: --num_samples 50000 reuses every map with new noise seeds until 50k photos exist,
--check_seeds 1 limits the photo check / round trip to one photo per map, and --num_shards/--shard_id
--generate_only split the work over GPUs (scripts/eval_seg_tasks.sh does this for all visible GPUs).

Zero-shot (base model): --prompt regions --cfg_img_scale 1.2. The training prompt's colour key makes the base
model return a recoloured copy of the map (photo rate 0 on pilots); region names without colours give photos.

Outputs (resumable per image/seed; shards share the directory):
    gen/<key>_s<seed>.png    generated photos            ref/<key>.png    real photos (training size)
    reseg/<key>_s<seed>.png  round-trip label maps       ceiling/<key>.png  segmenter on the real photo
    log.jsonl                per image/seed records (log.shard<i>.jsonl when sharded)
    sheet.jpg                map | generated | real | round trip, first 8 images
    metrics.json

    python -m inference.benchmark.eval_seg_seg2img --dataset cocostuff \\
        --adapter results/cocostuff_lora_i2p_s2i/final --output_dir results/cocostuff_eval/final/seg2img --limit 500
"""

import argparse
import glob
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
    parser.add_argument("--prompt", choices=("training", "regions"), default="training",
                        help="training: the fine-tuning prompt with the colour key of the classes present; regions: "
                             "region names without colours, for the base model (a colour key makes it copy the "
                             "map; use with --cfg_img_scale 1.2).")
    parser.add_argument("--seeds", type=int, default=1, help="Photos per map (noise seeds 0..N-1).")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Generate exactly N photos, cycling over the maps with a new seed each round (seed 0 for "
                             "every map first); overrides --seeds. E.g. 50000 for FID-50k.")
    parser.add_argument("--check_seeds", type=int, default=None,
                        help="Photo check and round trip only for seeds < K (default: every sample); e.g. 1 = once "
                             "per map, so a large FID run does not pay ~2.5x per photo for them.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the samples (and the ceiling maps) over N processes, e.g. one per GPU.")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--generate_only", action="store_true",
                        help="Write this shard's files and skip the metrics; a final run without sharding then "
                             "computes them from the files (nothing is generated again).")
    parser.add_argument("--no_roundtrip", action="store_true", help="Skip the round-trip layout check.")
    parser.add_argument("--no_photo_check", action="store_true")
    parser.add_argument("--no_fid", action="store_true")
    args = parser.parse_args()

    ds = get_dataset(args)
    prompt_fn = ds.seg2img_prompt if args.prompt == "training" else ds.seg2img_region_prompt
    out = args.output_dir
    for sub in ("gen", "ref", "reseg", "ceiling"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    keys = ds.keys(args.split, args.limit)
    if args.num_samples:
        rounds = -(-args.num_samples // len(keys))
        samples = [(k, s) for s in range(rounds) for k in keys][: args.num_samples]
        keys = list(dict.fromkeys(k for k, _ in samples))  # fewer samples than maps: only the maps used
    else:
        samples = [(k, s) for k in keys for s in range(args.seeds)]
    checked = lambda s: args.check_seeds is None or s < args.check_seeds
    shard = slice(args.shard_id, None, args.num_shards)
    jobs = [(k, s) for k, s in samples[shard] if not os.path.exists(f"{out}/gen/{k}_s{s}.png")]
    # Ceiling: the same img2palette segmenter on the real photos of these maps (one label map per key).
    ceiling_todo = [] if args.no_roundtrip or os.path.exists(f"{out}/ceiling.json") else \
        [k for k in keys[shard] if not os.path.exists(f"{out}/ceiling/{k}.png")]
    print(f"[seg2img] dataset={ds.name} split={args.split} maps={len(keys)} samples={len(samples)} "
          f"shard={args.shard_id}/{args.num_shards} remaining={len(jobs)} ceiling_remaining={len(ceiling_todo)} "
          f"adapter={args.adapter}", flush=True)

    model = None
    if jobs or ceiling_todo:
        model, info = load_model(args)
        params, transforms = generation_params(args, ds)
        if args.shard_id == 0:
            with open(f"{out}/config.json", "w") as f:
                json.dump({**vars(args), **info, "inference_overrides": params}, f, indent=2, default=str)

    def save_png(image, path):  # atomic: shards share the directories and resume skips existing files
        image.save(path + ".tmp", format="PNG")
        os.replace(path + ".tmp", path)

    log_path = f"{out}/log.jsonl" if args.num_shards == 1 else f"{out}/log.shard{args.shard_id}.jsonl"
    start = time.time()
    for n, (k, s) in enumerate(jobs, 1):
        photo, label = ds.load(k)
        if s == 0 and not os.path.exists(f"{out}/ref/{k}.png"):
            save_png(photo, f"{out}/ref/{k}.png")
        t0 = time.time()
        gen = model.generate(contents=[{"type": "image", "value": Image.fromarray(ds.palette[label])},
                                       {"type": "text", "value": prompt_fn(label)}],
                             mode="edit", noise_seed=1000 * s + int(k) % 1_000_000, **params, **transforms)
        gen = gen.convert("RGB").resize(photo.size, Image.BICUBIC)
        rec = {"key": k, "seed": s, "seconds": round(time.time() - t0, 2)}
        if checked(s) and not args.no_photo_check:
            rec["is_photo"] = is_photo(model, ds, gen)
        if checked(s) and not args.no_roundtrip:
            _, pred, off = segment_to_palette(model, ds, gen, args.seed + int(k) % 1_000_000, {}, transforms)
            save_png(Image.fromarray(pred), f"{out}/reseg/{k}_s{s}.png")
            rec["roundtrip_off_palette"] = off
        save_png(gen, f"{out}/gen/{k}_s{s}.png")  # written last: marks the job done for resume
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        eta = (time.time() - start) / n * (len(jobs) - n) / 60
        print(f"[seg2img] {n}/{len(jobs)} {k} seed={s} {rec['seconds']:.1f}s eta={eta:.0f}min", flush=True)

    for n, k in enumerate(ceiling_todo, 1):
        _, pred, _ = segment_to_palette(model, ds, ds.load(k)[0], args.seed + int(k) % 1_000_000, {}, transforms)
        save_png(Image.fromarray(pred), f"{out}/ceiling/{k}.png")
        print(f"[seg2img] ceiling {n}/{len(ceiling_todo)} {k}", flush=True)

    if args.generate_only:
        print(f"[seg2img] shard {args.shard_id}/{args.num_shards} done; metrics come from a final unsharded run")
        return

    labels = {k: ds.load(k)[1] for k in keys}
    if not args.no_roundtrip and not os.path.exists(f"{out}/ceiling.json"):
        conf = np.zeros((ds.num_ids, ds.num_ids), np.int64)
        for k in keys:
            conf += confusion(ds, labels[k], np.asarray(Image.open(f"{out}/ceiling/{k}.png")))
        json.dump({"mIoU": summarize(ds, conf)["mIoU"], "num_images": len(keys)}, open(f"{out}/ceiling.json", "w"))

    # One record per sample of this run, from every shard's log (the last record wins on a re-run).
    records = {}
    for path in sorted(glob.glob(f"{out}/log*.jsonl")):
        for line in open(path):
            rec = json.loads(line)
            records[(rec["key"], rec["seed"])] = rec
    logs = [records[ks] for ks in samples if ks in records]
    if len(logs) < len(samples):
        print(f"[seg2img] WARNING {len(samples) - len(logs)} samples have no log record", flush=True)
    metrics = {"dataset": ds.name, "num_maps": len(keys), "num_generated": len(logs),
               "seconds_per_image": float(np.mean([l["seconds"] for l in logs]))}
    if not args.no_photo_check:
        photo_logs = [l for l in logs if "is_photo" in l]
        metrics.update(photo_rate=float(np.mean([l["is_photo"] for l in photo_logs])), photo_checked=len(photo_logs))
    if not args.no_roundtrip:
        conf = np.zeros((ds.num_ids, ds.num_ids), np.int64)
        conf_ungated = np.zeros_like(conf)
        rt_logs = [l for l in logs if "roundtrip_off_palette" in l]
        for l in rt_logs:
            pred = np.asarray(Image.open(f"{out}/reseg/{l['key']}_s{l['seed']}.png"))
            conf_ungated += confusion(ds, labels[l["key"]], pred)
            if not l.get("is_photo", True):
                pred = np.zeros_like(pred)  # copied maps / silhouettes count as unlabeled
            conf += confusion(ds, labels[l["key"]], pred)
        rt, rt_ungated = summarize(ds, conf), summarize(ds, conf_ungated)
        ceiling = json.load(open(f"{out}/ceiling.json"))["mIoU"]
        # Gated: outputs failing the photo check score as unlabeled (catches copied maps, but also zeroes
        # stylised outputs the check rejects). Ungated: every output's re-segmentation; read it together
        # with photo_rate, since a copied map scores high here.
        metrics.update(roundtrip_mIoU=rt["mIoU"], ceiling_mIoU=ceiling,
                       layout_ratio=rt["mIoU"] / ceiling if ceiling else None,
                       roundtrip_mIoU_ungated=rt_ungated["mIoU"],
                       layout_ratio_ungated=rt_ungated["mIoU"] / ceiling if ceiling else None,
                       roundtrip_checked=len(rt_logs), roundtrip_per_class_IoU=rt["per_class_IoU"],
                       roundtrip_per_class_IoU_ungated=rt_ungated["per_class_IoU"])
    if not args.no_fid:
        n_gen, n_ref = len(glob.glob(f"{out}/gen/*.png")), len(glob.glob(f"{out}/ref/*.png"))
        metrics.update(fid_kid(f"{out}/gen", f"{out}/ref"))
        metrics["fid_note"] = f"{n_gen} generated vs {n_ref} real; FID is biased below ~2k"
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)
    write_sheet(out, ds, keys, labels)
    print("[seg2img] " + json.dumps({k: v for k, v in metrics.items() if not k.startswith("roundtrip_per_class")}))


if __name__ == "__main__":
    main()
