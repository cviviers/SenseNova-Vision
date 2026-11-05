# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Rename colour-legend entries by how well each name explains the photo (analysis by synthesis), COCO-Stuff.

A symm_legend model segments a photo by writing a colour legend and then drawing the map in the legend's colours. On
COCO-Stuff most legend errors are swaps between related classes (wall-other / wall-concrete, fog / sky, bush / grass,
house / building), while the map itself is drawn well. This script keeps the drawn map and, for every legend entry
whose class belongs to a group of related classes, tries each name of the group: the map is redrawn flat in the legend
colours, and each candidate legend is scored by the flow's denoising error on the photo (and the map) at a few noise
levels, with the same noise for every candidate, over the latent tokens the entry covers. The name with the lowest
error wins if it beats the original by --min_margin. Each entry is decided with the other entries unchanged; the
metrics also report the mIoU for a range of margins.

    python -m inference.benchmark.legend_rescore --dataset cocostuff --adapter <dir> --symm \\
        --legend_dir /data/runs/<run>/legends --raw_dir /data/runs/<run>-e50g2/img2legend/raw \\
        --output_dir <out> --limit 300

--states diagonal scores the symmetric flow's own states (photo at t, map at 1 - t, both streams); map_clean holds the
map clean and scores the photo stream alone (models trained with symm_cond_prob). Without --raw_dir the map is
segmented here (--symm_seg_steps, --symm_cfg_scale). Outputs (resumable): raw/<key>.png (the drawn map), orig/ and
pred/<key>.png (label maps before and after), legends/<key>.txt, log.jsonl, metrics.json (mIoU before and after and per
margin, and how often the chosen name is the group's best match in the ground truth).
"""

import argparse
import copy
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data.seg_multitask import SYMM_ROPE_IMAGE, dequantize
from inference.benchmark.seg_eval_common import (
    add_model_args,
    confusion,
    get_dataset,
    legend_to_label,
    load_model,
    segment_symm_legend,
    summarize,
    symm_flow,
)
from utils import extract_ordered_caption_instances
from utils.parsing_output import class_mask_from_raw_output

# Related COCO-Stuff classes (the stuff hierarchy of COCO-Stuff, with fog joined to sky, potted plant to the plants,
# and furniture stuff and things merged).
GROUPS = {
    "water": ["river", "sea", "waterdrops", "water-other"],
    "ground": ["dirt", "gravel", "mud", "pavement", "platform", "playingfield", "railroad", "road", "sand", "snow",
               "ground-other"],
    "solid": ["hill", "mountain", "rock", "stone", "wood", "solid-other"],
    "sky": ["clouds", "sky-other", "fog"],
    "plant": ["branch", "bush", "flower", "grass", "leaves", "moss", "plant-other", "straw", "tree", "potted plant"],
    "structural": ["cage", "fence", "net", "railing", "structural-other"],
    "building": ["bridge", "building-other", "house", "roof", "skyscraper", "tent"],
    "food": ["food-other", "fruit", "salad", "vegetable"],
    "textile": ["banner", "blanket", "cloth", "clothes", "curtain", "mat", "napkin", "pillow", "rug", "textile-other",
                "towel"],
    "furniture": ["cabinet", "counter", "cupboard", "desk-stuff", "door-stuff", "furniture-other", "light",
                  "mirror-stuff", "shelf", "stairs", "table", "dining table", "chair", "couch", "bed"],
    "window": ["window-blind", "window-other"],
    "ceiling": ["ceiling-other", "ceiling-tile"],
    "floor": ["carpet", "floor-marble", "floor-other", "floor-stone", "floor-tile", "floor-wood"],
    "wall": ["wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile", "wall-wood"],
    "raw-material": ["cardboard", "metal", "paper", "plastic"],
    "vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
}


def group_of(ds):
    """{class id: ids of its group}, for groups whose legend names map back to their ids."""
    index = {n: i for i, n in enumerate(ds.class_names)}
    out = {}
    for members in GROUPS.values():
        ids = [index[n] for n in members if ds.legend_class_id(ds.m.legend_name(index[n])) == index[n]]
        for i in ids:
            out[i] = ids
    return out


def replace_entry(text, i, name):
    """The legend text with the i-th entry's phrase replaced by name (colours and order unchanged)."""
    match = list(re.finditer(r"<p>(.*?)<color>", text))[i]
    return text[: match.start(1)] + name + text[match.end(1):]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--legend_dir", required=True, help="The model's own legends, <key>.txt (a --legend_cache).")
    parser.add_argument("--raw_dir", default=None, help="raw/ of a segmentation run with those legends.")
    parser.add_argument("--states", choices=("diagonal", "map_clean"), default="diagonal")
    parser.add_argument("--levels", default="0.3,0.5,0.7", help="Photo noise levels to score at.")
    parser.add_argument("--noise_seeds", type=int, default=1, help="Noise draws per level.")
    parser.add_argument("--min_share", type=float, default=0.005, help="Rescore entries covering at least this share.")
    parser.add_argument("--min_margin", type=float, default=0.0,
                        help="Change a name only if its error is lower than the original's by more than this.")
    parser.add_argument("--report_margins", default="0,0.0003,0.001,0.003,0.01,0.03",
                        help="Margins for which metrics.json reports the mIoU.")
    parser.add_argument("--max_text_tokens", type=int, default=1024)
    args = parser.parse_args()
    args.symm, args.format, args.legend_cache = True, "legend", args.legend_dir
    if args.dataset != "cocostuff":
        parser.error("the class groups are defined for cocostuff")

    ds = get_dataset(args)
    out = args.output_dir
    for sub in ("raw", "orig", "pred", "legends"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    keys = ds.keys("val", args.limit)
    todo = [k for k in keys if not os.path.exists(f"{out}/pred/{k}.png")]
    levels = [float(v) for v in args.levels.split(",")]
    print(f"[rescore] images={len(keys)} remaining={len(todo)} states={args.states} levels={levels}", flush=True)

    if todo:
        model, info = load_model(args)
        flow = symm_flow(model, ds, args)
        groups = group_of(ds)
        with open(f"{out}/config.json", "w") as f:
            json.dump({**vars(args), **info}, f, indent=2, default=str)

    for n, k in enumerate(todo, 1):
        photo, gt = ds.load(k)
        seed = args.seed + int(k) % 1_000_000
        if args.raw_dir:
            text = open(f"{args.legend_dir}/{k}.txt").read()
            raw = Image.open(f"{args.raw_dir}/{k}.png").convert("RGB")
        else:
            raw, _, rec = segment_symm_legend(flow, ds, photo, seed, args, k)
            text = rec["text"]
        if raw.size != photo.size:
            raw = raw.resize(photo.size, Image.NEAREST)
        entries = extract_ordered_caption_instances(text)
        orig_label, _ = legend_to_label(ds, raw, text, photo.size)
        record = {"key": k, "entries": len(entries), "changes": []}
        new_text = text
        if entries and len(re.findall(r"<p>(.*?)<color>", text)) == len(entries):
            parsed = class_mask_from_raw_output(raw, text)
            index = parsed["class_mask"]
            palette = np.asarray(parsed["class_define"], dtype=np.uint8)  # entry colours, then black
            flat = Image.fromarray(palette[index])
            flat = dequantize(flat, random.Random(seed), ds.symm_beta)
            todo_entries = []
            for i, (phrase, _) in enumerate(entries):
                cid = ds.legend_class_id(phrase)
                if cid in groups and (index == i).mean() >= args.min_share:
                    todo_entries.append((i, cid))
            if todo_entries:
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                    zp, shape = flow.encode(photo)
                    zm, _ = flow.encode(flat)
                    num = zp.shape[0]
                    stream_ids = torch.cat([torch.full((num,), 1), torch.full((num,), 2)]).to(flow.device)
                    gen = torch.Generator(device="cpu").manual_seed(seed)
                    draws = []
                    for level in levels:
                        for _ in range(args.noise_seeds):
                            map_level = 1.0 - level if args.states == "diagonal" else 0.0
                            draws.append((level, map_level, torch.randn(zp.shape, generator=gen).to(flow.device),
                                          torch.randn(zm.shape, generator=gen).to(flow.device)))
                    prompt_context = flow._context(ds.symm_legend_prompt(), None)
                    cache = {}

                    def errors(legend):
                        """Per-token squared error of each draw (photo stream, plus the map stream on the diagonal)."""
                        if legend not in cache:
                            context = flow.inferencer.update_context_text(legend, copy.deepcopy(prompt_context))
                            query = flow._query(context, shape, [SYMM_ROPE_IMAGE, context["ropes"][0]])
                            per_token = torch.zeros(num, device=flow.device)
                            for level, map_level, noise_p, noise_m in draws:
                                x = torch.cat([(1 - level) * zp + level * noise_p,
                                               (1 - map_level) * zm + map_level * noise_m])
                                timestep = torch.cat([torch.full((num,), level), torch.full((num,), map_level)])
                                v = model.model._forward_flow(
                                    x_t=x, timestep=timestep.to(flow.device), per_token_timesteps=True,
                                    packed_stream_ids=stream_ids, past_key_values=context["past_key_values"],
                                    **query).float()
                                per_token += ((v[:num] - (noise_p - zp)) ** 2).mean(-1)
                                if args.states == "diagonal":
                                    per_token += ((v[num:] - (noise_m - zm)) ** 2).mean(-1)
                            cache[legend] = per_token / len(draws)
                        return cache[legend]

                    stride = model.model.latent_downsample
                    for i, cid in todo_entries:
                        region = index == i
                        weight = F.avg_pool2d(torch.from_numpy(region.astype(np.float32))[None, None], stride)
                        weight = weight.flatten().to(flow.device)  # share of each latent token inside the entry
                        scores = {}
                        for g in groups[cid]:
                            legend = text if g == cid else replace_entry(text, i, ds.m.legend_name(g))
                            scores[g] = float((errors(legend) * weight).sum() / weight.sum())
                        best = min(scores, key=scores.get)
                        overlap = {g: int((gt[region] == g).sum()) for g in groups[cid]}
                        oracle = max(overlap, key=overlap.get) if max(overlap.values()) > 0 else cid
                        margin = scores[cid] - scores[best]
                        record["changes"].append({"entry": i, "from": int(cid), "to": int(best), "oracle": int(oracle),
                                                  "share": round(float(region.mean()), 4),
                                                  "margin": round(margin, 6)})
                        if best != cid and margin > args.min_margin:
                            new_text = replace_entry(new_text, i, ds.m.legend_name(best))
        raw.save(f"{out}/raw/{k}.png")
        new_label, _ = legend_to_label(ds, raw, new_text, photo.size)
        Image.fromarray(orig_label).save(f"{out}/orig/{k}.png")
        with open(f"{out}/legends/{k}.txt", "w") as f:
            f.write(new_text)
        with open(f"{out}/log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        Image.fromarray(new_label).save(f"{out}/pred/{k}.png")  # written last: marks the key done for resume
        changed = sum(c["from"] != c["to"] for c in record["changes"])
        print(f"[rescore] {n}/{len(todo)} {k} entries={len(entries)} rescored={len(record['changes'])} "
              f"changed={changed}", flush=True)

    conf_orig = np.zeros((ds.num_ids, ds.num_ids), np.int64)
    conf_new = np.zeros_like(conf_orig)
    for k in keys:
        gt = ds.load(k)[1]
        conf_orig += confusion(ds, gt, np.asarray(Image.open(f"{out}/orig/{k}.png")))
        conf_new += confusion(ds, gt, np.asarray(Image.open(f"{out}/pred/{k}.png")))
    records = {}
    for line in open(f"{out}/log.jsonl"):
        rec = json.loads(line)
        records[rec["key"]] = rec
    changes = [c for k in keys for c in records[k]["changes"]]
    by_margin = {}
    for threshold in [float(v) for v in args.report_margins.split(",")]:
        conf = np.zeros_like(conf_orig)
        for k in keys:
            photo, gt = ds.load(k)
            raw = Image.open(f"{out}/raw/{k}.png").convert("RGB")
            text = open(f"{args.legend_dir}/{k}.txt").read()
            for c in records[k]["changes"]:
                if c["to"] != c["from"] and c["margin"] > threshold:
                    text = replace_entry(text, c["entry"], ds.m.legend_name(c["to"]))
            conf += confusion(ds, gt, legend_to_label(ds, raw, text, photo.size)[0])
        changed = [c for c in changes if c["to"] != c["from"] and c["margin"] > threshold]
        by_margin[str(threshold)] = {"mIoU": summarize(ds, conf)["mIoU"], "changed": len(changed),
                                     "changes_correct": float(np.mean([c["to"] == c["oracle"] for c in changed]))
                                     if changed else None,
                                     "changes_were_correct": float(np.mean([c["from"] == c["oracle"] for c in changed]))
                                     if changed else None}
    metrics = {"num_images": len(keys), "states": args.states, "levels": levels,
               "mIoU_orig": summarize(ds, conf_orig)["mIoU"], "mIoU_rescored": summarize(ds, conf_new)["mIoU"],
               "entries_rescored": len(changes), "entries_changed": sum(c["from"] != c["to"] for c in changes),
               "orig_matches_oracle": float(np.mean([c["from"] == c["oracle"] for c in changes])) if changes else None,
               "chosen_matches_oracle": float(np.mean([c["to"] == c["oracle"] for c in changes])) if changes else None,
               "by_margin": by_margin}
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)
    print("[rescore] " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
