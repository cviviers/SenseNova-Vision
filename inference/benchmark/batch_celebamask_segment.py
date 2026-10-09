# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Face parsing on CelebAMask-HQ with SenseNova-Vision.

Writes one uint8 label PNG per image (512x512, the annotation resolution) using
the CelebAMask-HQ class ids, so predictions can be scored against the ground
truth with tools/evaluation/segment/evaluate_celebamask.py.

Two ways of asking the model for a multi-class face parse:

* ``binary`` (default): one binary referring-segmentation request per class
  ("<p>left eye</p>" -> white-on-black mask). This is how face-parsing data
  (LaPa) was used in training (paper, Table 12), so it is in distribution, but it
  costs one image generation per class.
* ``generic``: one request per image listing all classes; the model answers with
  a colour legend (<p>class<color>(R,G,B)</color></p>) and a colour-coded mask,
  decoded through the legend. This is the paper's semantic/panoptic segmentation
  route (Sec. 3, "generic segmentation"), but face parts were never trained in
  this format, so expect spurious or missing classes.
* ``palette``: one request per image that assigns each class a fixed hex colour
  in the prompt ("The hair is #E6194B. ... The background is #000000."); the
  generated image is decoded by snapping each pixel to the nearest requested
  colour. One generation per image and no legend text to parse, but the paper
  treats this as exploratory (Fig. 19: colours or regions can be off).

Binary masks can overlap, so they are merged into one label map with
``--merge``: ``area`` (default) paints larger masks first so smaller parts stay
on top (in pilots "lower lip" often covered both lips and would otherwise erase
"upper lip"); ``class_order`` paints in class-id order, the rule CelebAMask-HQ
uses to build its own label maps. The raw per-class masks are kept in
``masks/<id>.npz``, so ``--remerge_only`` can rebuild predictions with another
merge rule without running the model again.

A binary request always returns a mask, even for a part that is not in the
image: in pilots "hat" on a bare head produced masks covering 30-45% of the
image, which then overwrote the real parts. For classes that are often absent
(``--gate_classes``, accessories and clothes by default) the binary method
first asks the model a yes/no question in understanding mode (~0.3s) and only
segments the class on "yes".

Runs are resumable (existing predictions are skipped) and can be sharded across
GPUs or machines with --num_shards / --shard_id.
"""

import argparse
import json
import os
import time

import numpy as np
from PIL import Image

from data.prompts import CAPTION_QUESTION_LIST, MASK_QUESTION_LIST
from inference.sensenova_vision import BASE_PARAMS, SenseNovaVisionModel
from utils import normalize_category
from utils.parsing_output import class_mask_from_raw_output

# CelebAMask-HQ class ids (0 = background) with the annotation-file suffix and
# the phrase used to prompt the model. "mouth" in CelebAMask-HQ is the inner
# mouth between the lips; left/right refer to the subject's side, which matches
# the model's convention.
CLASSES = [
    # (id, annotation suffix, prompt phrase)
    (1, "skin", "face skin"),
    (2, "nose", "nose"),
    (3, "eye_g", "eyeglasses"),
    (4, "l_eye", "left eye"),
    (5, "r_eye", "right eye"),
    (6, "l_brow", "left eyebrow"),
    (7, "r_brow", "right eyebrow"),
    (8, "l_ear", "left ear"),
    (9, "r_ear", "right ear"),
    (10, "mouth", "inner mouth"),
    (11, "u_lip", "upper lip"),
    (12, "l_lip", "lower lip"),
    (13, "hair", "hair"),
    (14, "hat", "hat"),
    (15, "ear_r", "earring"),
    (16, "neck_l", "necklace"),
    (17, "neck", "neck"),
    (18, "cloth", "clothes"),
]
CLASS_NAMES = ["background"] + [suffix for _, suffix, _ in CLASSES]
# Yes/no questions used to gate classes that are frequently absent.
PRESENCE_QUESTIONS = {
    "eye_g": "Is the person wearing eyeglasses or sunglasses? Answer yes or no.",
    "hat": "Is the person wearing a hat or cap? Answer yes or no.",
    "ear_r": "Is the person wearing earrings? Answer yes or no.",
    "neck_l": "Is the person wearing a necklace? Answer yes or no.",
    "cloth": "Is any clothing visible in the image? Answer yes or no.",
}
# Extra spellings accepted when decoding the `generic` legend.
GENERIC_ALIASES = {
    "skin": 1, "face": 1, "glasses": 3, "eye glasses": 3, "eyeglass": 3,
    "mouth": 10, "lip": 11, "earrings": 15, "necklaces": 16, "cloth": 18,
    "clothing": 18, "shirt": 18,
}

# CelebA identities 162771-182637 (1-based file numbers) form the validation
# split; this is the rule used by CelebAMask-HQ's own g_partition.py on orig_idx.
SPLIT_RANGES = {"train": (0, 162771), "val": (162771, 182638), "test": (182638, 10**9)}


def load_split(dataset_root, split):
    lo, hi = SPLIT_RANGES[split]
    ids = []
    with open(os.path.join(dataset_root, "CelebA-HQ-to-CelebA-mapping.txt")) as f:
        next(f)
        for line in f:
            idx, orig_idx, _ = line.split()
            if lo <= int(orig_idx) < hi:
                ids.append(int(idx))
    return sorted(ids)


def binary_prompt(phrase):
    return MASK_QUESTION_LIST[0].format(categories=f"<p>{phrase}</p>", task_type="binary")


def generic_prompt(phrases):
    categories = ", ".join(f"<p>{p}</p>" for p in phrases)
    return (
        MASK_QUESTION_LIST[0].format(categories=categories, task_type="panoptic")
        + " "
        + CAPTION_QUESTION_LIST[0].format(task_type="panoptic")
    )


def merge_masks(masks, merge, size):
    """masks: {class_id: bool HxW}. Returns a uint8 label map (0 = background)."""
    label = np.zeros((size, size), dtype=np.uint8)
    if merge == "area":
        order = sorted(masks, key=lambda cid: -int(masks[cid].sum()))
    else:
        order = sorted(masks)
    for class_id in order:
        label[masks[class_id]] = class_id
    return label


def save_masks(path, masks, size):
    ids = np.array(sorted(masks), dtype=np.uint8)
    packed = np.packbits(np.stack([masks[i] for i in ids]) if len(ids) else np.zeros((0, size, size), bool), axis=-1)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, class_ids=ids, packed=packed, size=size)
    os.replace(tmp, path)


def load_masks(path):
    data = np.load(path)
    size = int(data["size"])
    unpacked = np.unpackbits(data["packed"], axis=-1, count=size).astype(bool)
    return {int(cid): unpacked[i] for i, cid in enumerate(data["class_ids"])}, size


def segment_binary(model, image, classes, params, seed, size, gate_classes=()):
    masks, presence = {}, {}
    for class_id, suffix, phrase in classes:
        if suffix in gate_classes:
            presence[suffix] = ask_presence(model, image, suffix)
            if not presence[suffix]:
                continue
        mask = model.generate(
            contents=[{"type": "image", "value": image}, {"type": "text", "value": binary_prompt(phrase)}],
            mode="dense_perception",
            noise_seed=seed,
            **params,
        )
        masks[class_id] = np.asarray(mask.convert("L").resize((size, size), Image.BILINEAR)) > 127
    return masks, {"presence": presence}


# Visually distinct colours (the hex palette used in the paper's Fig. 19 examples).
PALETTE_HEX = ["#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4", "#46F0F0", "#F032E6",
               "#BCF60C", "#FABEBE", "#008080", "#E6BEFF", "#9A6324", "#FFFAC8", "#800000", "#AAFFC3",
               "#808000", "#FFD8B1"]
BACKGROUND_HEX = "#000000"


def palette_prompt(classes):
    parts = [f"The {phrase} {'are' if phrase in ('eyeglasses', 'clothes') else 'is'} {PALETTE_HEX[class_id - 1]}."
             for class_id, _, phrase in classes]
    return ("Generate a semantic segmentation visualization of the input. "
            + " ".join(parts) + f" The background is {BACKGROUND_HEX}.")


def ask_presence(model, image, suffix):
    answer = model.generate(
        contents=[{"type": "image", "value": image}, {"type": "text", "value": PRESENCE_QUESTIONS[suffix]}],
        mode="understanding",
    )
    return answer.strip().lower().startswith("yes")


def segment_palette(model, image, classes, params, seed, size, gate_classes=()):
    presence = {suffix: ask_presence(model, image, suffix) for _, suffix, _ in classes if suffix in gate_classes}
    classes = [c for c in classes if presence.get(c[1], True)]
    out = model.generate(
        contents=[{"type": "image", "value": image}, {"type": "text", "value": palette_prompt(classes)}],
        mode="dense_perception",
        noise_seed=seed,
        **params,
    )
    rgb = np.asarray(out.convert("RGB").resize((size, size), Image.NEAREST)).astype(np.int32)
    raw_image = out.convert("RGB")
    ids = [0] + [class_id for class_id, _, _ in classes]
    colours = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)]
                        for h in [BACKGROUND_HEX] + [PALETTE_HEX[c - 1] for c in ids[1:]]], dtype=np.int32)
    dist = ((rgb[:, :, None, :] - colours[None, None]) ** 2).sum(-1)
    label = np.array(ids, dtype=np.uint8)[dist.argmin(-1)]
    return ({cid: label == cid for cid in ids[1:] if (label == cid).any()},
            {"presence": presence, "_raw_image": raw_image})


def segment_generic(model, image, classes, params, seed, size, gate_classes=()):
    name_to_id = {normalize_category(phrase): cid for cid, _, phrase in classes}
    name_to_id.update({k: v for k, v in GENERIC_ALIASES.items() if v in {c for c, _, _ in classes}})
    output = model.generate(
        contents=[{"type": "image", "value": image},
                  {"type": "text", "value": generic_prompt([p for _, _, p in classes])}],
        mode="caption_generate",
        noise_seed=seed,
        return_intermediate_outputs=True,
        **params,
    )
    text = output.get("text", "")
    mask_image = output.get("image")
    if isinstance(mask_image, list):
        mask_image = mask_image[0] if mask_image else None
    label = np.zeros((size, size), dtype=np.uint8)
    unmatched = []
    if mask_image is not None:
        parsed = class_mask_from_raw_output(mask_image.resize((size, size), Image.NEAREST), text)
        for instance_idx, phrase in enumerate(parsed["gcg_phrases"]):
            base = phrase.rsplit("-", 1)[0] if phrase.rsplit("-", 1)[-1].isdigit() else phrase
            class_id = name_to_id.get(normalize_category(base))
            if class_id is None:
                unmatched.append(phrase)
                continue
            label[parsed["class_mask"] == instance_idx] = class_id
    return {cid: label == cid for cid in np.unique(label) if cid}, {"text": text, "unmatched_phrases": unmatched}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", default="/datasets/CelebAMask-HQ/CelebAMask-HQ",
                        help="Directory containing CelebA-HQ-img/ and CelebA-HQ-to-CelebA-mapping.txt.")
    parser.add_argument("--split", default="val", choices=list(SPLIT_RANGES))
    parser.add_argument("--output_dir", default="results/celebamask_hq/val_binary")
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "sensenova/SenseNova-Vision-7B-MoT"))
    parser.add_argument("--dtype", default=os.environ.get("SENSENOVA_DTYPE", "bf16"))
    parser.add_argument("--method", default="binary", choices=["binary", "generic", "palette"])
    parser.add_argument("--classes", default=None,
                        help="Comma-separated annotation names to segment (e.g. hair,skin,l_eye). Default: all 18.")
    parser.add_argument("--gate_classes", default=",".join(PRESENCE_QUESTIONS),
                        help="Binary/palette methods: classes kept only if the model answers 'yes' to a presence "
                             f"question. Any of {list(PRESENCE_QUESTIONS)}; pass '' to disable.")
    parser.add_argument("--merge", default="area", choices=["area", "class_order"],
                        help="How overlapping class masks are combined (see module docstring).")
    parser.add_argument("--remerge_only", action="store_true",
                        help="Rebuild pred/ from saved masks/ with --merge; does not load the model.")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Resize inputs to this size before inference. 512 matches the annotations; "
                             "1024 (native) is ~4x slower.")
    parser.add_argument("--num_timesteps", type=int, default=None,
                        help="Diffusion steps per mask (model default: 50). 20 was close to 50 in a pilot.")
    parser.add_argument("--cfg_text_scale", type=float, default=None,
                        help="Text classifier-free guidance (model default: 4.0). 1.0 disables it, halving "
                             "the forward passes per diffusion step.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N images of the split (pilot runs).")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    classes = CLASSES
    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
        unknown = wanted - {suffix for _, suffix, _ in CLASSES}
        if unknown:
            parser.error(f"unknown classes {sorted(unknown)}; choose from {CLASS_NAMES[1:]}")
        classes = [c for c in CLASSES if c[1] in wanted]

    gate_classes = {c.strip() for c in args.gate_classes.split(",") if c.strip()}
    if gate_classes - set(PRESENCE_QUESTIONS):
        parser.error(f"--gate_classes supports only {list(PRESENCE_QUESTIONS)}")

    ids = load_split(args.dataset_root, args.split)[: args.limit]
    ids = ids[args.shard_id :: args.num_shards]
    pred_dir = os.path.join(args.output_dir, "pred")
    mask_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    if args.remerge_only:
        done = [i for i in ids if os.path.exists(os.path.join(mask_dir, f"{i}.npz"))]
        for image_id in done:
            masks, size = load_masks(os.path.join(mask_dir, f"{image_id}.npz"))
            Image.fromarray(merge_masks(masks, args.merge, size)).save(os.path.join(pred_dir, f"{image_id}.png"))
        print(f"[celebamask] re-merged {len(done)} images with merge={args.merge}")
        return

    todo = [i for i in ids if not os.path.exists(os.path.join(pred_dir, f"{i}.png"))]
    print(f"[celebamask] split={args.split} images={len(ids)} remaining={len(todo)} method={args.method} "
          f"classes={len(classes)} image_size={args.image_size} merge={args.merge}", flush=True)
    with open(os.path.join(args.output_dir, f"config_shard{args.shard_id}.json"), "w") as f:
        json.dump({**vars(args), "class_names": CLASS_NAMES,
                   "prompted_classes": {cid: phrase for cid, _, phrase in classes}}, f, indent=2)
    if not todo:
        return

    model = SenseNovaVisionModel(model_path=args.model_path, dtype=args.dtype)
    mode = "caption_generate" if args.method == "generic" else "dense_perception"
    params = dict(BASE_PARAMS[mode])
    if args.num_timesteps:
        params["num_timesteps"] = args.num_timesteps
    if args.cfg_text_scale is not None:
        params["cfg_text_scale"] = args.cfg_text_scale
    segment = {"binary": segment_binary, "generic": segment_generic, "palette": segment_palette}[args.method]

    log_path = os.path.join(args.output_dir, f"log_shard{args.shard_id}.jsonl")
    start = time.time()
    for n, image_id in enumerate(todo, 1):
        t0 = time.time()
        image = Image.open(os.path.join(args.dataset_root, "CelebA-HQ-img", f"{image_id}.jpg")).convert("RGB")
        if args.image_size and image.size != (args.image_size, args.image_size):
            image = image.resize((args.image_size, args.image_size), Image.BICUBIC)
        masks, extra = segment(model, image, classes, params, args.seed + image_id, size=512,
                               gate_classes=gate_classes)
        raw_image = extra.pop("_raw_image", None)
        if raw_image is not None:  # palette method: keep the generated colour map for inspection
            os.makedirs(os.path.join(args.output_dir, "raw"), exist_ok=True)
            raw_image.save(os.path.join(args.output_dir, "raw", f"{image_id}.png"))
        save_masks(os.path.join(mask_dir, f"{image_id}.npz"), masks, 512)
        # Write atomically so an interrupted run never leaves a half-written PNG that resume would skip.
        tmp = os.path.join(pred_dir, f".{image_id}.png.tmp")
        Image.fromarray(merge_masks(masks, args.merge, 512)).save(tmp, format="PNG")
        os.replace(tmp, os.path.join(pred_dir, f"{image_id}.png"))
        with open(log_path, "a") as f:
            f.write(json.dumps({"id": image_id, "seconds": round(time.time() - t0, 2), **extra}) + "\n")
        eta_h = (time.time() - start) / n * (len(todo) - n) / 3600
        print(f"[celebamask] {n}/{len(todo)} id={image_id} {time.time() - t0:.1f}s eta={eta_h:.1f}h", flush=True)


if __name__ == "__main__":
    main()
