# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Segmentation-to-face generation on CelebAMask-HQ with SenseNova-Vision.

The inverse of batch_celebamask_segment.py: the ground-truth face-parsing map
of each image is rendered as a colour-coded mask (the CelebAMask-HQ palette used
by the visualizer) and given to the model in ``edit`` mode with a generic
instruction to turn it into a realistic photograph. The prompt is the same fixed
text for every image (``--prompt`` presets in PROMPTS, or ``--prompt_text``); it
never describes the person in the image. The exceptions are the dynamic presets
``celeb_dynamic`` and ``celeb_dynamic_short`` (DYNAMIC_PROMPTS), which append the names of the regions
present in that image's segmentation map (derived from the mask itself).

Mask-to-image synthesis is not one of the model's trained tasks. Prompts that
list the mask colours in the model's segmentation-legend format
(``<p>hair<color>(0, 0, 204)</color></p>``) made it copy the colour map back
instead of drawing a face, so the presets name regions without colours.

``--consistency`` re-segments every generated face with the binary method from
batch_celebamask_segment.py and scores it against the input mask (round-trip
mIoU). Treat it with care: that segmenter draws face parts even where there is
no face (a featureless generated silhouette scored 0.18, above two real faces).

The model always runs in BF16 with a 30 GiB GPU budget (whole model on a 32 GB
GPU, no CPU offload).

Outputs (resumable; existing gen/<id>.png are skipped):
    inputs/<id>.png       colour-coded input mask
    gen/<id>.png          generated face (512x512)
    vis/<id>.jpg          mask | generated | original photo
    reseg/<id>.png        label map of the generated face (with --consistency)
    log_shard<k>.jsonl    prompt, seconds, round-trip mIoU (with --consistency)
    consistency_shard<k>.json  round-trip metrics for the images generated in that run (with --consistency)
"""

import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageFilter

from inference.benchmark.batch_celebamask_segment import (
    CLASSES,
    PRESENCE_QUESTIONS,
    SPLIT_RANGES,
    load_split,
    merge_masks,
    segment_binary,
)
from inference.sensenova_vision import BASE_PARAMS, SenseNovaVisionModel
from tools.evaluation.segment.evaluate_celebamask import CLASS_NAMES, gt_label_map, scores
from tools.evaluation.segment.visualize_celebamask import PALETTE

DTYPE = "bf16"
MAX_MEM_PER_GPU = "30GiB"
SIZE = 512
FACE_QUESTION = "Is this a realistic photograph of a human face? Answer yes or no."


# Generic instructions: identical for every image, nothing about the person in it.
# Plain-language colour names for the CelebAMask-HQ palette used to draw the input masks.
COLOUR_NAMES = {
    1: "red", 2: "green", 3: "olive yellow", 4: "blue", 5: "magenta", 6: "cyan", 7: "light pink", 8: "brown",
    9: "bright red", 10: "lime green", 11: "yellow", 12: "navy blue", 13: "dark blue", 14: "pink", 15: "teal",
    16: "dark green", 17: "orange", 18: "bright green",
}


def colour_key():
    """Fixed key for all 18 classes (identical for every image), in plain words plus RGB values."""
    parts = [f"{phrase} {'are' if phrase in ('eyeglasses', 'clothes') else 'is'} {COLOUR_NAMES[cid]} ({', '.join(str(int(v)) for v in PALETTE[cid])})"
             for cid, _, phrase in CLASSES]
    return "Colour key: " + "; ".join(parts) + "; black (0, 0, 0) is the background."


_CELEBA = (
    "Transform this semantic face segmentation map into a realistic human portrait photograph. Strictly "
    "preserve the position, shape, and boundaries of every segmented facial region, including the hair, skin, "
    "eyes, eyebrows, nose, lips, and ears. Generate natural skin texture, realistic hair strands, detailed "
    "facial features, and coherent photographic lighting. The result should resemble an authentic, "
    "unretouched CelebA face photograph. Do not change the facial structure or retain any segmentation colors."
)

# Generic instructions: identical for every image, nothing about the person in it.
PROMPTS = {
    "celeba": _CELEBA + " " + colour_key(),
    "celeba_nokey": _CELEBA,
    "sketch": (
        "Turn this flat colour-block painting of a face into a realistic photograph of a real person. Keep the "
        "exact position, size and outline of the head, hair, face, eyes, eyebrows, nose, lips, ears, neck and "
        "clothing. Add natural skin texture, individual hair strands, detailed eyes and soft studio lighting, like "
        "an authentic CelebA portrait photo."
    ),
    "convert": (
        "Convert this face segmentation map into a realistic photograph of a real person. Keep the same face "
        "shape, pose and the position of the hair, eyes, eyebrows, nose, lips, ears and neck, but replace every "
        "flat colour with natural photographic texture: skin, hair, eyes and clothing. Do not output a "
        "segmentation map."
    ),
    "short": "Turn this face segmentation map into a photorealistic portrait photo.",
    "minimal": "Generate a realistic photo that matches this segmentation mask.",
    "explain": (
        "This image is a semantic segmentation map of a human face: each flat colour marks one region, such as "
        "hair, skin, eyes, eyebrows, nose, lips, ears, neck or clothing, and black is the background. Generate the "
        "photorealistic portrait photograph that this map was made from, with exactly the same layout."
    ),
    "studio": (
        "Render this segmentation map as a high-resolution studio portrait photograph of a real person, with "
        "natural lighting and sharp focus. Keep every region in the same place and with the same shape."
    ),
}


# Naturalistic colour per class (index = class id), fixed for every image: turns the label map into a flat
# colour-block "painting" instead of an arbitrary palette, so edit mode refines it rather than copying a mask.
NATURAL_COLOURS = np.array([
    [150, 150, 150],  # background: neutral grey
    [224, 172, 140],  # skin
    [208, 152, 120],  # nose
    [40, 40, 45],     # eyeglasses
    [70, 55, 50],     # left eye
    [70, 55, 50],     # right eye
    [85, 60, 45],     # left eyebrow
    [85, 60, 45],     # right eyebrow
    [216, 160, 130],  # left ear
    [216, 160, 130],  # right ear
    [110, 40, 40],    # inner mouth
    [186, 96, 96],    # upper lip
    [196, 104, 104],  # lower lip
    [95, 70, 50],     # hair
    [60, 60, 90],     # hat
    [212, 175, 55],   # earring
    [200, 180, 90],   # necklace
    [206, 150, 120],  # neck
    [70, 80, 110],    # clothes
], dtype=np.uint8)


def render_input(label, style):
    """Image given to the model: the CelebAMask-HQ palette mask, or a softened naturalistic colour-block sketch."""
    if style == "palette":
        return Image.fromarray(PALETTE[label])
    return Image.fromarray(NATURAL_COLOURS[label]).filter(ImageFilter.GaussianBlur(2))


# Dynamic presets: a fixed base followed by the names of the regions present in that image's map.
DYNAMIC_PROMPTS = {
    "celeb_dynamic": (
        "Convert this facial segmentation map into a photorealistic CelebA-style face portrait. Strictly preserve "
        "the geometry and boundaries of all facial regions while generating natural skin textures, realistic hair, "
        "detailed facial features, and soft photographic lighting. Do not alter the facial structure or retain "
        "segmentation colors."
    ),
    "celeb_dynamic_short": (
        "Convert this facial segmentation map into a photorealistic CelebA-style face photo, keeping every region's "
        "shape and position and no segmentation colours."
    ),
}
# How each dynamic preset introduces the region list.
REGION_LEAD = {"celeb_dynamic": "The segmentation map contains these regions: ", "celeb_dynamic_short": "Regions: "}


def regions_sentence(label, lead="The segmentation map contains these regions: "):
    present = set(np.unique(label).tolist())
    names = [phrase for cid, _, phrase in CLASSES if cid in present]
    return lead + ", ".join(names) + "."


def make_prompt(preset, label, prompt_text=None):
    if prompt_text:
        return prompt_text
    if preset in DYNAMIC_PROMPTS:
        return DYNAMIC_PROMPTS[preset] + " " + regions_sentence(label, REGION_LEAD[preset])
    return PROMPTS[preset]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_root", default="/datasets/CelebAMask-HQ/CelebAMask-HQ")
    parser.add_argument("--split", default="val", choices=list(SPLIT_RANGES))
    parser.add_argument("--output_dir", default="results/celebamask_hq/mask2face_val")
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "sensenova/SenseNova-Vision-7B-MoT"))
    parser.add_argument("--prompt", default="celeba", choices=list(PROMPTS) + list(DYNAMIC_PROMPTS),
                        help="Generic instruction preset (see PROMPTS).")
    parser.add_argument("--prompt_text", default=None, help="Custom generic instruction; overrides --prompt.")
    parser.add_argument("--num_timesteps", type=int, default=None, help="Diffusion steps (model default: 50).")
    parser.add_argument("--cfg_text_scale", type=float, default=None, help="Text guidance (edit default: 4.0).")
    parser.add_argument("--cfg_img_scale", type=float, default=None,
                        help="Image (mask) guidance (edit default: 2.0). In the pilot 1.0 and 2.0 gave similar layout fidelity.")
    parser.add_argument("--input_style", default="palette", choices=["palette", "natural"],
                        help="palette: CelebAMask-HQ colour-coded mask. natural: fixed naturalistic colour per "
                             "class with softened edges (a colour-block sketch).")
    parser.add_argument("--consistency", action="store_true",
                        help="Re-segment each generated face (binary method, 20 steps) and report round-trip mIoU.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ids = load_split(args.dataset_root, args.split)[: args.limit][args.shard_id :: args.num_shards]
    dirs = {name: os.path.join(args.output_dir, name) for name in ("inputs", "gen", "vis")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    todo = [i for i in ids if not os.path.exists(os.path.join(dirs["gen"], f"{i}.png"))]
    print(f"[mask2face] split={args.split} images={len(ids)} remaining={len(todo)} dtype={DTYPE} "
          f"max_mem_per_gpu={MAX_MEM_PER_GPU} consistency={args.consistency}", flush=True)
    with open(os.path.join(args.output_dir, f"config_shard{args.shard_id}.json"), "w") as f:
        json.dump({**vars(args), "dtype": DTYPE, "max_mem_per_gpu": MAX_MEM_PER_GPU}, f, indent=2)
    if not todo:
        return

    print(f"[mask2face] prompt preset: {'custom' if args.prompt_text else args.prompt}", flush=True)
    model = SenseNovaVisionModel(model_path=args.model_path, dtype=DTYPE, max_mem_per_gpu=MAX_MEM_PER_GPU)
    params = dict(BASE_PARAMS["edit"])
    for key in ("num_timesteps", "cfg_text_scale", "cfg_img_scale"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    seg_params = dict(BASE_PARAMS["dense_perception"], num_timesteps=20)

    n_cls = len(CLASS_NAMES)
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    face_flags = []
    log_path = os.path.join(args.output_dir, f"log_shard{args.shard_id}.jsonl")
    start = time.time()
    for n, image_id in enumerate(todo, 1):
        t0 = time.time()
        label = gt_label_map(args.dataset_root, image_id)
        prompt = make_prompt(args.prompt, label, args.prompt_text)
        mask_image = render_input(label, args.input_style)
        mask_image.save(os.path.join(dirs["inputs"], f"{image_id}.png"))

        face = model.generate(
            contents=[{"type": "image", "value": mask_image}, {"type": "text", "value": prompt}],
            mode="edit",
            noise_seed=args.seed + image_id,
            **params,
        ).convert("RGB")
        if face.size != (SIZE, SIZE):
            face = face.resize((SIZE, SIZE), Image.BICUBIC)
        face.save(os.path.join(dirs["gen"], f"{image_id}.png"))
        gen_seconds = time.time() - t0

        original = Image.open(os.path.join(args.dataset_root, "CelebA-HQ-img", f"{image_id}.jpg")).convert("RGB")
        panel = Image.new("RGB", (3 * SIZE, SIZE))
        for i, tile in enumerate((mask_image, face, original.resize((SIZE, SIZE), Image.BICUBIC))):
            panel.paste(tile, (i * SIZE, 0))
        panel.save(os.path.join(dirs["vis"], f"{image_id}.jpg"), quality=90)

        record = {"id": image_id, "seconds_generate": round(gen_seconds, 2), "prompt": prompt}
        if args.consistency:
            # The binary segmenter draws face parts even where there is no face, so a non-face output
            # (e.g. a dark silhouette) is scored as all background instead of being segmented.
            is_face = model.generate(
                contents=[{"type": "image", "value": face}, {"type": "text", "value": FACE_QUESTION}],
                mode="understanding",
            ).strip().lower().startswith("yes")
            extra = {"presence": {}}
            if is_face:
                masks, extra = segment_binary(model, face, CLASSES, seg_params, args.seed + image_id, size=SIZE,
                                              gate_classes=set(PRESENCE_QUESTIONS))
                pred = merge_masks(masks, "area", SIZE)
            else:
                pred = np.zeros((SIZE, SIZE), dtype=np.uint8)
            record["is_face"] = is_face
            face_flags.append(is_face)
            os.makedirs(os.path.join(args.output_dir, "reseg"), exist_ok=True)
            Image.fromarray(pred).save(os.path.join(args.output_dir, "reseg", f"{image_id}.png"))
            img_conf = np.bincount(label.astype(np.int64).ravel() * n_cls + pred.ravel(),
                                   minlength=n_cls * n_cls).reshape(n_cls, n_cls)
            conf += img_conf
            iou, _ = scores(img_conf)
            record["roundtrip_miou"] = float(np.nanmean(iou[1:]))
            record["presence"] = extra["presence"]
        record["seconds_total"] = round(time.time() - t0, 2)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        eta_h = (time.time() - start) / n * (len(todo) - n) / 3600
        msg = f"[mask2face] {n}/{len(todo)} id={image_id} gen={gen_seconds:.1f}s"
        if args.consistency:
            msg += f" roundtrip_mIoU={record['roundtrip_miou']:.3f}"
        print(msg + f" eta={eta_h:.1f}h", flush=True)

    if args.consistency:
        iou, f1 = scores(conf)
        summary = {
            "num_images": len(todo),
            "roundtrip_mIoU_face_classes": float(np.nanmean(iou[1:])),
            "face_rate": float(np.mean(face_flags)) if face_flags else None,
            "roundtrip_mean_F1_face_classes": float(np.nanmean(f1[1:])),
            "pixel_accuracy": float(np.diag(conf).sum() / conf.sum()),
            "per_class_IoU": {name: float(iou[i]) for i, name in enumerate(CLASS_NAMES)},
            "note": "Computed over the images generated in this invocation.",
        }
        with open(os.path.join(args.output_dir, f"consistency_shard{args.shard_id}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps({k: v for k, v in summary.items() if k != "per_class_IoU"}, indent=2))


if __name__ == "__main__":
    main()
