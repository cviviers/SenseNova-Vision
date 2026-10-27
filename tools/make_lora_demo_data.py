# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Generate the synthetic "glorp" dataset used by the single-GPU LoRA demo.

A *glorp* is a made-up object: a teal hexagon with a white dot in its centre.
Every image also contains distractors that share one property with it (teal
circles, hexagons of other colours, ...), so the model has to learn the full
concept rather than a colour or a shape. The base model has never seen the word,
which makes the effect of fine-tuning easy to measure on the held-out split.

Two training JSONLs are written in the repository's training formats:

* ``glorp_bbox.jsonl`` (``cv_detection_bbox`` / SftJSONLIterableDataset): text
  answers in the same ``<p>label</p><bbox>[x1, y1, x2, y2]</bbox>`` format the
  model emits for bbox detection,
* ``glorp_seg.jsonl`` (``seg_sft`` / SegJSONLIterableDataset): binary masks as
  image targets, using the same prompt as ``--task binary_seg``.

    python tools/make_lora_demo_data.py --output_dir datas/lora_demo
"""

import argparse
import json
import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter

GLORP = "glorp"
TEAL = (0, 150, 140)
DISTRACTOR_COLORS = [
    (220, 60, 50), (240, 180, 30), (60, 90, 210), (150, 70, 180),
    (90, 170, 60), (240, 120, 170), (120, 80, 40),
]
BBOX_PROMPT = (
    "Detect all instances of <p>{name}</p> in the image. Output the results as a structured "
    "text list with each detection including category and bounding box coordinates in <bbox> format."
)
SEG_PROMPT = (
    "Can you segment the image based on the following categories: <p>{name}</p>? "
    "Please output the binary segmentation masks."
)


def regular_polygon(cx, cy, r, sides, rotation):
    return [
        (cx + r * math.cos(rotation + 2 * math.pi * k / sides), cy + r * math.sin(rotation + 2 * math.pi * k / sides))
        for k in range(sides)
    ]


def star(cx, cy, r, rotation):
    pts = []
    for k in range(10):
        rr = r if k % 2 == 0 else r * 0.45
        a = rotation + math.pi * k / 5
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def background(size, rng):
    c1 = [rng.randint(170, 250) for _ in range(3)]
    c2 = [rng.randint(120, 220) for _ in range(3)]
    img = Image.new("RGB", (size, size))
    px = img.load()
    vertical = rng.random() < 0.5
    for y in range(size):
        for x in range(size):
            t = (y if vertical else x) / (size - 1)
            px[x, y] = tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(10, 25)):  # faint clutter
        x, y, s = rng.randint(0, size), rng.randint(0, size), rng.randint(5, 30)
        shade = tuple(min(255, max(0, v + rng.randint(-25, 25))) for v in c1)
        draw.line([(x, y), (x + rng.randint(-s, s), y + rng.randint(-s, s))], fill=shade, width=rng.randint(1, 3))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def place(existing, size, rng, r_min, r_max):
    for _ in range(200):
        r = rng.randint(r_min, r_max)
        cx, cy = rng.randint(r + 4, size - r - 4), rng.randint(r + 4, size - r - 4)
        if all(math.hypot(cx - ex, cy - ey) > r + er + 6 for ex, ey, er in existing):
            existing.append((cx, cy, r))
            return cx, cy, r
    return None


def draw_glorp(draw, mask_draw, cx, cy, r, rotation):
    poly = regular_polygon(cx, cy, r, 6, rotation)
    draw.polygon(poly, fill=TEAL, outline=(0, 90, 85))
    d = max(3, int(r * 0.22))
    draw.ellipse([cx - d, cy - d, cx + d, cy + d], fill=(255, 255, 255))
    mask_draw.polygon(poly, fill=255)
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def draw_distractor(draw, cx, cy, r, rotation, rng):
    kind = rng.choice(["teal_circle", "teal_square", "hexagon", "hexagon_dot", "circle", "triangle", "star", "square"])
    color = rng.choice(DISTRACTOR_COLORS)
    if kind == "teal_circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=TEAL)
    elif kind == "teal_square":
        draw.polygon(regular_polygon(cx, cy, r, 4, rotation), fill=TEAL)
    elif kind in ("hexagon", "hexagon_dot"):
        draw.polygon(regular_polygon(cx, cy, r, 6, rotation), fill=color)
        if kind == "hexagon_dot":
            d = max(3, int(r * 0.22))
            draw.ellipse([cx - d, cy - d, cx + d, cy + d], fill=(255, 255, 255))
    elif kind == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif kind == "triangle":
        draw.polygon(regular_polygon(cx, cy, r, 3, rotation), fill=color)
    elif kind == "star":
        draw.polygon(star(cx, cy, r, rotation), fill=color)
    else:
        draw.polygon(regular_polygon(cx, cy, r, 4, rotation), fill=color)


def make_sample(size, rng):
    img = background(size, rng)
    mask = Image.new("L", (size, size), 0)
    draw, mask_draw = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    placed, boxes = [], []
    n_glorp = rng.choices([1, 2, 3], weights=[5, 3, 2])[0]
    shapes = ["glorp"] * n_glorp + ["distractor"] * rng.randint(3, 6)
    rng.shuffle(shapes)
    for shape in shapes:
        spot = place(placed, size, rng, size // 22, size // 9)
        if spot is None:
            continue
        cx, cy, r = spot
        rotation = rng.uniform(0, math.pi)
        if shape == "glorp":
            boxes.append(draw_glorp(draw, mask_draw, cx, cy, r, rotation))
        else:
            draw_distractor(draw, cx, cy, r, rotation, rng)
    return img, mask, boxes


def bbox_answer(boxes, size):
    """Same layout the model emits: `<p>label</p><bbox>[x1, y1, x2, y2]</bbox>...` with
    coordinates normalised to [0, 1] (3 decimals) and a trailing period."""
    tags = "".join(
        "<bbox>[" + ", ".join(f"{v / size:.3f}" for v in box) + "]</bbox>"
        for box in sorted(boxes, key=lambda b: (b[1], b[0]))
    )
    return f"<p>{GLORP}</p>{tags}."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default="datas/lora_demo")
    parser.add_argument("--num_train", type=int, default=400)
    parser.add_argument("--num_test", type=int, default=24)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for split, count in (("train", args.num_train), ("test", args.num_test)):
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "masks"), exist_ok=True)
        with open(os.path.join(split_dir, "glorp_bbox.jsonl"), "w") as f_box, \
                open(os.path.join(split_dir, "glorp_seg.jsonl"), "w") as f_seg, \
                open(os.path.join(split_dir, "annotations.jsonl"), "w") as f_ann:
            for idx in range(count):
                img, mask, boxes = make_sample(args.size, rng)
                image_rel, mask_rel = f"images/{idx:05d}.png", f"masks/{idx:05d}.png"
                img.save(os.path.join(split_dir, image_rel))
                mask.convert("RGB").save(os.path.join(split_dir, mask_rel))
                f_box.write(json.dumps({
                    "image": image_rel,
                    "conversations": [
                        {"from": "human", "value": "<image>" + BBOX_PROMPT.format(name=GLORP)},
                        {"from": "gpt", "value": bbox_answer(boxes, args.size)},
                    ],
                }) + "\n")
                f_seg.write(json.dumps({
                    "image": image_rel,
                    "seg": mask_rel,
                    "conversations": [
                        {"from": "human", "value": "<image>" + SEG_PROMPT.format(name=GLORP)},
                        {"from": "gpt", "value": "<SEG>"},
                    ],
                }) + "\n")
                f_ann.write(json.dumps({"image": image_rel, "mask": mask_rel, "boxes": boxes}) + "\n")
        print(f"{split}: {count} samples -> {split_dir}")


if __name__ == "__main__":
    main()
