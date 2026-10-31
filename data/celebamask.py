# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""CelebAMask-HQ face parsing for SenseNova-Vision: definitions, prompts and an on-the-fly training dataset.

Single source of truth for the CelebAMask-HQ fine-tuning modes. The static converter
(tools/data_prepare/segmentation/celebamask_hq_sft.py) and the training-time dataset below use
the same class list, palette and prompts, so pre-generated and on-the-fly data are identical.

Modes (each with one fixed, explicit prompt or template):

    img2palette          photo -> full segmentation map in the official CelebAMask-HQ palette
    img2palette_partial  photo -> map of a requested subset of classes (official colours),
                         every other pixel black; the subset is the only variable in the prompt
    img2binseg           photo -> binary mask (white on black) of one class; absent classes -> all black
    seg2img              official-palette segmentation map -> photo
    img2palette_legend   photo -> colour legend (text, CE loss) + map in those colours: the released model's
                         own generic-segmentation format (see LEGEND_PROMPT)

CelebAMaskMultiTaskDataset draws a mode per sample (``mode_weights``), and for the
partial / binary modes a random subset / class, so every epoch sees new combinations.
It reads the dataset directly (CelebA-HQ-img + CelebAMask-HQ-mask-anno); nothing has
to be pre-generated.
"""

import os

import numpy as np
from PIL import Image

from .cocostuff import LEGEND_COLOURS
from .prompts import CAPTION_QUESTION_LIST, MASK_QUESTION_LIST
from .seg_multitask import SegMultiTaskDataset

# ----------------------------------------------------------------------------- definitions
# Official CelebAMask-HQ class order (ids 0-18) and colour list (face_parsing utils).
CLASS_NAMES = ["background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear",
               "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth"]
PALETTE = np.array([[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204],
                    [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
                    [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51],
                    [0, 204, 0]], dtype=np.uint8)
PHRASES = ["background", "face skin", "nose", "eyeglasses", "left eye", "right eye", "left eyebrow",
           "right eyebrow", "left ear", "right ear", "inner mouth", "upper lip", "lower lip", "hair", "hat",
           "earring", "necklace", "neck", "clothes"]
COLOUR_NAMES = ["black", "red", "green", "olive yellow", "blue", "magenta", "cyan", "light pink", "brown",
                "bright red", "lime green", "yellow", "navy blue", "dark blue", "pink", "teal", "dark green",
                "orange", "bright green"]
FACE_CLASS_IDS = list(range(1, 19))
SIZE = 512
# CelebA identities 162771-182637 (1-based) are the validation split (CelebAMask-HQ g_partition.py rule).
SPLIT_RANGES = {"train": (0, 162771), "val": (162771, 182638), "test": (182638, 10**9)}
# Left/right classes refer to the subject's own side, which appears on the opposite side of the image.
SIDE = {4: "left", 5: "right", 6: "left", 7: "right", 8: "left", 9: "right"}


def _rgb(cid):
    return "(" + ", ".join(str(int(v)) for v in PALETTE[cid]) + ")"


def colour_key(class_ids=FACE_CLASS_IDS):
    """'face skin is red (204, 0, 0); ...' for the given classes, in class-id order."""
    return "; ".join(f"{PHRASES[c]} {'are' if PHRASES[c] in ('eyeglasses', 'clothes') else 'is'} "
                     f"{COLOUR_NAMES[c]} {_rgb(c)}" for c in sorted(class_ids))


# ----------------------------------------------------------------------------- prompts
# img2palette and seg2img share the colour key, so each starts with an explicit task header that states the
# direction (input -> output) before anything else. Colour-key rule (as in data/cocostuff.py): img2palette lists
# every class (the answer format, not the answer); seg2img lists the classes present in the input map.
IMG2PALETTE_PROMPT = (
    "Task: face photo to segmentation map. The input is a face photograph. Output its CelebAMask-HQ semantic "
    "face segmentation map, not a photograph. Colour every pixel with exactly one of these colours: "
    f"{colour_key()}; background is black (0, 0, 0). Use flat colours only, with no shading or texture."
)

PARTIAL_PROMPT_TEMPLATE = (
    "Segment only these CelebAMask-HQ classes in this face photograph: {names}. Colour them with exactly these "
    "colours: {key}. Colour every other pixel black (0, 0, 0), including all other face parts and the "
    "background. Use flat colours only, with no shading or texture."
)

BINARY_PROMPT_TEMPLATE = (
    "Segment the {phrase} (CelebAMask-HQ class {cid}, \"{suffix}\") in this face photograph.{note} Output a "
    "binary mask with the {phrase} in white (255, 255, 255) and everything else in black (0, 0, 0). If the image "
    "has no {phrase}, output an all-black mask."
)

SEG2IMG_PROMPT_TEMPLATE = (
    "Task: segmentation map to face photo. The input is a CelebAMask-HQ semantic face segmentation map. Output a "
    "realistic human portrait photograph, not a segmentation map. The colours in the map mean: "
    "{key}; background is black (0, 0, 0). Strictly preserve the position, shape, and "
    "boundaries of every region. Generate natural skin texture, realistic hair strands, detailed facial "
    "features, and coherent photographic lighting, like an authentic, unretouched CelebA face photograph. Do "
    "not retain any segmentation colours."
)


# img2palette_legend: the released model's own generic-segmentation prompt with the 18 face classes, trained as
# in data/cocostuff.py (see the comment there): the answer is a colour legend <p>class<color>(R, G, B)</color></p>
# of the face classes present, largest first, followed by a mask in those colours. Colours belong to the legend
# position (the same fixed sequence as COCO-Stuff), so the mask can only be read through the legend. Background
# is not a listed class and stays black. Also the zero-shot baseline in the model's native format.
LEGEND_PROMPT = (
    MASK_QUESTION_LIST[0].format(categories=", ".join(f"<p>{PHRASES[c]}</p>" for c in FACE_CLASS_IDS),
                                 task_type="semantic")
    + " " + CAPTION_QUESTION_LIST[0].format(task_type="semantic")
)


def legend_task(label):
    """(None, LEGEND_PROMPT, mask in legend colours, {"answer": legend text, "classes": ids in legend order})."""
    ids, counts = np.unique(label[label > 0], return_counts=True)
    order = [int(c) for c in ids[np.argsort(-counts, kind="stable")]]
    colours = LEGEND_COLOURS[: len(order)]
    lut = np.zeros((256, 3), dtype=np.uint8)  # background stays black
    for cid, rgb in zip(order, colours):
        lut[cid] = rgb
    legend = ", ".join(f"<p>{PHRASES[c]}<color>({r}, {g}, {b})</color></p>" for c, (r, g, b) in zip(order, colours))
    return None, LEGEND_PROMPT, Image.fromarray(lut[label]), {"answer": legend, "classes": order}


def seg2img_prompt(label):
    """SEG2IMG prompt for one input map: the colour key lists the face classes present in that map."""
    present = [c for c in FACE_CLASS_IDS if (label == c).any()]
    return SEG2IMG_PROMPT_TEMPLATE.format(key=colour_key(present) or "no face classes")


def partial_prompt(class_ids):
    ids = sorted(class_ids)
    return PARTIAL_PROMPT_TEMPLATE.format(names=", ".join(PHRASES[c] for c in ids), key=colour_key(ids))


def binary_prompt(cid):
    note = ""
    if cid in SIDE:
        image_side = "right" if SIDE[cid] == "left" else "left"
        note = f" This is the person's own {SIDE[cid]}, which appears on the {image_side} side of the image."
    return BINARY_PROMPT_TEMPLATE.format(phrase=PHRASES[cid], cid=cid, suffix=CLASS_NAMES[cid], note=note)


# ----------------------------------------------------------------------------- data access
def load_split_ids(dataset_root, split):
    lo, hi = SPLIT_RANGES[split]
    ids = []
    with open(os.path.join(dataset_root, "CelebA-HQ-to-CelebA-mapping.txt")) as f:
        next(f)
        for line in f:
            idx, orig_idx, _ = line.split()
            if lo <= int(orig_idx) < hi:
                ids.append(int(idx))
    return sorted(ids)


def load_label_map(dataset_root, image_id):
    """Label map (512x512, ids 0-18) painted from the per-part PNGs in class order, as in CelebAMask-HQ g_mask.py."""
    anno_dir = os.path.join(dataset_root, "CelebAMask-HQ-mask-anno", str(image_id // 2000))
    label = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for cid in FACE_CLASS_IDS:
        path = os.path.join(anno_dir, f"{image_id:05d}_{CLASS_NAMES[cid]}.png")
        if os.path.exists(path):
            label[np.asarray(Image.open(path).convert("L")) > 127] = cid
    return label


def load_photo(dataset_root, image_id):
    return Image.open(os.path.join(dataset_root, "CelebA-HQ-img", f"{image_id}.jpg")).convert("RGB") \
        .resize((SIZE, SIZE), Image.BICUBIC)


def palette_image(label, class_ids=None):
    """Official-palette rendering; with class_ids, every other class is drawn black."""
    if class_ids is not None:
        label = np.where(np.isin(label, list(class_ids)), label, 0)
    return Image.fromarray(PALETTE[label])


def binary_image(label, cid):
    return Image.fromarray(np.where(label == cid, 255, 0).astype(np.uint8)).convert("RGB")


def sample_task(rng, label, mode, partial_size_range=(1, 17), binary_present_prob=0.8):
    """Draw one training example for `mode`: returns (input_image, prompt, target_image, info)."""
    present = [c for c in FACE_CLASS_IDS if (label == c).any()]
    if mode == "img2palette":
        return None, IMG2PALETTE_PROMPT, palette_image(label), {}
    if mode == "img2palette_partial":
        k = rng.randint(partial_size_range[0], partial_size_range[1])
        subset = sorted(rng.sample(FACE_CLASS_IDS, k))
        return None, partial_prompt(subset), palette_image(label, subset), {"classes": subset}
    if mode == "img2binseg":
        absent = [c for c in FACE_CLASS_IDS if c not in present]
        pool = present if (rng.random() < binary_present_prob or not absent) and present else absent
        cid = rng.choice(pool)
        return None, binary_prompt(cid), binary_image(label, cid), {"class": cid, "present": cid in present}
    if mode == "seg2img":
        return palette_image(label), seg2img_prompt(label), None, {"classes": present}
    if mode == "img2palette_legend":
        return legend_task(label)
    raise ValueError(f"unknown mode {mode!r}")


# ----------------------------------------------------------------------------- training dataset
class CelebAMaskMultiTaskDataset(SegMultiTaskDataset):
    """Training-time generator for all CelebAMask-HQ modes (registered as ``celebamask_multitask``).

    YAML group arguments (besides the usual image_transform_args / vit_image_transform_args):
        split: train | val | test                     (default train)
        mode_weights: {img2palette: 0.2, img2palette_partial: 0.2, img2binseg: 0.3, seg2img: 0.3}
        partial_size_range: [1, 17]                   classes per partial request (any subset of the 18)
        binary_present_prob: 0.8                      img2binseg: share of requests for a present class
        shuffle_seed: 0
    """

    MODES = SegMultiTaskDataset.MODES + ("img2palette_legend",)

    def __init__(self, dataset_name, transform, tokenizer, vit_transform, data_dir_list, num_used_data,
                 local_rank=0, world_size=1, num_workers=8, data_status=None, split="train", mode_weights=None,
                 partial_size_range=(1, 17), binary_present_prob=0.8, shuffle_lines=True, shuffle_seed=0,
                 task_prompt=None):
        super().__init__(dataset_name, transform, tokenizer, vit_transform, local_rank, world_size, num_workers,
                         data_status, mode_weights, shuffle_seed)
        self.split = split
        self.partial_size_range = tuple(partial_size_range)
        self.binary_present_prob = binary_present_prob
        # data_paths are (image_id, dataset_root) string pairs so the base class can shard and shuffle them.
        self.data_paths = []
        for root, n in zip(data_dir_list, num_used_data):
            self.data_paths.extend((str(i), root) for i in load_split_ids(root, split)[:n])
        self.set_epoch()

    def load(self, key):
        image_id, root = int(key[0]), key[1]
        return load_photo(root, image_id), load_label_map(root, image_id)

    def draw(self, rng, label, mode):
        return sample_task(rng, label, mode, self.partial_size_range, self.binary_present_prob)
