# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""COCO-Stuff (https://github.com/nightrome/cocostuff) for SenseNova-Vision: classes, palette, prompts and an
on-the-fly training dataset with the same four modes as data/celebamask.py.

Data (no extraction needed; zips or extracted directories both work):
    images  train2017.zip / val2017.zip (COCO 2017)            -> <split>/<12-digit id>.jpg
    labels  stuffthingmaps_trainval2017.zip (COCO-Stuff)        -> <split>/<12-digit id>.png
            pixel value v = class id - 1 (ids 1-182 from labels.txt), 255 = unlabeled

Classes: the 171 COCO-Stuff classes that occur (80 things + 91 stuff); the 11 unused thing ids
(street sign, hat, shoe, eye glasses, plate, mirror, window, desk, door, blender, hair brush) are
excluded. Internally labels are class ids with 0 = unlabeled.

Palette: COCO-Stuff has no official colour list, so one is generated as in the SenseNova-Vision
paper's segmentation data: greedy farthest-point sampling in the RGB cube (deterministic), with black
reserved for unlabeled pixels and white kept free for binary masks. ``write_class_table`` exports it.

Modes (one fixed prompt or template each; see the PROMPT constants below):
    img2palette          photo -> full map, every class in its colour, unlabeled black (fixed prompt, no colour key)
    img2palette_partial  photo -> map of the requested classes only (mostly classes present in the image,
                         plus a few absent distractors); everything else black
    img2binseg           photo -> binary mask of one class (present with probability binary_present_prob)
    seg2img              full colour map -> photo (colour key of the classes present in the input map)
    img2palette_legend   photo -> colour legend (text, CE loss) + map in those colours: the released model's
                         own generic-segmentation format (see LEGEND_PROMPT)
"""

import io
import json
import os
import zipfile

import numpy as np
from PIL import Image

from .prompts import CAPTION_QUESTION_LIST, MASK_QUESTION_LIST
from .seg_multitask import SegMultiTaskDataset

UNUSED_IDS = (12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91)
# labels.txt of nightrome/cocostuff (0 = unlabeled, 1-182 classes), embedded so the module is self-contained.
_LABELS = (
    "unlabeled person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant "
    "street_sign stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe hat "
    "backpack umbrella shoe eye_glasses handbag tie suitcase frisbee skis snowboard sports_ball kite "
    "baseball_bat baseball_glove skateboard surfboard tennis_racket bottle plate wine_glass cup fork knife "
    "spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant "
    "bed mirror dining_table window desk toilet door tv laptop mouse remote keyboard cell_phone microwave oven "
    "toaster sink refrigerator blender book clock vase scissors teddy_bear hair_drier toothbrush hair_brush "
    "banner blanket branch bridge building-other bush cabinet cage cardboard carpet ceiling-other ceiling-tile "
    "cloth clothes clouds counter cupboard curtain desk-stuff dirt door-stuff fence floor-marble floor-other "
    "floor-stone floor-tile floor-wood flower fog food-other fruit furniture-other grass gravel ground-other "
    "hill house leaves light mat metal mirror-stuff moss mountain mud napkin net paper pavement pillow "
    "plant-other plastic platform playingfield railing railroad river road rock roof rug salad sand sea shelf "
    "sky-other skyscraper snow solid-other stairs stone straw structural-other table tent textile-other towel "
    "tree vegetable wall-brick wall-concrete wall-other wall-panel wall-stone wall-tile wall-wood water-other "
    "waterdrops window-blind window-other wood"
).split()
CLASS_NAMES = [n.replace("_", " ") for n in _LABELS]  # index = class id (0 = unlabeled)
assert len(CLASS_NAMES) == 183
USED_IDS = [i for i in range(1, 183) if i not in UNUSED_IDS]
THING_IDS = [i for i in USED_IDS if i <= 91]


def phrase(cid):
    """Readable name: 'sky-other' -> 'sky (other)', 'floor-wood' -> 'floor wood'."""
    name = CLASS_NAMES[cid]
    if name.endswith("-other"):
        return name[: -len("-other")] + " (other)"
    return name.replace("-stuff", " (stuff)").replace("-", " ")


def _farthest_point_palette(n, levels=16):
    grid = np.linspace(0, 255, levels).round().astype(np.int32)
    cand = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    chosen = [np.array([0, 0, 0]), np.array([255, 255, 255])]  # reserved: unlabeled / binary foreground
    dist = np.minimum(((cand - chosen[0]) ** 2).sum(1), ((cand - chosen[1]) ** 2).sum(1))
    out = []
    for _ in range(n):
        idx = int(dist.argmax())
        out.append(cand[idx].copy())
        dist = np.minimum(dist, ((cand - cand[idx]) ** 2).sum(1))
    return np.array(out, dtype=np.uint8)


PALETTE = np.zeros((256, 3), dtype=np.uint8)  # index = class id; 0 (unlabeled) and unused ids stay black
PALETTE[USED_IDS] = _farthest_point_palette(len(USED_IDS))


def _rgb(cid):
    return "(" + ", ".join(str(int(v)) for v in PALETTE[cid]) + ")"


def colour_key(class_ids):
    return "; ".join(f"{phrase(c)} is {_rgb(c)}" for c in sorted(class_ids))


# ----------------------------------------------------------------------------- prompts
# img2palette and seg2img start with an explicit task header that states the direction (input -> output), and
# follow the same colour-key rule as data/celebamask.py:
#   img2palette  key for every class: the answer format, not the answer (~2,800 tokens for 171 classes; the
#                palette is generated here, so a model cannot know it without the key);
#   seg2img      key for the classes present in the input map only (derivable from the input itself).
IMG2PALETTE_PROMPT = (
    "Task: photo to segmentation map. The input is a photograph. Output its COCO-Stuff semantic segmentation "
    "map, not a photograph. Colour every pixel with exactly one of these colours: "
    f"{colour_key(USED_IDS)}; pixels that belong to none of these classes are black (0, 0, 0). Use flat "
    "colours only, with no shading or texture."
)
PARTIAL_PROMPT_TEMPLATE = (
    "Segment only these COCO-Stuff classes in this photograph: {names}. Colour them with exactly these colours: "
    "{key}. Colour every other pixel black (0, 0, 0), including all other classes and unlabeled regions. Use flat "
    "colours only, with no shading or texture."
)
BINARY_PROMPT_TEMPLATE = (
    "Segment every {phrase} (COCO-Stuff class {cid}, \"{name}\") in this photograph. Output a binary mask with "
    "all {phrase} pixels in white (255, 255, 255) and everything else in black (0, 0, 0). If the image has no "
    "{phrase}, output an all-black mask."
)
SEG2IMG_PROMPT_TEMPLATE = (
    "Task: segmentation map to photo. The input is a COCO-Stuff semantic segmentation map. Output a realistic "
    "photograph, not a segmentation map. The colours in this map mean: {key}; black (0, 0, 0) marks unlabeled "
    "regions. Strictly preserve the position, shape, and boundaries of every region. Generate natural textures, "
    "materials, and lighting, like an authentic COCO photograph. Do not retain any segmentation colours."
)


def seg2img_prompt(label):
    """SEG2IMG prompt for one input map: the colour key lists the classes present in that map."""
    return SEG2IMG_PROMPT_TEMPLATE.format(key=colour_key([int(c) for c in np.unique(label) if c]) or "no classes")


# img2palette_legend: the released model's generic-segmentation format (paper Sec. 3; data/prompts.py, trained
# there through SegJSONLIterableDataset with think_mode). The prompt lists every class as <p>name</p>; the answer
# is a colour legend <p>name<color>(R, G, B)</color></p> of the classes present (text, with CE loss) followed by
# a mask in those colours. As in the base model's own (greedy) answers on COCO, colours follow the legend
# position, not the class: the largest class comes first and is white, then a fixed sequence (the palette in
# its farthest-point order, so the most-used early positions are as far apart as possible). The mask's colours
# are therefore defined by the legend text, and the legend itself is fully predictable from the image (no
# randomly drawn colours, whose digits the CE loss could never learn).
def legend_name(cid):
    """Class name as the base model writes it in a legend: 'sky-other' -> 'sky', 'wall-brick' -> 'wall brick'."""
    return CLASS_NAMES[cid].replace("-other", "").replace("-stuff", "").replace("-", " ")


LEGEND_PROMPT = (
    MASK_QUESTION_LIST[0].format(categories=", ".join(f"<p>{legend_name(c)}</p>" for c in USED_IDS),
                                 task_type="semantic")
    + " " + CAPTION_QUESTION_LIST[0].format(task_type="semantic")
)


LEGEND_COLOURS = [(255, 255, 255)] + [tuple(int(v) for v in PALETTE[c]) for c in USED_IDS]  # farthest-point order


def legend_task(label):
    """(None, LEGEND_PROMPT, mask in legend colours, {"answer": legend text, "classes": ids in legend order})."""
    ids, counts = np.unique(label[label > 0], return_counts=True)
    order = [int(c) for c in ids[np.argsort(-counts, kind="stable")]]
    colours = LEGEND_COLOURS[: len(order)]
    lut = np.zeros((256, 3), dtype=np.uint8)  # unlabeled stays black
    for cid, rgb in zip(order, colours):
        lut[cid] = rgb
    legend = ", ".join(f"<p>{legend_name(c)}<color>({r}, {g}, {b})</color></p>" for c, (r, g, b) in zip(order, colours))
    return None, LEGEND_PROMPT, Image.fromarray(lut[label]), {"answer": legend, "classes": order}


def partial_prompt(class_ids):
    ids = sorted(class_ids)
    return PARTIAL_PROMPT_TEMPLATE.format(names=", ".join(phrase(c) for c in ids), key=colour_key(ids))


def binary_prompt(cid):
    return BINARY_PROMPT_TEMPLATE.format(phrase=phrase(cid), cid=cid, name=CLASS_NAMES[cid])


def palette_image(label, class_ids=None):
    if class_ids is not None:
        label = np.where(np.isin(label, list(class_ids)), label, 0)
    return Image.fromarray(PALETTE[label])


def binary_image(label, cid):
    return Image.fromarray(np.where(label == cid, 255, 0).astype(np.uint8)).convert("RGB")


def sample_task(rng, label, mode, partial_present_range=(1, 8), partial_absent_max=2, binary_present_prob=0.8):
    """One training example for `mode`: (source_or_None, prompt, target_or_None, info)."""
    present = [int(c) for c in np.unique(label) if c]
    absent = [c for c in USED_IDS if c not in set(present)]
    if mode == "img2palette":
        return None, IMG2PALETTE_PROMPT, palette_image(label), {}
    if mode == "img2palette_partial":
        k_present = rng.randint(min(partial_present_range[0], len(present)), min(partial_present_range[1],
                                                                                  len(present)))
        k_absent = rng.randint(0 if k_present else 1, partial_absent_max)
        subset = sorted(rng.sample(present, k_present) + rng.sample(absent, k_absent))
        return None, partial_prompt(subset), palette_image(label, subset), {"classes": subset}
    if mode == "img2binseg":
        pool = present if present and (rng.random() < binary_present_prob) else absent
        cid = rng.choice(pool)
        return None, binary_prompt(cid), binary_image(label, cid), {"class": cid, "present": cid in present}
    if mode == "seg2img":
        return palette_image(label), seg2img_prompt(label), None, {"classes": present}
    if mode == "img2palette_legend":
        return legend_task(label)
    raise ValueError(f"unknown mode {mode!r}")


def write_class_table(path):
    """Export id / name / phrase / RGB / thing-or-stuff for the 171 used classes."""
    rows = [{"id": c, "name": CLASS_NAMES[c], "phrase": phrase(c), "rgb": PALETTE[c].tolist(),
             "kind": "thing" if c in THING_IDS else "stuff"} for c in USED_IDS]
    with open(path, "w") as f:
        json.dump({"unlabeled": {"id": 0, "rgb": [0, 0, 0]}, "classes": rows}, f, indent=1)


# ----------------------------------------------------------------------------- data access
class _Source:
    """Read <split>/<name> from a zip archive or a directory; zip handles are opened lazily per process."""

    def __init__(self, path):
        self.path = path
        self.is_zip = path.endswith(".zip")
        self._zip, self._pid = None, None

    def _archive(self):
        if self._zip is None or self._pid != os.getpid():
            self._zip, self._pid = zipfile.ZipFile(self.path), os.getpid()
        return self._zip

    def names(self, split, suffix):
        if self.is_zip:
            names = sorted(os.path.basename(n)[: -len(suffix)] for n in self._archive().namelist()
                           if n.startswith(f"{split}/") and n.endswith(suffix))
            self._zip.close()  # each data-loader worker reopens its own handle
            self._zip = None
            return names
        return sorted(f[: -len(suffix)] for f in os.listdir(os.path.join(self.path, split)) if f.endswith(suffix))

    def open(self, split, name):
        if self.is_zip:
            return Image.open(io.BytesIO(self._archive().read(f"{split}/{name}")))
        return Image.open(os.path.join(self.path, split, name))


def load_label(raw):
    """stuffthingmaps PNG (v = id - 1, 255 = unlabeled) -> class-id array with 0 = unlabeled."""
    v = np.asarray(raw, dtype=np.uint8)
    return np.where(v == 255, 0, v.astype(np.uint16) + 1).astype(np.uint8)


class CocoStuffMultiTaskDataset(SegMultiTaskDataset):
    """Training-time generator for COCO-Stuff (registered as ``cocostuff_multitask``).

    data_dir entries (data/dataset_info.py or inline ``datasets:`` in the YAML) are dicts:
        {images: <train2017.zip | dir>, labels: <stuffthingmaps_trainval2017.zip | dir>, split: train2017}
    YAML group arguments (besides image_transform_args / vit_image_transform_args):
        mode_weights: {img2palette: 0.2, img2palette_partial: 0.3, img2binseg: 0.3, seg2img: 0.2}
        partial_present_range: [1, 8]   present classes per partial request
        partial_absent_max: 2           absent distractor classes per partial request (0..max)
        binary_present_prob: 0.8
        shuffle_seed: 0
    """

    MODES = SegMultiTaskDataset.MODES + ("img2palette_legend",)
    DEFAULT_MODE_WEIGHTS = {"img2palette": 0.2, "img2palette_partial": 0.3, "img2binseg": 0.3, "seg2img": 0.2}

    def __init__(self, dataset_name, transform, tokenizer, vit_transform, data_dir_list, num_used_data,
                 local_rank=0, world_size=1, num_workers=8, data_status=None, mode_weights=None,
                 partial_present_range=(1, 8), partial_absent_max=2, binary_present_prob=0.8, shuffle_lines=True,
                 shuffle_seed=0, task_prompt=None):
        super().__init__(dataset_name, transform, tokenizer, vit_transform, local_rank, world_size, num_workers,
                         data_status, mode_weights, shuffle_seed)
        self.partial_present_range = tuple(partial_present_range)
        self.partial_absent_max = partial_absent_max
        self.binary_present_prob = binary_present_prob
        self.sources = []
        self.data_paths = []
        for idx, (cfg, n) in enumerate(zip(data_dir_list, num_used_data)):
            images, labels = _Source(cfg["images"]), _Source(cfg["labels"])
            self.sources.append((images, labels, cfg["split"]))
            names = labels.names(cfg["split"], ".png")[:n]
            self.data_paths.extend((name, str(idx)) for name in names)
        self.set_epoch()

    def load(self, key):
        name, idx = key[0], int(key[1])
        images, labels, split = self.sources[idx]
        photo = images.open(split, f"{name}.jpg").convert("RGB")
        return photo, load_label(labels.open(split, f"{name}.png"))

    def draw(self, rng, label, mode):
        return sample_task(rng, label, mode, self.partial_present_range, self.partial_absent_max,
                           self.binary_present_prob)
