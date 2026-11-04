# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared base for training-time segmentation multi-task datasets (CelebAMask-HQ, COCO-Stuff).

A subclass provides
    load(key)                      -> (photo: PIL.Image, label: uint8/uint16 array of class ids, same size)
    draw(rng, label, mode)         -> (source_or_None, prompt, target_or_None, info)
and the base class handles mode sampling, sizing, the sequence plan, sharding and resume.

Modes:
    img2palette          photo -> full colour-coded segmentation map
    img2palette_partial  photo -> map of a requested subset of classes, every other pixel black
    img2binseg           photo -> binary mask (white on black) of one class
    seg2img              colour-coded segmentation map -> photo
    symm                 photo and map together, on one symmetric flow (below)
    symm_legend          symm with the map in the colour-legend format of img2palette_legend (below)

Every sample is: input image (VAE + ViT, no loss) -> prompt (no loss) -> target image (loss), the
same sequence as SegJSONLIterableDataset / EditJSONLIterableDataset. A mode whose draw() returns an
"answer" in info (e.g. COCO-Stuff's img2palette_legend colour legend) puts it as text with loss between
the prompt and the target image, as SegJSONLIterableDataset does in think_mode.

A symm sample is: photo ViT tokens (optional, no loss) -> direction-free prompt (no loss) -> one noise split
holding the photo's and the map's VAE tokens (both with loss). With one shared t, the photo has noise level t and
the map 1 - t, each with its own noise: t = 0 is a clean photo and a noise map (segmentation starts there), t = 1 a
noise photo and a clean map (synthesis starts there). See docs/SYMMETRIC.md.

A symm_legend sample adds the colour legend of img2palette_legend (the classes present, largest first, each with the
colour of its legend position) as text between the prompt and the split, and draws the map in those colours. The
legend is known from the map, so synthesis takes it as given; segmentation predicts it from the photo, which the
text can only see through the ViT prefix. It therefore has a loss only in samples with the ViT prefix.

Sizing: the repository's ImageTransform resizes bicubically, which would blend class colours at
mask boundaries. Each sample is therefore first brought to the exact size the transform will
choose (photo: bicubic, label: nearest) and the target maps are rendered at that size, so the
transform leaves them untouched.
"""

import random

import numpy as np
from PIL import Image

from .distributed_iterable_dataset import DistributedIterableDataset
from .edit_dataset_jsonl import EditJSONLIterableDataset
from .jsonl_dataset import JSONLIterableDataset

MODES = ("img2palette", "img2palette_partial", "img2binseg", "seg2img", "symm", "symm_legend")
SYMM_MODES = ("symm", "symm_legend")  # modes on the symmetric flow (timestep shift 1.0, stream identity)

# RoPE ids of a symm sample. The photo stream sits before the prefix and the map stream after it, so the two are as
# far apart as an input and its target in the other modes. Adjacent ids cost most of the warm start.
SYMM_ROPE_IMAGE = 0
SYMM_ROPE_VIT = 1
SYMM_ROPE_TEXT = 2


def uses_symm(dataset_meta):
    """True if a dataset YAML samples a symm mode (which needs --timestep_shift 1.0)."""
    return any(isinstance(group, dict) and any((group.get("mode_weights") or {}).get(m, 0) > 0 for m in SYMM_MODES)
               for group in dataset_meta.values())


def dequantize(image, rng, beta):
    """Add uniform noise in [-beta, beta] (0-255 scale) per pixel and channel to a flat-colour map."""
    if beta <= 0:
        return image
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(rng.getrandbits(64)).uniform(-beta, beta, pixels.shape)
    return Image.fromarray(np.clip(np.rint(pixels + noise), 0, 255).astype(np.uint8))


class SegMultiTaskDataset(DistributedIterableDataset):
    MODES = MODES
    DEFAULT_MODE_WEIGHTS = {"img2palette": 0.2, "img2palette_partial": 0.2, "img2binseg": 0.3, "seg2img": 0.3}

    # Reuse the repository's sequence-plan and bookkeeping helpers unchanged.
    new_sample = JSONLIterableDataset.new_sample
    _add_text = EditJSONLIterableDataset._add_text
    _add_image = EditJSONLIterableDataset._add_image
    get_resume_row = JSONLIterableDataset.get_resume_row
    log_resume = JSONLIterableDataset.log_resume
    log_bad_sample = JSONLIterableDataset.log_bad_sample

    def __init__(self, dataset_name, transform, tokenizer, vit_transform, local_rank=0, world_size=1,
                 num_workers=8, data_status=None, mode_weights=None, shuffle_seed=0, symm_beta=0.0,
                 symm_vit_prob=0.5, symm_endpoint_prob=0.2, symm_text_dropout=0.0, symm_cond_prob=0.0,
                 symm_cond_photo_share=0.5):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.vit_transform = vit_transform
        self.data_status = data_status
        self.shuffle_seed = shuffle_seed
        self.symm_beta = symm_beta          # dequantization of the map, 0-255 scale
        self.symm_vit_prob = symm_vit_prob  # share of symm samples with the photo's ViT tokens as a clean prefix
        self.symm_endpoint_prob = symm_endpoint_prob  # share of symm samples at t = 0 or 1 instead of uniform t
        # share of symm samples without the prompt and legend, so that inference can use text guidance
        self.symm_text_dropout = symm_text_dropout
        # share of symm samples that keep one stream clean while the other has its own noise level (conditional
        # slices, for sampling with the known stream held clean); the photo is the clean one in cond_photo_share
        self.symm_cond_prob = symm_cond_prob
        self.symm_cond_photo_share = symm_cond_photo_share
        weights = dict(mode_weights or self.DEFAULT_MODE_WEIGHTS)
        unknown = set(weights) - set(self.MODES)
        if unknown:
            raise ValueError(f"unknown modes {sorted(unknown)}; choose from {self.MODES}")
        self.modes = [m for m in self.MODES if weights.get(m, 0) > 0]
        self.mode_weights = [weights[m] for m in self.modes]

    # -- to implement ---------------------------------------------------------------------
    def load(self, key):
        raise NotImplementedError

    def draw(self, rng, label, mode):
        raise NotImplementedError

    # -- shared ---------------------------------------------------------------------------
    def fit(self, photo, label):
        """Resize photo (bicubic) and label (nearest) to the size the image transform will choose."""
        size = self.transform.resize_transform(photo).size
        if photo.size != size:
            photo = photo.resize(size, Image.BICUBIC)
        if label.shape[::-1] != size:
            label = np.asarray(Image.fromarray(label).resize(size, Image.NEAREST))
        return photo, label

    def make_example(self, rng, photo, label):
        """Draw a mode and its task; returns (mode, source, prompt, target, info) as PIL images + text."""
        photo, label = self.fit(photo, label)
        mode = rng.choices(self.modes, weights=self.mode_weights)[0]
        source, prompt, target, info = self.draw(rng, label, mode)
        if mode == "seg2img":
            target = photo
        else:
            source = photo
        return mode, source, prompt, target, info

    def build_symm_sample(self, rng, photo, prompt, mask, legend=None):
        """Prefix + one noise split with both streams; pack_sequence draws t and sets their noise levels.

        legend (symm_legend): colour-legend text after the prompt, with a loss only when the ViT prefix is present.
        With probability symm_text_dropout the prompt and legend are both left out: the unconditional branch that
        text guidance needs. (No draw when it is 0, so the default keeps every other draw unchanged.)"""
        sample = self.new_sample()
        use_vit = rng.random() < self.symm_vit_prob
        if use_vit:
            sample["rope_start"] = SYMM_ROPE_VIT
            sample = self._add_image(sample, photo, need_loss=False, need_vae=False, need_vit=True, enable_cfg=False)
        else:
            sample["rope_start"] = SYMM_ROPE_TEXT
        drop_text = self.symm_text_dropout > 0 and rng.random() < self.symm_text_dropout
        if not drop_text:
            sample = self._add_text(sample, prompt, need_loss=False, enable_cfg=False)
            if legend:
                sample = self._add_text(sample, legend, need_loss=use_vit, enable_cfg=False)
        sample = self._add_image(sample, photo, need_loss=True, need_vae=False, need_vit=False)
        sample["sequence_plan"][-1].update(split_start=True, split_end=False, symm_stream="image",
                                           rope_id=SYMM_ROPE_IMAGE, symm_endpoint_prob=self.symm_endpoint_prob,
                                           symm_cond_prob=self.symm_cond_prob,
                                           symm_cond_photo_share=self.symm_cond_photo_share)
        sample = self._add_image(sample, dequantize(mask, rng, self.symm_beta), need_loss=True, need_vae=False,
                                 need_vit=False)
        sample["sequence_plan"][-1].update(split_start=False, split_end=True, symm_stream="mask")
        return sample

    def build_sample(self, rng, photo, label):
        mode, source, prompt, target, info = self.make_example(rng, photo, label)
        if mode in SYMM_MODES:
            return self.build_symm_sample(rng, source, prompt, target, info.get("answer")), mode, prompt, info
        sample = self.new_sample()
        sample = self._add_image(sample, source, need_loss=False, need_vae=True, need_vit=True)
        sample = self._add_text(sample, prompt, need_loss=False)
        if info.get("answer"):
            sample = self._add_text(sample, info["answer"], need_loss=True)
        sample = self._add_image(sample, target, need_loss=True, need_vae=False, need_vit=False)
        return sample, mode, prompt, info

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        row_start_id = self.get_resume_row(worker_id)
        self.log_resume(worker_id, row_start_id, len(data_paths_per_worker))
        epoch = 0
        while True:
            # Distinct, reproducible task draws per rank / worker; advance each pass so epochs differ.
            rng = random.Random(f"{self.shuffle_seed}-{self.local_rank}-{worker_id}-{epoch}")
            for row_idx, key in enumerate(data_paths_per_worker[row_start_id:], start=row_start_id):
                try:
                    sample, mode, _, _ = self.build_sample(rng, *self.load(key))
                except Exception as exc:
                    self.log_bad_sample(worker_id, row_idx, exc)
                    continue
                # "mode" is informational (for logging); resume only uses the other three keys.
                sample["data_indexes"] = {"data_indexes": row_idx, "worker_id": worker_id,
                                          "dataset_name": self.dataset_name, "mode": mode}
                yield sample
            row_start_id = 0
            epoch += 1
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
