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

Every sample is: input image (VAE + ViT, no loss) -> prompt (no loss) -> target image (loss), the
same sequence as SegJSONLIterableDataset / EditJSONLIterableDataset. A mode whose draw() returns an
"answer" in info (e.g. COCO-Stuff's img2palette_legend colour legend) puts it as text with loss between
the prompt and the target image, as SegJSONLIterableDataset does in think_mode.

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

MODES = ("img2palette", "img2palette_partial", "img2binseg", "seg2img")


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
                 num_workers=8, data_status=None, mode_weights=None, shuffle_seed=0):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.vit_transform = vit_transform
        self.data_status = data_status
        self.shuffle_seed = shuffle_seed
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

    def build_sample(self, rng, photo, label):
        mode, source, prompt, target, info = self.make_example(rng, photo, label)
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
