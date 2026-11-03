# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""iREPA representation alignment for the symmetric flow (docs/SYMMETRIC.md).

REPA aligns intermediate hidden states of the generator with the patch features of a frozen vision encoder through a
projection head and a patch-wise cosine loss. iREPA (arXiv 2512.10794, github.com/end2end-diffusion/irepa) changes two
things, both kept here:

    projection     a 3x3 convolution on the token grid instead of REPA's MLP
    normalisation  spatial z-score of the encoder's patch features: (f - alpha * mean(f)) / (std(f) + 1e-6), mean and
                   std over the patches of one image, alpha 0.6

For the symmetric flow, the latent tokens of both streams are aligned with the encoder's features of the clean photo at
the same grid position, so that every map token is anchored to what the photo shows there. The encoder sees the photo
at patch_size pixels per latent token, so its patch grid is the token grid. Only the head trains (with the adapter);
it is not needed at inference.
"""

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def spatial_zscore(feat, alpha=0.6, eps=1e-6):
    """iREPA spatial normalisation of one image's patch features (patches, dim)."""
    return (feat - alpha * feat.mean(dim=0, keepdim=True)) / (feat.std(dim=0, keepdim=True) + eps)


class DinoTeacher(nn.Module):
    """Frozen DINOv2 (Hugging Face): layer-normed patch tokens of an image on a given patch grid."""

    def __init__(self, name="facebook/dinov2-base"):
        super().__init__()
        from transformers import Dinov2Model

        self.model = Dinov2Model.from_pretrained(name, torch_dtype=torch.bfloat16).eval().requires_grad_(False)
        self.dim = self.model.config.hidden_size
        self.patch = self.model.config.patch_size
        self.skip = 1 + getattr(self.model.config, "num_register_tokens", 0)  # CLS and register tokens
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, image, grid):
        """(h * w, dim) features, row-major, of an image in [-1, 1] of shape (3, H, W) on an h x w patch grid."""
        h, w = grid
        x = (image[None].float() + 1) / 2
        x = F.interpolate(x, size=(h * self.patch, w * self.patch), mode="bicubic", align_corners=False,
                          antialias=True).clamp(0, 1)
        x = ((x - self.mean) / self.std).to(torch.bfloat16)
        return self.model(pixel_values=x).last_hidden_state[0, self.skip:].float()


class IREPA(nn.Module):
    """Projection head and alignment loss. teacher maps (image, (h, w)) to (h * w, teacher.dim) features."""

    def __init__(self, hidden_size, teacher, pixels_per_token, alpha=0.6):
        super().__init__()
        self.teacher = teacher
        self.alpha = alpha
        self.pixels_per_token = pixels_per_token
        self.head = nn.Conv2d(hidden_size, teacher.dim, kernel_size=3, padding=1)

    def forward(self, hidden, images, grids, stream_ids):
        """Alignment loss of one packed batch, and the mean cosine per stream for logging.

        hidden      (latent tokens, hidden_size): states of all latent tokens in packing order (Bagel irepa_hidden)
        images      (images, 3, H, W) padded VAE inputs in [-1, 1], in the same order
        grids       (h, w) token grid of each image (patchified_vae_latent_shapes)
        stream_ids  (latent tokens,) 1 = photo stream, 2 = map stream, 0 = no stream (packed_stream_ids)

        A map-stream image is aligned with the photo of its sample, the photo-stream image packed just before it.
        Images of other modes (stream 0) are not aligned.
        """
        cosines, by_stream = [], {1: [], 2: []}
        start, target = 0, None
        for index, (h, w) in enumerate(grids):
            count = h * w
            stream = int(stream_ids[start])
            if stream == 1:
                photo = images[index, :, : h * self.pixels_per_token, : w * self.pixels_per_token]
                target = spatial_zscore(self.teacher(photo, (h, w)), self.alpha)
            elif stream == 2 and target is None:
                raise ValueError("a map-stream image without a photo-stream image before it")
            if stream in (1, 2):
                tokens = hidden[start : start + count].reshape(h, w, -1).permute(2, 0, 1)[None]
                pred = self.head(tokens.to(self.head.weight.dtype))[0].permute(1, 2, 0).reshape(count, -1)
                cos = F.cosine_similarity(pred.float(), target, dim=-1)
                cosines.append(cos)
                by_stream[stream].append(cos.detach())
            if stream == 2:
                target = None
            start += count
        if not cosines:
            return hidden.sum() * 0, {}
        loss = 1 - torch.cat(cosines).mean()
        logs = {f"irepa_cos_{name}": float(torch.cat(by_stream[s]).mean())
                for s, name in ((1, "image"), (2, "mask")) if by_stream[s]}
        return loss, logs
