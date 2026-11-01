# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight segmentation decoder: VAE latent to per-pixel class logits.

The VAE is trained on natural images and blurs flat colour maps, so the predicted map latent is read directly into
classes instead (train/train_seg_decoder.py). Three upsampling stages take the latent back to pixel resolution.
"""

import torch
from torch import nn

SIZES = {"S": dict(width=128, blocks=2), "M": dict(width=256, blocks=3), "L": dict(width=384, blocks=4)}


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class SegDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, size="S"):
        super().__init__()
        width, blocks = SIZES[size]["width"], SIZES[size]["blocks"]
        self.config = dict(in_channels=in_channels, num_classes=num_classes, size=size)
        channels = [width, width // 2, width // 4]
        layers = [nn.Conv2d(in_channels, channels[0], 3, padding=1)]
        for stage, ch in enumerate(channels):
            layers.extend(ResBlock(ch) for _ in range(blocks))
            next_ch = channels[min(stage + 1, len(channels) - 1)]
            layers.extend([nn.Upsample(scale_factor=2.0, mode="nearest"), nn.Conv2d(ch, next_ch, 3, padding=1)])
        layers.extend([nn.GroupNorm(8, channels[-1]), nn.SiLU(), nn.Conv2d(channels[-1], num_classes, 1)])
        self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)


def unpack_latent(model, tokens, shape):
    """Packed latent tokens of one image to a (1, c, h, w) VAE latent; shape is the image's (H, W)."""
    p, c = model.latent_patch_size, model.latent_channel
    h, w = shape[0] // model.latent_downsample, shape[1] // model.latent_downsample
    latent = tokens.reshape(1, h, w, p, p, c)
    return torch.einsum("nhwpqc->nchpwq", latent).reshape(1, c, h * p, w * p)


def load_seg_decoder(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    decoder = SegDecoder(**checkpoint["config"])
    decoder.load_state_dict(checkpoint["decoder"])
    return decoder.to(device).eval()
