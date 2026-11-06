# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Segmentation decoder: VAE latent to per-pixel class (or legend slot) logits.

The VAE is trained on natural images and blurs flat colour maps, so the predicted map latent is read directly into
classes instead (train/train_seg_decoder.py). Three upsampling stages take the latent back to pixel resolution.

Sizes S, M and L are small convolutional decoders trained from scratch. Size "vae" is the VAE's own decoder with its
RGB output layer replaced by one for the classes; init_from_vae copies every other weight from the VAE, so training
starts from the upsampler that already renders the map's fine detail.
"""

import torch
from torch import nn

SIZES = {"S": dict(width=128, blocks=2), "M": dict(width=256, blocks=3), "L": dict(width=384, blocks=4), "vae": None}
# The VAE of modeling/autoencoder.py (load_ae): latents are stored as scale * (z - shift).
VAE_ARGS = dict(resolution=256, in_channels=3, ch=128, ch_mult=[1, 2, 4, 4], num_res_blocks=2, z_channels=16)
VAE_SCALE, VAE_SHIFT = 0.3611, 0.1159


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
        self.config = dict(in_channels=in_channels, num_classes=num_classes, size=size)
        if size == "vae":
            from modeling.autoencoder import Decoder

            if in_channels != VAE_ARGS["z_channels"]:
                raise ValueError(f"size vae reads {VAE_ARGS['z_channels']}-channel VAE latents, not {in_channels}")
            self.net = Decoder(out_ch=num_classes, **VAE_ARGS)
            return
        width, blocks = SIZES[size]["width"], SIZES[size]["blocks"]
        channels = [width, width // 2, width // 4]
        layers = [nn.Conv2d(in_channels, channels[0], 3, padding=1)]
        for stage, ch in enumerate(channels):
            layers.extend(ResBlock(ch) for _ in range(blocks))
            next_ch = channels[min(stage + 1, len(channels) - 1)]
            layers.extend([nn.Upsample(scale_factor=2.0, mode="nearest"), nn.Conv2d(ch, next_ch, 3, padding=1)])
        layers.extend([nn.GroupNorm(8, channels[-1]), nn.SiLU(), nn.Conv2d(channels[-1], num_classes, 1)])
        self.net = nn.Sequential(*layers)

    def forward(self, latent):
        if self.config["size"] == "vae":  # undo the latent scaling, as AutoEncoder.decode does
            latent = latent / VAE_SCALE + VAE_SHIFT
        return self.net(latent)


def init_from_vae(decoder, ae_path):
    """Copy the VAE decoder's weights (ae.safetensors) into a size-"vae" decoder, except the output layer, whose
    shape differs (classes instead of RGB) and keeps its fresh initialisation. Returns the number of tensors copied."""
    from safetensors.torch import load_file

    assert decoder.config["size"] == "vae"
    weights = {k[len("decoder."):]: v for k, v in load_file(ae_path).items()
               if k.startswith("decoder.") and not k.startswith("decoder.conv_out.")}
    missing, unexpected = decoder.net.load_state_dict(weights, strict=False)
    assert not unexpected and sorted(missing) == ["conv_out.bias", "conv_out.weight"], (missing, unexpected)
    return len(weights)


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
