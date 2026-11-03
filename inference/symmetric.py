# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Segmentation and synthesis on a model trained with the symmetric flow (mode `symm`, docs/SYMMETRIC.md).

The photo and its colour-coded map share one query with one t: the photo has noise level t and the map 1 - t.
Segmentation starts at t = 0 from a clean photo and noise for the map, usually for a single step. Synthesis starts
at t = 1 from a clean map and noise for the photo. Only the text and the optional ViT tokens are cached: unlike the
other modes, the input image is part of the flow and changes at every step.

A model trained on mode symm_legend also takes the colour legend as text after the prompt: synthesis computes it from
the input map, and segmentation generates it from the ViT prefix first (generate_legend), as img2palette_legend does.
"""

import random

import torch

from data.seg_multitask import SYMM_ROPE_IMAGE, SYMM_ROPE_TEXT, SYMM_ROPE_VIT, dequantize
from inference.inferencer import move_generation_input_to_device


class SymmetricFlow:
    def __init__(self, vision_model, vae_transform=None, vit_transform=None):
        self.inferencer = vision_model.inferencer
        self.model = vision_model.model
        self.vae_model = vision_model.vae_model
        self.new_token_ids = vision_model.new_token_ids
        self.device = vision_model.device
        self.vae_transform = vae_transform or vision_model.vae_transform
        self.vit_transform = vit_transform or vision_model.vit_transform

    def encode(self, image):
        """VAE latent of a PIL image as packed tokens (num_tokens, patch_latent_dim), and its (H, W)."""
        tensor = self.vae_transform(image.convert("RGB"))
        H, W = tensor.shape[1:]
        tensor = tensor[None].to(self.device, self.inferencer.vae_dtype)
        latent = self.vae_model.encode(tensor)[0]
        p = self.model.latent_patch_size
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample
        latent = latent[:, : h * p, : w * p].reshape(self.model.latent_channel, h, p, w, p)
        latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.model.latent_channel)
        return latent.float(), (H, W)

    def _context(self, prompt, vit_image, legend=None):
        """Cached prefix: optional ViT tokens, then the prompt and legend (None: the unconditional branch of a model
        trained with symm_text_dropout)."""
        context = self.inferencer.init_gen_context()
        context["ropes"] = [SYMM_ROPE_TEXT if vit_image is None else SYMM_ROPE_VIT]
        if vit_image is not None:
            context = self.inferencer.update_context_image(vit_image, context, vae=False, vit=True,
                                                           vit_transform=self.vit_transform)
        if prompt is not None:
            context = self.inferencer.update_context_text(prompt, context)
        if legend is not None:
            context = self.inferencer.update_context_text(legend, context)
        return context

    @torch.no_grad()
    def generate_legend(self, photo, prompt, max_tokens=1024):
        """Colour legend for a photo (mode symm_legend): greedy text from the ViT prefix and the prompt."""
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            context = self._context(prompt, photo)
            return self.inferencer.gen_text(context, max_length=max_tokens, do_sample=False)

    def _query(self, context, shape, rope):
        """One query holding len(rope) images, as InterleaveInferencer.gen_image builds with num_output_vae."""
        query = self.model.prepare_vae_latent(
            curr_kvlens=context["kv_lens"] + [0] * (len(rope) - 1),
            curr_rope=rope,
            image_sizes=[shape] * len(rope),
            new_token_ids=self.new_token_ids,
        )
        query.pop("packed_init_noises")
        query = move_generation_input_to_device(query, self.device)
        query["packed_seqlens"] = query["packed_seqlens"].sum(0, keepdim=True)
        query["key_values_lens"] = query["key_values_lens"].sum(0, keepdim=True)
        return query

    @torch.no_grad()
    def generate_conditional_legend(self, photo, prompt, max_tokens=1024):
        """Colour legend of a model trained on img2palette_legend, as its caption_generate mode writes it: greedy
        text after the photo (VAE + ViT) and the prompt. For the one-step control with a slot decoder."""
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            context = self.inferencer.update_context_image(photo, self.inferencer.init_gen_context(), vae=True, vit=True,
                                                           vae_transform=self.vae_transform,
                                                           vit_transform=self.vit_transform)
            context = self.inferencer.update_context_text(prompt, context)
            return self.inferencer.gen_text(context, max_length=max_tokens, do_sample=False)

    def _run(self, known, direction, prompt, shape, seed, num_steps, vit_image, recompute_known, legend=None,
             cfg_scale=1.0):
        context = self._context(prompt, vit_image, legend)
        query = self._query(context, shape, [SYMM_ROPE_IMAGE, context["ropes"][0]])
        uncond = None
        if cfg_scale != 1.0:  # the same query without prompt and legend (models trained with symm_text_dropout)
            plain = self._context(None, vit_image)
            uncond = {**self._query(plain, shape, [SYMM_ROPE_IMAGE, plain["ropes"][0]]),
                      "past_key_values": plain["past_key_values"]}
        generator = torch.Generator(device="cpu").manual_seed(seed)
        created_noise = torch.randn(known.shape, generator=generator).to(self.device)
        known_noise = torch.randn(known.shape, generator=generator).to(self.device) if recompute_known else None
        return self.model.generate_symmetric(
            known_latent=known,
            created_noise=created_noise,
            direction=direction,
            num_steps=num_steps,
            known_noise=known_noise,
            cfg_scale=cfg_scale,
            uncond=uncond,
            past_key_values=context["past_key_values"],
            **query,
        )

    @torch.no_grad()
    def one_step_from_pair(self, photo, mask_latent, t, prompt, seed=0, vit_image=None, legend=None):
        """One-step map estimate from a photo and map noised to t, for training the segmentation decoder."""
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            photo_latent, shape = self.encode(photo)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            photo_noise = torch.randn(photo_latent.shape, generator=generator).to(self.device)
            mask_noise = torch.randn(mask_latent.shape, generator=generator).to(self.device)
            mask_t = t * mask_latent + (1 - t) * mask_noise
            pair = torch.cat([(1 - t) * photo_latent + t * photo_noise, mask_t], dim=0)
            num_tokens = photo_latent.shape[0]
            noise_level = torch.cat([torch.full((num_tokens,), t), torch.full((num_tokens,), 1 - t)]).to(self.device)
            stream_ids = torch.cat([torch.full((num_tokens,), 1), torch.full((num_tokens,), 2)]).to(self.device)
            context = self._context(prompt, vit_image, legend)
            velocity = self.model._forward_flow(
                x_t=pair, timestep=noise_level, past_key_values=context["past_key_values"], per_token_timesteps=True,
                packed_stream_ids=stream_ids,
                **self._query(context, shape, [SYMM_ROPE_IMAGE, context["ropes"][0]]),
            ).float()
            return mask_t - (1 - t) * velocity[num_tokens:]

    @torch.no_grad()
    def conditional_latent(self, photo, prompt, seed=0, noise_level=1.0, mask_latent=None, legend=None):
        """One-step map estimate of a model trained on the img2palette mode: clean photo in the cached prefix, map
        at noise_level (pure noise by default). The control for the symmetric flow, with no guidance. legend: the
        colour legend after the prompt, for a model trained on img2palette_legend."""
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            context = self.inferencer.update_context_image(photo, self.inferencer.init_gen_context(), vae=True, vit=True,
                                                           vae_transform=self.vae_transform,
                                                           vit_transform=self.vit_transform)
            context = self.inferencer.update_context_text(prompt, context)
            if legend is not None:
                context = self.inferencer.update_context_text(legend, context)
            shape = self.vae_transform(photo.convert("RGB")).shape[1:]
            shape = (int(shape[0]), int(shape[1]))
            num_tokens = (shape[0] // self.model.latent_downsample) * (shape[1] // self.model.latent_downsample)
            size = (num_tokens, self.model.patch_latent_dim)
            noise = torch.randn(size, generator=torch.Generator(device="cpu").manual_seed(seed)).to(self.device)
            state = noise if mask_latent is None else (1 - noise_level) * mask_latent + noise_level * noise
            timestep = torch.full((num_tokens,), float(noise_level), device=self.device)
            velocity = self.model._forward_flow(x_t=state, timestep=timestep, past_key_values=context["past_key_values"],
                                                **self._query(context, shape, [context["ropes"][0]])).float()
            return state - noise_level * velocity, shape

    @torch.no_grad()
    def segment_latent(self, photo, prompt, seed=0, num_steps=1, use_vit=True, recompute_known=False, legend=None,
                       cfg_scale=1.0):
        """Predicted map latent tokens for a photo, and the (H, W) they decode to. legend: symm_legend models;
        cfg_scale: text guidance (models trained with symm_text_dropout)."""
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            known, shape = self.encode(photo)
            latent = self._run(known, "segment", prompt, shape, seed, num_steps, photo if use_vit else None,
                               recompute_known, legend, cfg_scale)
        return latent, shape

    @torch.no_grad()
    def segment(self, photo, prompt, seed=0, num_steps=1, use_vit=True, recompute_known=False, legend=None,
                cfg_scale=1.0):
        """Colour-coded map as a PIL image, decoded by the VAE."""
        latent, shape = self.segment_latent(photo, prompt, seed, num_steps, use_vit, recompute_known, legend,
                                            cfg_scale)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            return self.inferencer.decode_image(latent, shape)

    @torch.no_grad()
    def synthesize(self, mask, prompt, seed=0, num_steps=25, beta=0.0, recompute_known=False, legend=None,
                   cfg_scale=1.0):
        """Photo for a colour-coded map, as a PIL image. beta dequantizes the map as in training. legend: the map's
        colour legend (symm_legend models; the map must then be drawn in the legend's colours)."""
        mask = dequantize(mask, random.Random(seed), beta)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            known, shape = self.encode(mask)
            latent = self._run(known, "synthesize", prompt, shape, seed, num_steps, None, recompute_known, legend,
                               cfg_scale)
            return self.inferencer.decode_image(latent, shape)
