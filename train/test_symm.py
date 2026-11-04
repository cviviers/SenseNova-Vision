# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Checks of the symmetric flow on the 2-layer tiny model (tools/make_tiny_model.py); needs a GPU, a few seconds.

    python -m train.test_symm --model_path results/tiny_model
"""

import argparse
import math
import random
import types

import numpy as np
import torch
from PIL import Image

from data.data_utils import get_flattened_position_ids_extrapolate
from data.dataset_base import SYMM_CLEAN_LEVEL, SYMM_T_EPS, PackedDataset, sample_symm_logit_t
from data.seg_multitask import (
    SYMM_ROPE_IMAGE,
    SYMM_ROPE_TEXT,
    SYMM_ROPE_VIT,
    SegMultiTaskDataset,
    dequantize,
    uses_symm,
)
from data.transforms import ImageTransform
from inference.inferencer import InterleaveInferencer
from inference.symmetric import SymmetricFlow
from modeling.seg_decoder import SegDecoder, load_seg_decoder, unpack_latent
from train.finetune_lora import build_model
from train.irepa import IREPA, spatial_zscore
from train.train_seg_decoder import segmentation_loss
from train.train_utils import per_group_losses, per_stream_losses, summarize_groups, accumulate_groups, symm_row_weights

SIZE = 128
PROMPT = "Two images of the same face. The first is a photograph, the second its segmentation map."
LEGEND = "<p>face skin<color>(255, 255, 255)</color></p>, <p>hair<color>(0, 128, 255)</color></p>"
DEVICE = "cuda"


def make_packer(new_token_ids):
    return types.SimpleNamespace(
        **new_token_ids, use_flex=False, max_num_tokens=10**6,
        get_flattened_position_ids=get_flattened_position_ids_extrapolate,
        data_config=types.SimpleNamespace(
            text_cond_dropout_prob=0.0, vit_cond_dropout_prob=0.0, vae_cond_dropout_prob=0.0, vit_patch_size=14,
            max_num_patch_per_side=70, vae_image_downsample=16, max_latent_size=64))


def make_dataset(tokenizer, vit_prob, beta=10.0):
    dataset = object.__new__(SegMultiTaskDataset)
    dataset.transform = ImageTransform(SIZE, SIZE, 16)
    dataset.vit_transform = ImageTransform(SIZE // 14 * 14, 224 // 14, 14)
    dataset.tokenizer = tokenizer
    dataset.symm_beta = beta
    dataset.symm_vit_prob = vit_prob
    dataset.symm_endpoint_prob = 0.0
    dataset.symm_text_dropout = 0.0
    dataset.symm_cond_prob = 0.0
    dataset.symm_cond_photo_share = 0.5
    return dataset


def make_images():
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8))
    mask = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    mask[SIZE // 4:] = (204, 0, 0)
    return photo, Image.fromarray(mask)


def pack(packer, samples):
    status = PackedDataset.set_sequence_status(packer)
    for sample in samples:
        status = PackedDataset.pack_sequence(packer, sample, status)
    return PackedDataset.to_tensor(packer, status)


def check_sample(packer, tokenizer, vit_prob):
    """Sequence plan, attention mask, RoPE ids and noise levels of one packed sample."""
    photo, mask = make_images()
    dataset = make_dataset(tokenizer, vit_prob)
    sample = dataset.build_symm_sample(random.Random(0), photo, PROMPT, mask)
    has_vit = vit_prob == 1.0
    assert [i["type"] for i in sample["sequence_plan"]] == ["vit_image"] * has_vit + ["text", "vae_image", "vae_image"]
    data = pack(packer, [sample])
    num_tokens = (SIZE // 16) ** 2
    prefix = data["sequence_length"] - 2 * (num_tokens + 2)
    can_attend = data["nested_attention_masks"][0] == 0
    assert not can_attend[:prefix, prefix:].any(), "the prefix must not see the noise split"
    assert can_attend[prefix:, :prefix].all() and can_attend[prefix:, prefix:].all()

    rope = data["packed_position_ids"]
    text_len = len(tokenizer.encode(PROMPT)) + 2
    assert rope[0].item() == (SYMM_ROPE_VIT if has_vit else SYMM_ROPE_TEXT)
    assert (rope[prefix : prefix + num_tokens + 2] == SYMM_ROPE_IMAGE).all()
    assert (rope[prefix + num_tokens + 2 :] == SYMM_ROPE_TEXT + text_len).all(), "same distance with and without ViT"

    latent_pos = data["packed_latent_position_ids"]
    assert torch.equal(latent_pos[:num_tokens], latent_pos[num_tokens:]), "2D positions are shared"
    noise_level = torch.sigmoid(data["packed_timesteps"])
    assert noise_level[:num_tokens].unique().numel() == 1 and noise_level[num_tokens:].unique().numel() == 1
    assert abs(noise_level[0] + noise_level[-1] - 1) < 1e-5, "noise levels must be t and 1 - t"
    assert SYMM_T_EPS * 0.99 <= noise_level[0] <= 1 - SYMM_T_EPS * 0.99
    assert len(data["mse_loss_indexes"]) == 2 * num_tokens
    return data


def to_device(data):
    data = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in data.items()}
    data["nested_attention_masks"] = [m.to(DEVICE) for m in data["nested_attention_masks"]]
    return data


def check_training_step(model, packer, tokenizer):
    """Forward + backward on two packed samples of different layouts; noising, loss rows, per-stream logs."""
    photo, mask = make_images()
    samples = [make_dataset(tokenizer, p).build_symm_sample(random.Random(i), photo, PROMPT, mask)
               for i, p in enumerate((1.0, 0.0))]
    data = pack(packer, samples)
    num_tokens = (SIZE // 16) ** 2
    data = to_device(data)
    data.pop("ce_loss_weights", None)
    images = data.pop("padded_images")
    data["padded_latent"] = model.vae_for_test.encode(images.to(DEVICE, torch.bfloat16)).float()
    groups = ["symm", "symm"]
    model.train()
    for param in model.llm2vae.parameters():
        param.requires_grad_(True)
    assert "packed_stream_ids" in data
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**data)
    assert out["mse"].shape == (4 * num_tokens, model.patch_latent_dim)
    weights = symm_row_weights(data, groups, 0.5)
    assert weights.shape == (4 * num_tokens,) and (weights == 0.5).sum() == 2 * num_tokens
    (out["mse"].mean(-1) * weights).mean().backward()
    assert model.llm2vae.bias.grad is not None and model.llm2vae.bias.grad.abs().sum() > 0
    delta = model.symm_embedding.delta
    assert delta.grad is not None and delta.grad.abs().sum() > 0, "the identity must train"
    model.symm_embedding.zero_grad(set_to_none=True)

    window = {}
    accumulate_groups(window, per_group_losses(out, data, groups))
    streams = per_stream_losses(out, data, groups)
    assert sum(v["mse"][1] for v in streams.values()) == 4 * num_tokens
    assert {k.split("_")[0] for k in streams} == {"image", "mask"}
    total = sum(v["mse"][0] for v in streams.values()) / (4 * num_tokens)
    assert abs(total - summarize_groups(window)["symm"]["mse"]) < 1e-3
    model.llm2vae.zero_grad(set_to_none=True)
    for param in model.llm2vae.parameters():
        param.requires_grad_(False)


def make_flow(model, vae, tokenizer, new_token_ids):
    vae_transform = ImageTransform(SIZE, SIZE, 16)
    vit_transform = ImageTransform(SIZE // 14 * 14, 224 // 14, 14)
    inferencer = InterleaveInferencer(model, vae, tokenizer, vae_transform, vit_transform, new_token_ids, DEVICE)
    wrapper = types.SimpleNamespace(inferencer=inferencer, model=model, vae_model=vae, new_token_ids=new_token_ids,
                                    device=DEVICE, vae_transform=vae_transform, vit_transform=vit_transform)
    return SymmetricFlow(wrapper)


def check_parity(model, flow, packer, tokenizer, use_vit, legend=None, text_dropped=False, levels=(0.3, 0.7)):
    """The inference path (cached prefix + joint query) must match the training path on identical inputs. levels: the
    photo's and the map's noise levels (t and 1 - t on the diagonal; one clean stream in a conditional slice)."""
    photo, mask = make_images()
    dataset = make_dataset(tokenizer, 1.0 if use_vit else 0.0, beta=0.0)
    dataset.symm_text_dropout = 1.0 if text_dropped else 0.0
    sample = dataset.build_symm_sample(random.Random(1), photo, PROMPT, mask, legend)
    data = to_device(pack(packer, [sample]))
    data.pop("ce_loss_weights", None)
    images = data.pop("padded_images")
    logit = [math.log(level / (1 - level)) for level in levels]
    num_tokens = (SIZE // 16) ** 2
    data["packed_timesteps"] = torch.cat([torch.full((num_tokens,), logit[0]),
                                          torch.full((num_tokens,), logit[1])]).to(DEVICE)
    data["padded_latent"] = model.vae_for_test.encode(images.to(DEVICE, torch.bfloat16)).float()

    captured, randn_like = {}, torch.randn_like
    torch.randn_like = lambda x, *a, **k: captured.setdefault("noise", randn_like(x, *a, **k))

    def hook(module, inputs, output):
        captured["train"] = output.detach().float().clone()

    handle = model.llm2vae.register_forward_hook(hook)
    model.train()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(**data)
    handle.remove()
    torch.randn_like = randn_like

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        zx, shape = flow.encode(photo)
        zm, _ = flow.encode(mask)
        clean = torch.cat([zx, zm])
        noise_level = torch.cat([torch.full((num_tokens,), levels[0]), torch.full((num_tokens,), levels[1])]).to(DEVICE)
        noisy = (1 - noise_level[:, None]) * clean + noise_level[:, None] * captured["noise"].float()
        context = flow._context(None if text_dropped else PROMPT, photo if use_vit else None, legend)
        query = model.prepare_vae_latent(context["kv_lens"] + [0], [SYMM_ROPE_IMAGE, context["ropes"][0]], [shape, shape],
                                         flow.new_token_ids)
        query.pop("packed_init_noises")
        query = {k: v.to(DEVICE) for k, v in query.items()}
        query["packed_seqlens"] = query["packed_seqlens"].sum(0, keepdim=True)
        query["key_values_lens"] = query["key_values_lens"].sum(0, keepdim=True)
        inferred = model._forward_flow(x_t=noisy, timestep=noise_level, past_key_values=context["past_key_values"],
                                       per_token_timesteps=True, packed_stream_ids=data["packed_stream_ids"],
                                       **query).float()
    diff = ((captured["train"] - inferred).norm() / captured["train"].norm()).item()
    # BF16 kernels differ between the packed training pass and the cached inference pass. Measured on the tiny model:
    # 0.005-0.007 with a noisy map, up to 0.013 with a nearly clean one (diagonal (0.999, 0.001) and slices alike).
    # A real layout error (RoPE, mask, noise level) is far larger.
    assert diff < 0.02, f"train and inference disagree by {diff:.4f}"
    return diff


def check_sampler(model, flow):
    """One step is the one-step projection, the known stream can be recomputed, both directions run."""
    photo, mask = make_images()
    latent, _ = flow.segment_latent(photo, PROMPT, seed=3, num_steps=1, use_vit=False)
    generator = torch.Generator(device="cpu").manual_seed(3)
    noise = torch.randn(latent.shape, generator=generator).to(DEVICE)
    with torch.no_grad():
        velocity_scale = (noise - latent).norm().item()
    assert velocity_scale > 0, "the one-step projection must move the noise"
    for kwargs in (dict(num_steps=2), dict(num_steps=3, recompute_known=True)):
        assert flow.segment(photo, PROMPT, seed=0, use_vit=True, **kwargs).size == (SIZE, SIZE)
        assert flow.synthesize(mask, PROMPT, seed=0, beta=10, **kwargs).size == (SIZE, SIZE)
    try:
        model.generate_symmetric(known_latent=None, created_noise=None, direction="sideways", packed_text_ids=None,
                                 packed_text_indexes=None, packed_vae_position_ids=None, packed_vae_token_indexes=None,
                                 packed_seqlens=None, packed_position_ids=None, packed_indexes=None,
                                 past_key_values=None, key_values_lens=None, packed_key_value_indexes=None)
    except ValueError:
        return
    raise AssertionError("an unknown direction must raise")


def check_decoder(model, flow, tmp_path="/tmp/test_symm_decoder.pt"):
    """Latent unpacking, output resolution, loss and the checkpoint round trip of the segmentation decoder."""
    photo, _ = make_images()
    tokens, shape = flow.encode(photo)
    latent = unpack_latent(model, tokens, shape)
    p, c = model.latent_patch_size, model.latent_channel
    h, w = shape[0] // model.latent_downsample, shape[1] // model.latent_downsample
    packed = torch.einsum("chpwq->hwpqc", latent[0].reshape(c, h, p, w, p)).reshape(-1, p * p * c)
    assert torch.equal(packed, tokens), "unpacking must invert packing"  # the VAE samples: encode only once

    decoder = SegDecoder(model.latent_channel, 19, "S").to(DEVICE)
    logits = decoder(latent)
    assert logits.shape == (1, 19, SIZE, SIZE)
    label = torch.zeros(1, SIZE, SIZE, dtype=torch.long, device=DEVICE)
    label[:, SIZE // 2 :] = 3
    ce, dice = segmentation_loss(logits, label, ignore_unlabeled=False)
    assert ce.item() > 0 and 0 < dice.item() < 1
    (ce + dice).backward()
    assert decoder.net[0].weight.grad.abs().sum() > 0

    ce, dice = segmentation_loss(logits.detach(), torch.zeros_like(label), ignore_unlabeled=True)
    assert ce.item() == 0 and dice.item() == 0, "an image with no labelled pixel must not produce nan"
    target = torch.where(label == 3, label, torch.zeros_like(label))
    exact = torch.full_like(logits, -50.0)
    exact[:, 3] = 50.0
    ce, dice = segmentation_loss(exact, target, ignore_unlabeled=True)
    assert ce.item() < 1e-3 and dice.item() < 1e-3, "a perfect prediction on the labelled pixels must score near 0"

    torch.save({"decoder": decoder.state_dict(), "config": decoder.config}, tmp_path)
    reloaded = load_seg_decoder(tmp_path, DEVICE)
    with torch.no_grad():
        assert torch.equal(reloaded(latent), decoder.eval()(latent))


def check_stream_embedding(model, packer, tokenizer):
    """Stream ids reach the batch, identities differ by stream, legacy batches and disabled models get nothing."""
    photo, mask = make_images()
    data = pack(packer, [make_dataset(tokenizer, 0.0).build_symm_sample(random.Random(0), photo, PROMPT, mask)])
    num_tokens = (SIZE // 16) ** 2
    ids = data["packed_stream_ids"]
    assert ids.tolist() == [1] * num_tokens + [2] * num_tokens
    assert model._stream_embedding(ids) == 0, "disabled by default"
    dtype = next(model.parameters()).dtype
    model.enable_symm_embedding()
    assert next(model.parameters()).dtype == dtype, "the identity must not change the dtype the model reports"
    embedding = model._stream_embedding(ids.to(DEVICE))
    assert embedding.shape == (2 * num_tokens, model.hidden_size) and embedding.dtype == torch.float32
    assert torch.equal(embedding[0], -embedding[-1]), "c starts at zero, so the identities are +u and -u"
    assert (model._stream_embedding(torch.zeros_like(ids, device=DEVICE)) == 0).all(), "other tokens are untouched"
    state = {k: v.detach().clone() for k, v in model.symm_embedding.state_dict().items()}
    model.enable_symm_embedding(state)
    assert torch.equal(model.symm_embedding.delta, state["delta"])

    legacy = dict(text_ids_list=[[5, 6]], image_tensor_list=[torch.zeros(3, SIZE, SIZE)], num_tokens=0, sequence_plan=[
        dict(type="text", enable_cfg=0, loss=0, special_token_loss=0, special_token_label=None),
        dict(type="vae_image", enable_cfg=0, loss=1, special_token_loss=0, special_token_label=None)])
    assert "packed_stream_ids" not in pack(packer, [legacy])


def check_endpoints(packer, tokenizer):
    """The share of samples at the ends follows symm_endpoint_prob; the two ends are balanced."""
    photo, mask = make_images()
    dataset = make_dataset(tokenizer, 0.0, beta=0.0)
    dataset.symm_endpoint_prob = 0.5
    level = []
    for seed in range(300):
        data = pack(packer, [dataset.build_symm_sample(random.Random(seed), photo, PROMPT, mask)])
        level.append(torch.sigmoid(data["packed_timesteps"][0]).item())
    level = np.array(level)
    low, high = (level < 2 * SYMM_T_EPS).mean(), (level > 1 - 2 * SYMM_T_EPS).mean()
    assert 0.17 < low < 0.33 and 0.17 < high < 0.33, (low, high)


def check_cond_slices(packer, tokenizer):
    """symm_cond_prob: one stream clean and the other at its own level, the photo clean in cond_photo_share of them;
    with symm_cond_prob 0 the diagonal's random draws are unchanged."""
    photo, mask = make_images()
    num_tokens = (SIZE // 16) ** 2
    dataset = make_dataset(tokenizer, 0.0, beta=0.0)
    dataset.symm_cond_prob = 1.0
    photo_clean = 0
    for seed in range(300):
        data = pack(packer, [dataset.build_symm_sample(random.Random(seed), photo, PROMPT, mask)])
        level = torch.sigmoid(data["packed_timesteps"])
        image, mask_level = level[0].item(), level[-1].item()
        assert level[:num_tokens].unique().numel() == 1 and level[num_tokens:].unique().numel() == 1
        clean = abs(image - SYMM_T_EPS) < 1e-6, abs(mask_level - SYMM_T_EPS) < 1e-6  # the free one is uniform
        assert clean[0] != clean[1], "exactly one stream is held clean"
        photo_clean += clean[0]
    assert 0.38 < photo_clean / 300 < 0.62, photo_clean
    dataset.symm_cond_prob, dataset.symm_endpoint_prob = 0.0, 0.3
    for seed in range(20):
        np.random.seed(seed)
        data = pack(packer, [dataset.build_symm_sample(random.Random(seed), photo, PROMPT, mask)])
        np.random.seed(seed)
        assert abs(data["packed_timesteps"][0].item() - sample_symm_logit_t(0.3)) < 1e-5, "diagonal draws changed"


def check_clean_weights(packer, tokenizer):
    """symm_row_weights: photo rows at image_weight, rows of a clean stream at clean_weight; per_stream_losses logs
    clean rows apart from the t bins."""
    photo, mask = make_images()
    num_tokens = (SIZE // 16) ** 2
    dataset = make_dataset(tokenizer, 0.0, beta=0.0)
    data = pack(packer, [dataset.build_symm_sample(random.Random(i), photo, PROMPT, mask) for i in range(2)])
    logit = lambda level: math.log(level / (1 - level))
    levels = (0.3, 0.7, SYMM_T_EPS, 0.6)  # a diagonal sample, then a slice with the photo clean
    data["packed_timesteps"] = torch.cat([torch.full((num_tokens,), logit(v)) for v in levels])
    groups = ["symm", "symm"]
    assert symm_row_weights(data, groups, 1.0, 1.0) is None
    weights = symm_row_weights(data, groups, 0.5, 0.0)
    expected = torch.cat([torch.full((num_tokens,), w) for w in (0.5, 1.0, 0.0, 1.0)])
    assert torch.equal(weights.float(), expected), "photo 0.5, map 1, clean photo 0"
    assert torch.equal(symm_row_weights(data, groups, 0.5, 1.0).float(),
                       torch.cat([torch.full((num_tokens,), w) for w in (0.5, 1.0, 0.5, 1.0)]))
    streams = per_stream_losses({"mse": torch.ones(4 * num_tokens, 4)}, data, groups)
    assert streams["image_clean"]["mse"][1] == num_tokens and "mask_clean" not in streams
    assert streams["image_t0.2"]["mse"][1] == num_tokens and streams["mask_t0.2"]["mse"][1] == num_tokens
    assert streams["mask_t0.4"]["mse"][1] == num_tokens, "the slice's map at level 0.6 sits at t = 0.4"


def check_hold_known(flow):
    """hold_known keeps the known stream clean: one step equals the plain sampler (both start with a clean photo),
    more steps differ; it excludes known_noise; synthesis runs."""
    photo, mask = make_images()

    def run(**kwargs):  # the VAE samples its posterior: same torch seed for every call
        torch.manual_seed(0)
        return flow.segment_latent(photo, PROMPT, seed=4, use_vit=True, legend=LEGEND, **kwargs)[0]

    assert torch.equal(run(num_steps=1), run(num_steps=1, hold_known=True))
    assert not torch.allclose(run(num_steps=3), run(num_steps=3, hold_known=True))
    try:
        run(num_steps=2, hold_known=True, recompute_known=True)
    except ValueError:
        pass
    else:
        raise AssertionError("hold_known with recompute_known must raise")
    assert flow.synthesize(mask, PROMPT, seed=0, num_steps=2, legend=LEGEND, hold_known=True,
                           cfg_scale=2.0).size == (SIZE, SIZE)


def check_legend_sample(packer, tokenizer, vit_prob):
    """symm_legend: the legend is text between the prompt and the split, with a loss only with the ViT prefix; the
    map stream's RoPE id follows the legend; the prefix still cannot see the split."""
    photo, mask = make_images()
    sample = make_dataset(tokenizer, vit_prob).build_symm_sample(random.Random(0), photo, PROMPT, mask, LEGEND)
    has_vit = vit_prob == 1.0
    plan = sample["sequence_plan"]
    assert [i["type"] for i in plan] == ["vit_image"] * has_vit + ["text", "text", "vae_image", "vae_image"]
    assert plan[-4]["loss"] == 0 and plan[-3]["loss"] == int(has_vit), "legend loss only with the ViT prefix"
    data = pack(packer, [sample])
    legend_len = len(tokenizer.encode(LEGEND))
    num_ce = len(data.get("ce_loss_indexes", []))
    assert (num_ce > 0) == has_vit and (not has_vit or legend_len <= num_ce <= legend_len + 1), num_ce
    num_tokens = (SIZE // 16) ** 2
    prefix = data["sequence_length"] - 2 * (num_tokens + 2)
    can_attend = data["nested_attention_masks"][0] == 0
    assert not can_attend[:prefix, prefix:].any() and can_attend[prefix:, :prefix].all()
    rope = data["packed_position_ids"]
    text_len = len(tokenizer.encode(PROMPT)) + 2 + legend_len + 2
    assert (rope[prefix : prefix + num_tokens + 2] == SYMM_ROPE_IMAGE).all()
    assert (rope[prefix + num_tokens + 2 :] == SYMM_ROPE_TEXT + text_len).all()
    assert len(data["mse_loss_indexes"]) == 2 * num_tokens


def check_text_dropout(packer, tokenizer):
    """symm_text_dropout leaves out the prompt and the legend together; the map stream then follows the prefix."""
    photo, mask = make_images()
    for vit_prob in (0.0, 1.0):
        dataset = make_dataset(tokenizer, vit_prob)
        dataset.symm_text_dropout = 1.0
        sample = dataset.build_symm_sample(random.Random(0), photo, PROMPT, mask, LEGEND)
        assert [i["type"] for i in sample["sequence_plan"]] == ["vit_image"] * (vit_prob == 1.0) + ["vae_image"] * 2
        data = pack(packer, [sample])
        num_tokens = (SIZE // 16) ** 2
        rope = data["packed_position_ids"]
        assert (rope[-(num_tokens + 2):] == SYMM_ROPE_TEXT).all() and len(data.get("ce_loss_indexes", [])) == 0
    dataset.symm_text_dropout = 0.3
    kept = sum(len(dataset.build_symm_sample(random.Random(s), photo, PROMPT, mask, LEGEND)["sequence_plan"]) == 5
               for s in range(400)) / 400
    assert 0.6 < kept < 0.8, kept


def check_legend_training_step(model, packer, tokenizer):
    """A batch mixing symm and symm_legend samples: CE on the legend with ViT, per-stream rows for both modes."""
    photo, mask = make_images()
    samples = [make_dataset(tokenizer, 1.0).build_symm_sample(random.Random(0), photo, PROMPT, mask),
               make_dataset(tokenizer, 1.0).build_symm_sample(random.Random(1), photo, PROMPT, mask, LEGEND),
               make_dataset(tokenizer, 0.0).build_symm_sample(random.Random(2), photo, PROMPT, mask, LEGEND)]
    groups = ["symm", "symm_legend", "symm_legend"]
    data = to_device(pack(packer, samples))
    data.pop("ce_loss_weights", None)
    data["padded_latent"] = model.vae_for_test.encode(data.pop("padded_images").to(DEVICE, torch.bfloat16)).float()
    num_tokens = (SIZE // 16) ** 2
    model.train()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**data)
    assert out["ce"] is not None and out["ce"].numel() == len(data["ce_loss_indexes"]) and torch.isfinite(out["ce"]).all()
    weights = symm_row_weights(data, groups, 0.5)
    assert weights.shape == (6 * num_tokens,) and (weights == 0.5).sum() == 3 * num_tokens
    streams = per_stream_losses(out, data, groups)
    assert sum(v["mse"][1] for v in streams.values()) == 6 * num_tokens


def check_legend_dataset():
    """The real dataset modules draw symm_legend samples in the legend's colours, and the evaluation reads such a
    map back to the label map through the legend text."""
    from data import celebamask, cocostuff
    from inference.benchmark.seg_eval_common import CelebAMaskEval, CocoStuffEval, legend_presence, legend_to_label

    rng = np.random.default_rng(0)
    for module, ds, ids in ((celebamask, CelebAMaskEval("unused"), celebamask.FACE_CLASS_IDS),
                            (cocostuff, CocoStuffEval.__new__(CocoStuffEval), cocostuff.USED_IDS)):
        if module is cocostuff:
            ds.m = cocostuff
            ds.eval_ids = list(cocostuff.USED_IDS)
        label = np.zeros((64, 64), dtype=np.uint8)
        chosen = [int(c) for c in rng.choice(ids, 5, replace=False)]
        for i, cid in enumerate(chosen):  # stripes of decreasing area
            label[i * 12 : i * 12 + 12 - i] = cid
        source, prompt, legend_map, info = module.sample_task(random.Random(0), label, "symm_legend")
        assert source is None and prompt == module.SYMM_LEGEND_PROMPT and info["classes"] == chosen
        colours = {tuple(c) for c in np.asarray(legend_map).reshape(-1, 3)}
        assert colours <= set(module.LEGEND_COLOURS[: len(chosen)]) | {(0, 0, 0)}, "legend colours and black only"
        decoded, decoded_info = legend_to_label(ds, legend_map, info["answer"], legend_map.size)
        assert np.array_equal(decoded, label), f"{module.__name__}: the legend must decode its own map"
        assert decoded_info["legend_ids"] == chosen and not decoded_info["unmatched"]
        assert legend_presence(ds, decoded_info["legend_ids"], label) == (5, 0, 0)
        assert legend_presence(ds, decoded_info["legend_ids"][:3], label) == (3, 0, 2)


def check_legend_sampler(flow):
    """Legend generation from the ViT prefix, and both directions with a legend in the context."""
    photo, mask = make_images()
    text = flow.generate_legend(photo, PROMPT, max_tokens=8)
    assert isinstance(text, str)
    assert flow.segment(photo, PROMPT, seed=0, use_vit=True, legend=LEGEND).size == (SIZE, SIZE)
    assert flow.synthesize(mask, PROMPT, seed=0, num_steps=2, beta=10, legend=LEGEND).size == (SIZE, SIZE)


def check_guidance(flow):
    """Text guidance: scale 1 is the unguided path exactly; scale 0 follows the path without prompt and legend;
    other scales move the created stream; both directions run."""
    photo, mask = make_images()
    base = dict(seed=5, num_steps=2, use_vit=True)

    def run(prompt, **kwargs):  # the VAE samples its posterior: same torch seed for every call
        torch.manual_seed(0)
        return flow.segment_latent(photo, prompt, **base, **kwargs)[0]

    plain = run(PROMPT, legend=LEGEND)
    assert torch.equal(plain, run(PROMPT, legend=LEGEND, cfg_scale=1.0))
    no_text, zero = run(None), run(PROMPT, legend=LEGEND, cfg_scale=0.0)
    diff = ((zero - no_text).norm() / no_text.norm()).item()
    assert diff < 0.05, f"scale 0 must follow the text-free branch ({diff:.4f})"
    guided = run(PROMPT, legend=LEGEND, cfg_scale=3.0)
    assert not torch.allclose(guided, plain)
    assert flow.synthesize(mask, PROMPT, seed=0, num_steps=2, legend=LEGEND, cfg_scale=2.0).size == (SIZE, SIZE)


class StubTeacher(torch.nn.Module):
    """Offline stand-in for DINOv2: pooled pixels through a fixed projection; counts its calls."""
    dim = 8

    def __init__(self):
        super().__init__()
        self.register_buffer("proj", torch.randn(3, self.dim, generator=torch.Generator().manual_seed(0)))
        self.calls = 0

    def forward(self, image, grid):
        self.calls += 1
        pooled = torch.nn.functional.adaptive_avg_pool2d(image[None].float(), grid)[0]
        return (pooled.permute(1, 2, 0).reshape(-1, 3) @ self.proj.to(image.device)).float()


def check_irepa(model, packer, tokenizer):
    """Hidden states after layer k at the latent tokens, one teacher pass per photo shared by both streams, the
    spatial z-score, a finite loss, and gradients into the head and the layers below k."""
    feat = torch.randn(64, 8) * 3 + 5
    normed = spatial_zscore(feat, alpha=1.0)
    assert torch.allclose(normed.mean(0), torch.zeros(8), atol=1e-4) and torch.allclose(normed.std(0), torch.ones(8), atol=1e-3)
    assert torch.allclose(spatial_zscore(feat, alpha=0.6).std(0), torch.ones(8), atol=1e-3)

    photo, mask = make_images()
    samples = [make_dataset(tokenizer, 1.0).build_symm_sample(random.Random(0), photo, PROMPT, mask),
               make_dataset(tokenizer, 0.0).build_symm_sample(random.Random(1), photo, PROMPT, mask, LEGEND)]
    data = to_device(pack(packer, samples))
    data.pop("ce_loss_weights", None)
    images = data.pop("padded_images")
    data["padded_latent"] = model.vae_for_test.encode(images.to(DEVICE, torch.bfloat16)).float()
    layer = model.language_model.model.layers[0]
    seen = {}
    handle = layer.register_forward_hook(lambda module, inputs, output: seen.setdefault("out", output))
    weight = layer.self_attn.q_proj_moe_gen.weight  # latent tokens' queries: the generation expert
    weight.requires_grad_(True)
    teacher = StubTeacher().to(DEVICE)
    irepa = IREPA(model.hidden_size, teacher, model.latent_downsample).to(DEVICE)
    model.train()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**data, irepa_layer=1)
        num_tokens = (SIZE // 16) ** 2
        assert out["irepa_hidden"].shape == (4 * num_tokens, model.hidden_size)
        assert torch.equal(out["irepa_hidden"], seen["out"][data["packed_vae_token_indexes"]]), "layer-1 states"
        loss, logs = irepa(out["irepa_hidden"], images.to(DEVICE), data["patchified_vae_latent_shapes"],
                           data["packed_stream_ids"])
    handle.remove()
    assert teacher.calls == 2, "one teacher pass per photo, shared by its map stream"
    assert torch.isfinite(loss) and 0 <= loss.item() <= 2 and set(logs) == {"irepa_cos_image", "irepa_cos_mask"}
    loss.backward()
    assert irepa.head.weight.grad is not None and irepa.head.weight.grad.abs().sum() > 0, "head gradient"
    assert weight.grad is not None and weight.grad.abs().sum() > 0, "gradient into the layers below the capture"
    weight.grad = None
    weight.requires_grad_(False)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        assert "irepa_hidden" not in model(**data), "no capture unless asked"


def check_guards():
    assert uses_symm({"a": {"mode_weights": {"symm": 1.0}}})
    assert uses_symm({"a": {"mode_weights": {"symm_legend": 1.0, "seg2img": 0}}})
    assert not uses_symm({"a": {"mode_weights": {"symm": 0}}, "b": {"mode_weights": {"seg2img": 1}}, "c": {}})
    _, mask = make_images()
    assert dequantize(mask, random.Random(0), 0) is mask
    pixels, flat = np.asarray(dequantize(mask, random.Random(0), 10), np.int32), np.asarray(mask, np.int32)
    assert 0 < np.abs(pixels - flat).max() <= 10
    assert np.array_equal(pixels, np.asarray(dequantize(mask, random.Random(0), 10), np.int32)), "must be seeded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="results/tiny_model")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    model, vae, _, tokenizer, new_token_ids = build_model(args.model_path, 1.0)
    model = model.to(DEVICE).to(torch.bfloat16).requires_grad_(False)
    torch.nn.init.normal_(model.llm2vae.weight, std=0.02)  # released head stays; the tiny model's is all zeros
    vae = vae.to(DEVICE).to(torch.bfloat16).eval().requires_grad_(False)
    model.vae_for_test = vae
    packer = make_packer(new_token_ids)

    check_guards()
    print("guards ok")
    for vit_prob in (0.0, 1.0):
        check_sample(packer, tokenizer, vit_prob)
    print("packing ok: attention mask, RoPE ids, shared positions, noise levels t and 1 - t")
    check_endpoints(packer, tokenizer)
    print("endpoint sampling ok")
    check_cond_slices(packer, tokenizer)
    print("conditional slices ok: one clean stream, photo share, diagonal draws unchanged without them")
    check_clean_weights(packer, tokenizer)
    print("clean-endpoint weights ok: photo weight, clean rows, per-stream logs")
    check_stream_embedding(model, packer, tokenizer)
    print("stream identity ok: ids, signs, restore, legacy batches untouched")
    check_training_step(model, packer, tokenizer)
    print("training step ok: rows, weights, per-stream logs, gradients")
    check_legend_dataset()
    print("legend data ok: legend colours, decoding through the legend, presence counts")
    for vit_prob in (0.0, 1.0):
        check_legend_sample(packer, tokenizer, vit_prob)
    print("legend packing ok: legend loss only with ViT, RoPE ids after the legend")
    check_text_dropout(packer, tokenizer)
    print("text dropout ok: prompt and legend dropped together, map RoPE id after the prefix")
    check_legend_training_step(model, packer, tokenizer)
    print("legend training step ok: CE on the legend, rows of both symm modes")
    check_irepa(model, packer, tokenizer)
    print("iREPA ok: layer capture, one teacher pass per photo, z-score, loss, gradients")
    flow = make_flow(model, vae, tokenizer, new_token_ids)
    for use_vit in (False, True):
        print(f"parity ok, ViT {use_vit}: {check_parity(model, flow, packer, tokenizer, use_vit):.5f} relative difference")
    for use_vit in (False, True):
        diff = check_parity(model, flow, packer, tokenizer, use_vit, LEGEND)
        print(f"parity ok with a legend, ViT {use_vit}: {diff:.5f} relative difference")
    for use_vit in (False, True):
        diff = check_parity(model, flow, packer, tokenizer, use_vit, text_dropped=True)
        print(f"parity ok without text, ViT {use_vit}: {diff:.5f} relative difference")
    for levels in ((SYMM_T_EPS, 0.6), (0.4, SYMM_T_EPS)):
        diff = check_parity(model, flow, packer, tokenizer, True, LEGEND, levels=levels)
        print(f"parity ok in a conditional slice, levels {levels}: {diff:.5f} relative difference")
    check_sampler(model, flow)
    print("sampler ok: segmentation and synthesis, integrated and recomputed known stream")
    check_legend_sampler(flow)
    print("legend sampler ok: legend generation, both directions with a legend")
    check_guidance(flow)
    print("guidance ok: scale 1 unchanged, scale 0 text-free, both directions")
    check_hold_known(flow)
    print("hold-known sampler ok: one step unchanged, known stream held, exclusive with recompute")
    check_decoder(model, flow)
    print("decoder ok: unpacking, resolution, loss, empty labels, checkpoint")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
