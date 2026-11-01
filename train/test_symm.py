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
from data.dataset_base import SYMM_T_EPS, PackedDataset
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
from train.train_seg_decoder import segmentation_loss
from train.train_utils import per_group_losses, per_stream_losses, summarize_groups, accumulate_groups, symm_row_weights

SIZE = 128
PROMPT = "Two images of the same face. The first is a photograph, the second its segmentation map."
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


def check_parity(model, flow, packer, tokenizer, use_vit):
    """The inference path (cached prefix + joint query) must match the training path on identical inputs."""
    photo, mask = make_images()
    sample = make_dataset(tokenizer, 1.0 if use_vit else 0.0, beta=0.0).build_symm_sample(random.Random(1), photo,
                                                                                          PROMPT, mask)
    data = to_device(pack(packer, [sample]))
    data.pop("ce_loss_weights", None)
    images = data.pop("padded_images")
    t = 0.3
    logit = math.log(t / (1 - t))
    num_tokens = (SIZE // 16) ** 2
    data["packed_timesteps"] = torch.cat([torch.full((num_tokens,), logit), torch.full((num_tokens,), -logit)]).to(DEVICE)
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
        noise_level = torch.cat([torch.full((num_tokens,), t), torch.full((num_tokens,), 1 - t)]).to(DEVICE)
        noisy = (1 - noise_level[:, None]) * clean + noise_level[:, None] * captured["noise"].float()
        context = flow._context(PROMPT, photo if use_vit else None)
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
    assert diff < 0.01, f"train and inference disagree by {diff:.4f}"
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


def check_guards():
    assert uses_symm({"a": {"mode_weights": {"symm": 1.0}}})
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
    check_stream_embedding(model, packer, tokenizer)
    print("stream identity ok: ids, signs, restore, legacy batches untouched")
    check_training_step(model, packer, tokenizer)
    print("training step ok: rows, weights, per-stream logs, gradients")
    flow = make_flow(model, vae, tokenizer, new_token_ids)
    for use_vit in (False, True):
        print(f"parity ok, ViT {use_vit}: {check_parity(model, flow, packer, tokenizer, use_vit):.5f} relative difference")
    check_sampler(model, flow)
    print("sampler ok: segmentation and synthesis, integrated and recomputed known stream")
    check_decoder(model, flow)
    print("decoder ok: unpacking, resolution, loss, empty labels, checkpoint")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
