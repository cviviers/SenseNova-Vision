# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU LoRA / QLoRA fine-tuning for SenseNova-Vision.

`pretrain_unified_navit.py` is a full-parameter FSDP trainer sized for 16+ x
80GB GPUs. This entry keeps its data pipeline and losses (packed CE + flow
matching MSE) but freezes the base model and trains LoRA adapters only, so a
single 24-32GB GPU (e.g. RTX 5090) can adapt the model:

  * the LLM decoder linears (~13B of the ~14B parameters) are stored in NF4
    (`--quant nf4`, default) or kept in BF16 (`--quant bf16`, needs >40GB),
  * LoRA adapters are injected into the understanding expert, the generation
    expert, or both (`--lora_experts`),
  * every decoder layer is activation-checkpointed.

Adapters are saved as `adapter_model.safetensors` + `adapter_config.json`.
Use `tools/merge_lora.py` to fold them into a regular model directory that the
standard inference entrypoints can load.
"""

import functools
import json
import math
import os
from dataclasses import asdict, dataclass, field
from time import time

import torch
import torch.nn as nn
import yaml
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import get_cosine_schedule_with_warmup

from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.dataset_info import DATASET_INFO
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.bagel.qwen2_navit import Qwen2MoTDecoderLayer
from modeling.qwen2 import Qwen2Tokenizer
from train.pretrain_unified_navit import _encode_padded_images_in_chunks

CHECKPOINT_WRAPPER_PREFIX = "_checkpoint_wrapped_module."
LAYER_PREFIX = r"language_model\.model\.layers\.\d+\."
LORA_TARGETS = {
    "und": LAYER_PREFIX + r"(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)",
    "gen": LAYER_PREFIX + r"(self_attn\.[qkvo]_proj_moe_gen|mlp_moe_gen\.(gate|up|down)_proj)",
}
LORA_TARGETS["both"] = f"({LORA_TARGETS['und']})|({LORA_TARGETS['gen']})"


@dataclass
class FinetuneArguments:
    model_path: str = field(
        default=os.environ.get("SENSENOVA_MODEL_DIR", "/models/SenseNova-Vision-7B-MoT"),
        metadata={"help": "SenseNova-Vision model directory (ema.safetensors, ae.safetensors, configs)."},
    )
    dataset_config_file: str = field(
        default="data/configs/lora_demo/glorp_lora.yaml",
        metadata={"help": "Dataset YAML in the PackedDataset format. Groups may define a `datasets:` "
                          "mapping to register JSONLs that are not in data/dataset_info.py."},
    )
    output_dir: str = field(default="results/lora_demo")
    quant: str = field(default="nf4", metadata={"help": "nf4 (QLoRA) or bf16 base weights."})
    lora_experts: str = field(
        default="both",
        metadata={"help": "Which MoT expert gets adapters: und (text/understanding), gen (image generation), both."},
    )
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lr: float = field(default=1e-4)
    weight_decay: float = field(default=0.0)
    warmup_steps: int = field(default=20)
    total_steps: int = field(default=300, metadata={"help": "Optimizer steps."})
    gradient_accumulation_steps: int = field(default=1)
    max_grad_norm: float = field(default=1.0)
    ce_weight: float = field(default=0.25)
    mse_weight: float = field(default=1.0)
    timestep_shift: float = field(default=4.0)
    text_cond_dropout_prob: float = field(default=0.05)
    vit_cond_dropout_prob: float = field(default=0.1)
    vae_cond_dropout_prob: float = field(default=0.1)
    expected_num_tokens: int = field(default=6144)
    max_num_tokens: int = field(default=8192)
    max_num_tokens_per_sample: int = field(default=6144)
    prefer_buffer_before: int = field(default=4096)
    max_buffer_size: int = field(default=50)
    num_workers: int = field(default=2)
    prefetch_factor: int = field(default=2)
    use_flex: bool = field(
        default=False,
        metadata={"help": "Packed flex-attention (as in pretraining). The default SDPA path with "
                          "per-sample masks avoids torch.compile and is simpler on consumer GPUs."},
    )
    log_every: int = field(default=10)
    save_every: int = field(default=100)
    seed: int = field(default=42)


def build_model(model_path: str, timestep_shift: float):
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.freeze_und = False

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=timestep_shift,
    )
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    state_dict = load_file(os.path.join(model_path, "ema.safetensors"), device="cpu")
    result = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    if result.unexpected_keys:
        print(f"[finetune] unexpected checkpoint keys: {result.unexpected_keys[:10]}")
    still_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if still_meta:
        raise RuntimeError(f"Checkpoint is missing weights for: {still_meta[:10]}")

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        raise RuntimeError("Tokenizer is missing SenseNova-Vision special tokens; use the released model directory.")
    return model, vae_model, vae_config, tokenizer, new_token_ids


def quantize_decoder_linears(model: Bagel, device: torch.device) -> None:
    """Swap every nn.Linear inside the LLM decoder layers for a frozen NF4 Linear4bit."""
    import bitsandbytes as bnb

    for layer in model.language_model.model.layers:
        for parent in list(layer.modules()):
            for child_name, child in list(parent.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                qlinear = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                    device="meta",
                )
                qlinear.weight = bnb.nn.Params4bit(
                    child.weight.data.to(torch.bfloat16), requires_grad=False, quant_type="nf4"
                )
                if child.bias is not None:
                    qlinear.bias = nn.Parameter(child.bias.data.to(torch.bfloat16), requires_grad=False)
                setattr(parent, child_name, qlinear.to(device))
        layer.to(device)


def inject_lora(model: Bagel, args: FinetuneArguments) -> dict:
    from peft import LoraConfig, inject_adapter_in_model

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGETS[args.lora_experts],
        bias="none",
    )
    inject_adapter_in_model(lora_config, model)
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
    return {
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_experts": args.lora_experts,
        "target_modules": LORA_TARGETS[args.lora_experts],
        "base_model_path": os.path.abspath(args.model_path),
        "quant": args.quant,
    }


def lora_state_dict(model: nn.Module) -> dict:
    return {
        name.replace(CHECKPOINT_WRAPPER_PREFIX, ""): param.detach().to("cpu", torch.float32).contiguous()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def save_adapter(model: nn.Module, adapter_config: dict, save_dir: str, step: int) -> None:
    os.makedirs(save_dir, exist_ok=True)
    save_file(lora_state_dict(model), os.path.join(save_dir, "adapter_model.safetensors"))
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump({**adapter_config, "step": step}, f, indent=2)
    print(f"[finetune] saved adapter to {save_dir}")


def build_dataloader(args: FinetuneArguments, tokenizer, new_token_ids, vae_config) -> DataLoader:
    with open(args.dataset_config_file) as f:
        dataset_meta = yaml.safe_load(f)

    # Allow small custom datasets to be declared inline instead of editing data/dataset_info.py.
    for group_name, group_args in dataset_meta.items():
        inline = group_args.pop("datasets", None)
        if inline:
            DATASET_INFO.setdefault(group_name, {}).update(inline)
            group_args.setdefault("dataset_names", list(inline))

    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    dataset_config.vit_patch_size = 14
    dataset_config.max_num_patch_per_side = 70
    dataset_config.vae_image_downsample = 2 * vae_config.downsample
    dataset_config.max_latent_size = 64
    dataset_config.text_cond_dropout_prob = args.text_cond_dropout_prob
    dataset_config.vae_cond_dropout_prob = args.vae_cond_dropout_prob
    dataset_config.vit_cond_dropout_prob = args.vit_cond_dropout_prob

    dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=0,
        world_size=1,
        num_workers=args.num_workers,
        expected_num_tokens=args.expected_num_tokens,
        max_num_tokens_per_sample=args.max_num_tokens_per_sample,
        max_num_tokens=args.max_num_tokens,
        max_buffer_size=args.max_buffer_size,
        prefer_buffer_before=args.prefer_buffer_before,
        interpolate_pos=False,
        use_flex=args.use_flex,
    )
    dataset.set_epoch(args.seed)
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )


def compute_loss(loss_dict: dict, args: FinetuneArguments):
    loss = 0.0
    logs = {}
    if loss_dict["has_ce"] and loss_dict["ce"].numel() > 0:
        ce = loss_dict["ce"].mean()
        loss = loss + args.ce_weight * ce
        logs["ce"] = ce.item()
    if loss_dict["has_mse"] and loss_dict["mse"].numel() > 0:
        mse = loss_dict["mse"].mean()
        loss = loss + args.mse_weight * mse
        logs["mse"] = mse.item()
    return loss, logs


@torch.no_grad()
def per_group_losses(loss_dict, data, groups):
    """Sum and token count of the image (mse) and text (ce) losses per sample group (e.g. a dataset's mode).

    Samples sit in the packed sequence in the same order as `groups`; each loss row belongs to the
    sample whose span (from sample_lens) contains its sequence position.
    """
    ends = torch.cumsum(torch.as_tensor(data["sample_lens"], device="cpu"), 0)
    out = {}
    for key, index_key, reduce in (("mse", "mse_loss_indexes", lambda x: x.float().mean(-1)),
                                   ("ce", "ce_loss_indexes", lambda x: x.float())):
        values = loss_dict.get(key)
        if values is None or values.numel() == 0 or index_key not in data:
            continue
        sample_ids = torch.searchsorted(ends, data[index_key].cpu(), right=True)
        per_token = reduce(values.detach()).cpu()
        for sid in sample_ids.unique().tolist():
            sel = sample_ids == sid
            total, count = out.setdefault(groups[sid], {}).get(key, (0.0, 0))
            out[groups[sid]][key] = (total + float(per_token[sel].sum()), count + int(sel.sum()))
    for group in out:
        out[group]["samples"] = groups.count(group)
    return out


def main():
    (args,) = HfArgumentParser(FinetuneArguments).parse_args_into_dataclasses()
    assert torch.cuda.is_available(), "CUDA is required"
    assert args.quant in {"nf4", "bf16"}, "--quant must be nf4 or bf16"
    assert args.lora_experts in LORA_TARGETS, f"--lora_experts must be one of {list(LORA_TARGETS)}"
    set_seed(args.seed)
    device = torch.device("cuda", torch.cuda.current_device())
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "finetune_args.json"), "w") as f:
        json.dump(asdict(args), f, indent=2)

    t0 = time()
    model, vae_model, vae_config, tokenizer, new_token_ids = build_model(args.model_path, args.timestep_shift)
    if args.quant == "nf4":
        quantize_decoder_linears(model, device)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    vae_model.to(device).eval().requires_grad_(False)

    adapter_config = inject_lora(model, args)
    model.to(device)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module: isinstance(module, Qwen2MoTDecoderLayer),
    )
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable)
    print(
        f"[finetune] model ready in {time() - t0:.0f}s | quant={args.quant} experts={args.lora_experts} "
        f"trainable={num_trainable / 1e6:.1f}M | gpu mem={torch.cuda.memory_allocated() / 2**30:.1f}GiB"
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_steps)
    loader = build_dataloader(args, tokenizer, new_token_ids, vae_config)

    step, micro_step = 0, 0
    window = {"loss": [], "ce": [], "mse": [], "tokens": 0, "groups": {}}
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    t_log = time()
    torch.cuda.reset_peak_memory_stats()
    for data in loader:
        data = data.cuda(device).to_dict()
        # Per-sample group for logging: the dataset's mode when it reports one, else the dataset name.
        groups = [d.get("mode", d.get("dataset_name", "all")) for d in data.get("batch_data_indexes", [])]
        for key in ("batch_data_indexes", "ce_loss_weights", "grouped_names", "batch_group_name"):
            data.pop(key, None)
        window["tokens"] += int(data["sequence_length"])

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if "padded_images" in data:
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                    data["padded_latent"] = _encode_padded_images_in_chunks(vae_model, data.pop("padded_images"))
            loss_dict = model(**data)
            loss, logs = compute_loss(loss_dict, args)
        if groups:
            batch_groups = per_group_losses(loss_dict, data, groups)
            total = sum(v["mse"][0] for v in batch_groups.values() if "mse" in v)
            count = sum(v["mse"][1] for v in batch_groups.values() if "mse" in v)
            if "mse" in logs and count and abs(total / count - logs["mse"]) > 1e-3 * max(1.0, logs["mse"]):
                print(f"[finetune] WARNING per-group mse {total / count:.4f} != batch mse {logs['mse']:.4f}: "
                      f"groups={groups} sample_lens={list(data['sample_lens'])} mse_rows={loss_dict['mse'].shape} "
                      f"mapped={count}", flush=True)
            for group, vals in batch_groups.items():
                acc = window["groups"].setdefault(group, {"mse": [0.0, 0], "ce": [0.0, 0], "samples": 0})
                acc["samples"] += vals.pop("samples")
                for key, (total, count) in vals.items():
                    acc[key][0] += total
                    acc[key][1] += count

        (loss / args.gradient_accumulation_steps).backward()
        window["loss"].append(float(loss))
        for key in ("ce", "mse"):
            if key in logs:
                window[key].append(logs[key])
        micro_step += 1
        if micro_step % args.gradient_accumulation_steps:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0 or step == 1:
            def mean(xs):
                return sum(xs) / len(xs) if xs else math.nan

            elapsed = time() - t_log
            per_group = {
                g: {k: acc[k][0] / acc[k][1] for k in ("mse", "ce") if acc[k][1]} | {"samples": acc["samples"]}
                for g, acc in sorted(window["groups"].items())
            }
            group_text = " ".join(
                f"{k}[{g}]={v[k]:.4f}" for g, v in per_group.items() for k in ("mse", "ce") if k in v
            )
            print(
                f"[finetune] step {step:5d}/{args.total_steps} loss={mean(window['loss']):.4f} "
                f"ce={mean(window['ce']):.4f} mse={mean(window['mse']):.4f} {group_text} "
                f"grad_norm={float(grad_norm):.3f} lr={scheduler.get_last_lr()[0]:.2e} "
                f"tok/s={window['tokens'] / elapsed:.0f} peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
                flush=True,
            )
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": mean(window["loss"]), "ce": mean(window["ce"]),
                                    "mse": mean(window["mse"]), "per_group": per_group,
                                    "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                                    "tok_per_s": window["tokens"] / elapsed}) + "\n")
            window = {"loss": [], "ce": [], "mse": [], "tokens": 0, "groups": {}}
            t_log = time()

        if step % args.save_every == 0 and step < args.total_steps:
            save_adapter(model, adapter_config, os.path.join(args.output_dir, f"checkpoint-{step:06d}"), step)
        if step >= args.total_steps:
            break

    save_adapter(model, adapter_config, os.path.join(args.output_dir, "final"), step)
    # DataLoader workers iterate an infinite dataset; exit without joining them.
    os._exit(0)


if __name__ == "__main__":
    main()
