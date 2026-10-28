#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Full fine-tuning (all model weights, FSDP) from the released SenseNova-Vision checkpoint on any dataset
# config, e.g. the two-mode CelebAMask-HQ / COCO-Stuff configs. Counterpart of scripts/finetune_lora.sh.
#
#   bash scripts/finetune_full.sh <dataset_config.yaml> [extra train/pretrain_unified_navit.py args]
#
# Hardware: the trainer keeps FP32 master weights, gradients, AdamW states and an EMA copy (~16 bytes per
# parameter + EMA, ~290 GB for 14.6B parameters) sharded over the GPUs of a node, so plan for >= 8 x 80 GB
# GPUs. Multi-node: run once per node with WORLD_SIZE (number of nodes), RANK, MASTER_ADDR, MASTER_PORT set.
#
# Environment (all optional):
#   SENSENOVA_MODEL_DIR      released model dir to start from (default /models/SenseNova-Vision-7B-MoT)
#   SENSENOVA_GPU_PER_NODE   default: all visible GPUs
#   SENSENOVA_NUM_SHARD      GPUs one model copy is sharded over (default: GPUs per node; must divide the total)
#   SENSENOVA_RUN_DIR        output dir (default results/full_<config name>); re-running resumes from its
#                            latest checkpoint
#
# Fine-tune defaults below can be overridden by passing the same flag again, e.g. --total_steps 2000.
# Each checkpoint stores full FP32 model + EMA weights (~58 GB each) plus sharded optimizer states.
# Export one for evaluation with tools/export_full_checkpoint.py.
set -euo pipefail

config="${1:?dataset config yaml}"
shift

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_dir}"
export PYTHONPATH="${repo_dir}:${repo_dir}/inference:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

model_dir="${SENSENOVA_MODEL_DIR:-${MODEL_PATH:-/models/SenseNova-Vision-7B-MoT}}"
gpu_per_node="${SENSENOVA_GPU_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
num_nodes="${WORLD_SIZE:-1}"
total_gpus=$((num_nodes * gpu_per_node))
num_shard="${SENSENOVA_NUM_SHARD:-${gpu_per_node}}"
(( total_gpus % num_shard == 0 )) || { >&2 echo "Error: num_shard (${num_shard}) must divide ${total_gpus} GPUs"; exit 1; }
num_replicate=$((total_gpus / num_shard))
run_name="$(basename "${config%.yaml}")"
run_dir="${SENSENOVA_RUN_DIR:-results/full_${run_name}}"

mkdir -p "${run_dir}"
export WANDB_DIR="${WANDB_DIR:-${run_dir}}"   # keep the offline wandb run with the training run

echo "config=${config} model=${model_dir} gpus=${total_gpus} (${num_nodes} node(s) x ${gpu_per_node})"
echo "num_shard=${num_shard} num_replicate=${num_replicate} run_dir=${run_dir}"

py_cmd=(
  train/pretrain_unified_navit.py
  --dataset_config_file "${config}"
  # Start from the released weights (EMA), model only; re-runs auto-resume from the latest checkpoint.
  --resume_from "${model_dir}"
  --resume_from_hf "${model_dir}"
  --resume_model_only True
  --finetune_from_ema True
  --auto_resume True
  --copy_init_moe False          # keep the trained generation expert (True would re-copy it from the und expert)
  --layer_module Qwen2MoTDecoderLayer
  --results_dir "${run_dir}"
  --checkpoint_dir "${run_dir}/checkpoint"
  --wandb_project sensenova-vision-finetune
  --wandb_name "${run_name}"
  --wandb_offline True
  # FSDP
  --sharding_strategy HYBRID_SHARD
  --num_shard "${num_shard}"
  --num_replicate "${num_replicate}"
  # Optimisation: the release's own fine-tune recipe (scripts/train_cv_unify.sh): constant LR 2.5e-5, EMA
  # 0.995, 2 accumulation steps; AdamW betas (0.9, 0.95), eps 1e-15 and no weight decay are the trainer
  # defaults. Shorter warmup (500 there) and run.
  # Units: total_steps, save_every and log_every count micro-batches (one packed batch per GPU, ~76 samples
  # of 512 px on 8 GPUs); warmup_steps and the EMA count optimizer updates (every 2 micro-batches). 2,000
  # steps = 1,000 updates, ~150k samples: ~6 passes over CelebAMask-HQ train or ~1.3 over COCO train2017.
  --lr 2.5e-5
  --lr_scheduler constant
  --warmup_steps 100
  --total_steps 2000
  --ema 0.995
  --max_grad_norm 1.0
  --gradient_accumulation_steps 2
  --log_every 10
  --save_every 500   # each checkpoint: FP32 model + EMA (~117 GB) plus optimizer shards
  # Losses and conditioning dropout (classifier-free guidance), as in pre-training.
  --timestep_shift 4.0
  --ce_weight 0.25
  --mse_weight 1.0
  --text_cond_dropout_prob 0.05
  --vit_cond_dropout_prob 0.1
  --vae_cond_dropout_prob 0.1
  # Sequence packing per GPU, as in train_cv_unify.sh: ~10 samples of 2.5-3.7k tokens per micro-step and GPU.
  # B200 (sm_100): pass --use_flex False (flex attention backward does not compile there with torch 2.8).
  --use_flex True
  --split_vae_encode True
  --max_latent_size 64
  --max_num_tokens 36864
  --expected_num_tokens 32768
  --max_num_tokens_per_sample 24064
  --prefer_buffer_before 15360
  --data_seed 42
  "$@"
)

if [ "${num_nodes}" -gt 1 ]; then
  torchrun --nnodes "${num_nodes}" --nproc_per_node "${gpu_per_node}" --node_rank "${RANK}" \
    --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" "${py_cmd[@]}"
else
  torchrun --standalone --nproc_per_node "${gpu_per_node}" "${py_cmd[@]}"
fi
