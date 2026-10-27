#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Single-GPU LoRA / QLoRA fine-tuning (see docs/DOCKER_BLACKWELL.md).
#
#   bash scripts/finetune_lora.sh [dataset_config.yaml] [extra train/finetune_lora.py args]
#
# Defaults to the synthetic "glorp" demo, generating its data on first use.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_dir}"
export PYTHONPATH="${repo_dir}:${repo_dir}/inference:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

config="data/configs/lora_demo/glorp_lora.yaml"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  config="$1"
  shift
fi

if [ "${config}" = "data/configs/lora_demo/glorp_lora.yaml" ] && [ ! -f datas/lora_demo/train/glorp_bbox.jsonl ]; then
  python tools/make_lora_demo_data.py --output_dir datas/lora_demo
fi

exec python train/finetune_lora.py \
  --model_path "${SENSENOVA_MODEL_DIR:-${MODEL_PATH:?Set MODEL_PATH or SENSENOVA_MODEL_DIR}}" \
  --dataset_config_file "${config}" \
  "$@"
