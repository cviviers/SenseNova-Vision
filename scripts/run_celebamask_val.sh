#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Face parsing on CelebAMask-HQ + IoU evaluation.
#
#   bash scripts/run_celebamask_val.sh [output_dir] [extra batch_celebamask_segment.py args]
#
# Examples (inside the Docker container):
#   bash scripts/run_celebamask_val.sh results/celebamask_hq/pilot --limit 5          # pilot
#   bash scripts/run_celebamask_val.sh results/celebamask_hq/val_binary                # full val split
#   bash scripts/run_celebamask_val.sh results/celebamask_hq/val_generic --method generic
# Re-running the same command resumes; evaluation covers whatever predictions exist.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_dir}"
export PYTHONPATH="${repo_dir}:${repo_dir}/inference:${PYTHONPATH:-}"

output_dir="results/celebamask_hq/val_binary"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  output_dir="$1"
  shift
fi
dataset_root="${CELEBAMASK_ROOT:-/datasets/CelebAMask-HQ/CelebAMask-HQ}"

python -m inference.benchmark.batch_celebamask_segment \
  --dataset_root "${dataset_root}" \
  --output_dir "${output_dir}" \
  "$@"

python tools/evaluation/segment/evaluate_celebamask.py \
  --dataset_root "${dataset_root}" \
  --pred_dir "${output_dir}/pred"
