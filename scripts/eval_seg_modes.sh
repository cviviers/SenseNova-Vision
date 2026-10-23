#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Evaluate both fine-tuned modes for one LoRA checkpoint (or the base model with "base").
#
#   bash scripts/eval_seg_modes.sh <celebamask|cocostuff> <adapter_dir|base> <output_dir> [--limit N] [...]
#
#   img2palette: photo -> palette map, mIoU vs ground truth           -> <output_dir>/img2palette/metrics.json
#   seg2img:     palette map -> photo, photo rate / layout / FID / KID -> <output_dir>/seg2img/metrics.json
# Extra arguments go to both scripts (--split --limit --num_timesteps --cfg_text_scale --cfg_img_scale --seed);
# seg2img-only flags (--seeds N, --no_fid, --no_roundtrip, --no_photo_check) go in SEG2IMG_ARGS.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_dir}"
export PYTHONPATH="${repo_dir}:${repo_dir}/inference:${PYTHONPATH:-}"

dataset="${1:?celebamask or cocostuff}"
adapter="${2:?adapter dir or 'base'}"
out="${3:?output dir}"
shift 3
adapter_args=()
[ "${adapter}" != "base" ] && adapter_args=(--adapter "${adapter}")

python -m inference.benchmark.eval_seg_img2palette --dataset "${dataset}" "${adapter_args[@]}" \
  --output_dir "${out}/img2palette" "$@"
python -m inference.benchmark.eval_seg_seg2img --dataset "${dataset}" "${adapter_args[@]}" \
  --output_dir "${out}/seg2img" "$@" ${SEG2IMG_ARGS:-}
