#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Evaluate both fine-tuned CelebAMask-HQ modes for one LoRA checkpoint (or the base model with "base").
#
#   bash scripts/eval_celebamask_modes.sh <adapter_dir|base> <output_dir> [--limit N] [--seeds N] [...]
#
#   img2palette: face photo -> palette map, IoU vs ground truth      -> <output_dir>/img2palette/metrics.json
#   seg2img:     palette map -> face photo, face rate / layout / FID -> <output_dir>/seg2img/metrics.json
# Extra arguments go to both scripts (shared: --split --limit --num_timesteps --cfg_text_scale --cfg_img_scale);
# --seeds/--no_fid/--no_roundtrip/--no_face_check are seg2img-only and set via SEG2IMG_ARGS.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_dir}"
export PYTHONPATH="${repo_dir}:${repo_dir}/inference:${PYTHONPATH:-}"

adapter="${1:?adapter dir or 'base'}"
out="${2:?output dir}"
shift 2
adapter_args=()
[ "${adapter}" != "base" ] && adapter_args=(--adapter "${adapter}")

python -m inference.benchmark.eval_celebamask_img2palette "${adapter_args[@]}" --output_dir "${out}/img2palette" "$@"
python -m inference.benchmark.eval_celebamask_seg2img "${adapter_args[@]}" --output_dir "${out}/seg2img" "$@" ${SEG2IMG_ARGS:-}
