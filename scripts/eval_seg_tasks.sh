#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Evaluate a checkpoint on the three segmentation tasks of the fine-tuning configs: the released model
# (zero-shot baseline), a LoRA adapter (--adapter <dir>) or an exported full fine-tune (--model_path <dir>).
#
#   bash scripts/eval_seg_tasks.sh <celebamask|cocostuff> <output_dir> [--limit N] [--adapter DIR] [...]
#
#   img2palette  photo -> palette map with the training prompt, mIoU   -> <output_dir>/img2palette/metrics.json
#   img2binseg   photo -> one binary mask per class (celebamask only;
#                inference/benchmark/batch_celebamask_segment.py), mIoU -> <output_dir>/img2binseg/metrics.json
#   seg2img      palette map -> photo, photo rate / layout / FID / KID  -> <output_dir>/seg2img/metrics.json
#
# Extra arguments go to every task (--adapter --model_path --split --limit --num_timesteps --cfg_text_scale
# --seed); seg2img-only flags (--seeds N, --num_samples N, --check_seeds K, --no_fid, --no_roundtrip,
# --no_photo_check) go in SEG2IMG_ARGS. All tasks resume. seg2img is sharded over all visible GPUs.
# FID-50k on CelebAMask-HQ val (2993 maps, ~17 seeds each; photo check and round trip once per map):
#   EVAL_TASKS=seg2img SEG2IMG_ARGS="--num_samples 50000 --check_seeds 1" \
#     bash scripts/eval_seg_tasks.sh celebamask results/eval/base_celebamask_fid50k
#
# Environment (all optional):
#   EVAL_TASKS                 comma list (default: img2palette,img2binseg,seg2img; cocostuff skips img2binseg)
#   BINSEG_PROMPT              img2binseg prompt: native (released model's <p>class</p> format, default) or
#                              training (the img2binseg prompt of the fine-tuning data)
#   MODEL_PATH                 model dir (default: SENSENOVA_MODEL_DIR, else /models/SenseNova-Vision-7B-MoT)
#   CELEBAMASK_HQ_ROOT         default /datasets/CelebAMask-HQ/CelebAMask-HQ
#   COCOSTUFF_ROOT             dir with train2017/val2017 and stuffthingmaps zips (default /datasets/segmentation)
#   SENSENOVA_MAX_MEM_PER_GPU  GPU memory budget for the weights (default 30GiB, sized for a 32 GB card)
set -euo pipefail

dataset="${1:?celebamask or cocostuff}"
out="${2:?output dir}"
shift 2

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_dir}"
export PYTHONPATH="${repo_dir}:${repo_dir}/inference:${PYTHONPATH:-}"
export MODEL_PATH="${MODEL_PATH:-${SENSENOVA_MODEL_DIR:-/models/SenseNova-Vision-7B-MoT}}"
max_mem="${SENSENOVA_MAX_MEM_PER_GPU:-30GiB}"

case "${dataset}" in
  celebamask) root="${CELEBAMASK_HQ_ROOT:-/datasets/CelebAMask-HQ/CelebAMask-HQ}" ;;
  cocostuff) root="${COCOSTUFF_ROOT:-/datasets/segmentation}" ;;
  *) >&2 echo "Error: dataset must be celebamask or cocostuff"; exit 1 ;;
esac
tasks="${EVAL_TASKS:-img2palette,img2binseg,seg2img}"
echo "dataset=${dataset} root=${root} model=${MODEL_PATH} tasks=${tasks} out=${out}"

for task in ${tasks//,/ }; do
  case "${task}" in
    img2palette)
      python -m inference.benchmark.eval_seg_img2palette --dataset "${dataset}" --dataset_root "${root}" \
        --max_mem_per_gpu "${max_mem}" --output_dir "${out}/img2palette" "$@" ;;
    img2binseg)
      if [ "${dataset}" != celebamask ]; then
        echo "skip img2binseg: only implemented for celebamask"
        continue
      fi
      python -m inference.benchmark.batch_celebamask_segment --method binary --prompt "${BINSEG_PROMPT:-native}" \
        --dataset_root "${root}" --output_dir "${out}/img2binseg" "$@"
      python tools/evaluation/segment/evaluate_celebamask.py --dataset_root "${root}" \
        --pred_dir "${out}/img2binseg/pred" ;;
    seg2img)
      seg2img_cmd=(python -m inference.benchmark.eval_seg_seg2img --dataset "${dataset}" --dataset_root "${root}"
        --max_mem_per_gpu "${max_mem}" --output_dir "${out}/seg2img" "$@" ${SEG2IMG_ARGS:-})
      num_gpus="$(nvidia-smi -L | wc -l)"
      if [ "${num_gpus}" -gt 1 ]; then
        # One shard per GPU writing into the same directory, then one pass that only computes the metrics.
        pids=()
        for ((i = 0; i < num_gpus; i++)); do
          CUDA_VISIBLE_DEVICES="${i}" "${seg2img_cmd[@]}" --num_shards "${num_gpus}" --shard_id "${i}" \
            --generate_only > >(sed "s/^/[gpu${i}] /") 2>&1 &
          pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
        [ "${failed}" = 0 ] || { >&2 echo "Error: a seg2img shard failed"; exit 1; }
        CUDA_VISIBLE_DEVICES=0 "${seg2img_cmd[@]}"
      else
        "${seg2img_cmd[@]}"
      fi ;;
    *)
      >&2 echo "Error: unknown task ${task}"; exit 1 ;;
  esac
done
