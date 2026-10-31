#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Evaluate a checkpoint on the three segmentation tasks of the fine-tuning configs: the released model
# (zero-shot baseline), a LoRA adapter (--adapter <dir>) or an exported full fine-tune (--model_path <dir>).
#
#   bash scripts/eval_seg_tasks.sh <celebamask|cocostuff> <output_dir> [--limit N] [--adapter DIR] [...]
#
#   img2palette  photo -> palette map with the training prompt, mIoU   -> <output_dir>/img2palette/metrics.json
#   img2legend   photo -> colour legend + mask, the released model's generic-segmentation format, decoded
#                through the legend (zero-shot baseline in the model's own format, both datasets; and the
#                legend-trained LoRA evaluations), mIoU                -> <output_dir>/img2legend/metrics.json
#   img2binseg   photo -> one binary mask per class (celebamask only;
#                inference/benchmark/batch_celebamask_segment.py), mIoU -> <output_dir>/img2binseg/metrics.json
#   seg2img      palette map -> photo, photo rate / layout / FID / KID  -> <output_dir>/seg2img/metrics.json
#
# Extra arguments go to every task (--adapter --model_path --split --limit --num_timesteps --cfg_text_scale
# --seed); seg2img-only flags (--prompt training|regions, --seeds N, --num_samples N, --check_seeds K, --no_fid,
# --no_roundtrip, --no_photo_check) go in SEG2IMG_ARGS. Zero-shot seg2img: --prompt regions --cfg_img_scale 1.2
# (the training prompt's colour key makes the base model copy the map). All tasks resume. img2legend, img2binseg and seg2img are sharded over all visible
# GPUs; img2palette uses the first one.
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
num_gpus="$(nvidia-smi -L | wc -l)"
echo "dataset=${dataset} root=${root} model=${MODEL_PATH} tasks=${tasks} gpus=${num_gpus} out=${out}"

# Run the command once per GPU with --num_shards/--shard_id appended (shards share the output directory).
run_shards() {
  local pids=() failed=0 i
  for ((i = 0; i < num_gpus; i++)); do
    CUDA_VISIBLE_DEVICES="${i}" "$@" --num_shards "${num_gpus}" --shard_id "${i}" > >(sed "s/^/[gpu${i}] /") 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  [ "${failed}" = 0 ] || { >&2 echo "Error: a shard failed"; exit 1; }
}

for task in ${tasks//,/ }; do
  case "${task}" in
    img2palette)
      python -m inference.benchmark.eval_seg_img2palette --dataset "${dataset}" --dataset_root "${root}" \
        --max_mem_per_gpu "${max_mem}" --output_dir "${out}/img2palette" "$@" ;;
    img2legend)
      legend_cmd=(python -m inference.benchmark.eval_seg_img2palette --dataset "${dataset}" --dataset_root "${root}"
        --format legend --max_mem_per_gpu "${max_mem}" --output_dir "${out}/img2legend" "$@")
      if [ "${num_gpus}" -gt 1 ]; then
        run_shards "${legend_cmd[@]}" --generate_only
      fi
      CUDA_VISIBLE_DEVICES=0 "${legend_cmd[@]}" ;;
    img2binseg)
      if [ "${dataset}" != celebamask ]; then
        echo "skip img2binseg: only implemented for celebamask"
        continue
      fi
      binseg_cmd=(python -m inference.benchmark.batch_celebamask_segment --method binary
        --prompt "${BINSEG_PROMPT:-native}" --dataset_root "${root}" --output_dir "${out}/img2binseg" "$@")
      if [ "${num_gpus}" -gt 1 ]; then
        run_shards "${binseg_cmd[@]}"
      else
        "${binseg_cmd[@]}"
      fi
      python tools/evaluation/segment/evaluate_celebamask.py --dataset_root "${root}" \
        --pred_dir "${out}/img2binseg/pred" ;;
    seg2img)
      seg2img_cmd=(python -m inference.benchmark.eval_seg_seg2img --dataset "${dataset}" --dataset_root "${root}"
        --max_mem_per_gpu "${max_mem}" --output_dir "${out}/seg2img" "$@" ${SEG2IMG_ARGS:-})
      if [ "${num_gpus}" -gt 1 ]; then
        # Shards only generate; one final pass computes the metrics from their files.
        run_shards "${seg2img_cmd[@]}" --generate_only
        CUDA_VISIBLE_DEVICES=0 "${seg2img_cmd[@]}"
      else
        "${seg2img_cmd[@]}"
      fi ;;
    *)
      >&2 echo "Error: unknown task ${task}"; exit 1 ;;
  esac
done
