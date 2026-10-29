#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Submit a fine-tuning run to the TU/e Spike cluster (Run:ai). Run this on your own machine; the job clones
# this repo from GitHub at a pinned commit into /data/code/<run> and trains with the configs in
# data/configs/spike/ (datasets, weights and outputs on the flowmatching PVC mounted at /data).
#
#   bash scripts/spike_submit.sh <run> <full|lora|tiny> <config.yaml> [extra trainer args]
#   bash scripts/spike_submit.sh <run> eval <celebamask|cocostuff> [extra eval args, e.g. --limit 500]
#       zero-shot: no extra model args; LoRA: --adapter /data/runs/<lora run>/final;
#       full fine-tune: --model_path /data/runs/<run>/export_<step>
#   bash scripts/spike_submit.sh <full run> export <step, e.g. 0000449>
#
#   full    scripts/finetune_full.sh (FSDP, all weights); resubmitting the same <run> resumes from its
#           latest checkpoint. B200 batch profile below: 2x the release's batch per update.
#   lora    scripts/finetune_lora.sh (single GPU; resubmitting the same <run> resumes from its latest
#           checkpoint); BF16 base and 131k-token batches sized for a B200 (profile below)
#   tiny    smoke test: full recipe on a 2-layer model built from the release (tools/make_tiny_model.py)
#   eval    scripts/eval_seg_tasks.sh (img2palette, img2binseg, seg2img; resumes; seg2img uses every GPU);
#           EVAL_TASKS, SEG2IMG_ARGS and BINSEG_PROMPT are passed through when set
#   export  tools/export_full_checkpoint.py: /data/runs/<run>/checkpoint/<step> -> /data/runs/<run>/export_<step>
#           (BF16 EMA weights, a model dir for --model_path); the workload is named <run>-export-<step>
#
# Commit and push first: the job only sees what is on GitHub. Outputs go to /data/runs/<run>.
#
# Environment (all optional):
#   SPIKE_REF         branch, tag or commit to run (default: the current local branch)
#   SPIKE_COMPUTE     Run:ai compute preset (default: eight-gpu for full, one-gpu for lora/tiny)
#   SPIKE_PROJECT     Run:ai project (default flowmatching)
#   SPIKE_IMAGE       default harbor.spike.tue.nl/flowmatching/sensenova-vision:1.0.0
#   SPIKE_PULL_CRED   personal dockerRegistry credential for Harbor (default harbor-pull)
#   SPIKE_UID/GID     user the job runs as; must own /data (default 20236095, `id` in the sshv3 workspace)
#   SPIKE_PREEMPTIBILITY  preemptible or non-preemptible (default: non-preemptible for full/lora/tiny, which
#                     must stay within the project quota; preemptible for eval and export, which may use idle
#                     GPUs over quota; eval resumes per image). Preempted training restarts from its latest
#                     checkpoint.
#   SPIKE_DRY_RUN=1   print the runai command instead of submitting it
set -euo pipefail

run="${1:?run name (lowercase letters, digits and dashes)}"
mode="${2:?mode: full, lora, tiny, eval or export}"
config="${3:?dataset config yaml (eval: celebamask or cocostuff; export: checkpoint step)}"
shift 3

[[ "${run}" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] || { >&2 echo "Error: run name must match [a-z0-9-]"; exit 1; }
case "${mode}" in
  full) default_compute=eight-gpu ;;
  lora | tiny | eval | export) default_compute=one-gpu ;;
  *) >&2 echo "Error: mode must be full, lora, tiny, eval or export"; exit 1 ;;
esac
case "${mode}" in
  eval | export) default_preempt=preemptible ;;
  *) default_preempt=non-preemptible ;;
esac
preemptibility="${SPIKE_PREEMPTIBILITY:-${default_preempt}}"
job_name="${run}"
case "${mode}" in
  eval)
    [[ "${config}" =~ ^(celebamask|cocostuff)$ ]] || { >&2 echo "Error: eval takes celebamask or cocostuff"; exit 1; }
    config_file=scripts/eval_seg_tasks.sh ;;
  export)
    [[ "${config}" =~ ^[0-9]{7}$ ]] || { >&2 echo "Error: export takes a 7-digit checkpoint step, e.g. 0000449"; exit 1; }
    config_file=tools/export_full_checkpoint.py
    # The training workload keeps the run's name; the export reads and writes the same /data/runs/<run>.
    job_name="${run}-export-$((10#${config}))" ;;
  *)
    config_file="${config}" ;;
esac

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
repo_url="https://github.com/cviviers/SenseNova-Vision"
project="${SPIKE_PROJECT:-flowmatching}"
compute="${SPIKE_COMPUTE:-${default_compute}}"
image="${SPIKE_IMAGE:-harbor.spike.tue.nl/flowmatching/sensenova-vision:1.0.0}"
pull_cred="${SPIKE_PULL_CRED:-harbor-pull}"
uid="${SPIKE_UID:-20236095}"
gid="${SPIKE_GID:-20236095}"
ref="${SPIKE_REF:-$(git -C "${repo_dir}" rev-parse --abbrev-ref HEAD)}"

[ -f "${repo_dir}/${config_file}" ] || { >&2 echo "Error: ${config_file} not found in ${repo_dir}"; exit 1; }

# Pin the commit now so retries and resumes of this run use the same code.
if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
  commit="${ref}"
else
  commit="$(git ls-remote "${repo_url}" "refs/heads/${ref}" "refs/tags/${ref}" | head -n1 | cut -f1)"
  [ -n "${commit}" ] || { >&2 echo "Error: ${ref} not found on ${repo_url} (pushed?)"; exit 1; }
fi
if [ "$(git -C "${repo_dir}" rev-parse HEAD)" != "${commit}" ]; then
  >&2 echo "Warning: local HEAD differs from ${ref} on GitHub (${commit:0:8}); the job runs the GitHub version."
fi
if ! git -C "${repo_dir}" cat-file -e "${commit}:${config_file}" 2>/dev/null; then
  >&2 echo "Warning: ${config_file} is not in commit ${commit:0:8} as far as the local clone knows; push it first."
fi

# Runs inside the container; $1 = mode, $2 = config, rest = extra trainer args.
job_script='
set -euo pipefail
mode="$1"; config="$2"; shift 2
# One checkout per job: jobs of the same run (e.g. two exports) would otherwise run git in one repo at once.
code_dir="/data/code/${JOB}"
if [ ! -d "${code_dir}/.git" ]; then
  git clone --quiet "${REPO_URL}" "${code_dir}"
fi
cd "${code_dir}"
git fetch --quiet origin
git checkout --quiet --force "${COMMIT}"
echo "repo=${REPO_URL} commit=$(git rev-parse HEAD) mode=${mode} config=${config}"

export SENSENOVA_RUN_DIR="/data/runs/${RUN}"
mkdir -p "${SENSENOVA_RUN_DIR}"
echo "${COMMIT}" >> "${SENSENOVA_RUN_DIR}/commits.txt"

# Spike presets give 13 CPU cores and ~262 GB RAM per GPU: 8 DataLoader workers per GPU (rank) leaves
# ~5 cores for the training process, pinning and NCCL. Extra args come last and override these.
loader_args=(--num_workers 8 --prefetch_factor 4)

# B200 (192 GB) batch profiles. With --use_flex False attention runs per sample, so memory grows linearly
# with the tokens per step; the 32k tokens per GPU of the release recipe leave most of a B200 unused. Batch-coupled
# settings (tokens, accumulation, lr) live together here; extra args come last and override them. If the
# logged peak memory passes ~165 GB, pass --expected_num_tokens 98304 --max_num_tokens 106496.
#
# Full: 131k tokens per GPU and no accumulation = ~300 samples of 512 px per update on 8 GPUs, 2x the
# release recipe (32k x 2 accumulation); lr scaled by sqrt(2) from 2.5e-5. total_steps, save_every and
# log_every then count updates: N steps = ~300 N samples (~4 N micro-steps of the release recipe).
# Estimated ~100-130 GB per GPU (sharded FP32 weights/grads/AdamW/EMA ~36 GB + activations).
# flex attention backward fails to compile on B200 (sm_100) with torch 2.8 / Triton 3.4 (full, tiny and lora).
full_args=(
  --use_flex False
  --expected_num_tokens 131072 --max_num_tokens 139264 --prefer_buffer_before 65536
  --gradient_accumulation_steps 1
  --lr 3.5e-5
)

# LoRA: BF16 base and 4x the v2 batch (32k tokens: 51 GB peak, ~6.6k tok/s, ~5 s/step, ~8.6 images/step),
# so ~34 images per step at an estimated ~115 GB peak; lr scaled by sqrt(4) from 1e-4.
lora_args=(
  --quant bf16
  --expected_num_tokens 131072 --max_num_tokens 139264
  --max_num_tokens_per_sample 16384 --prefer_buffer_before 65536
  --lr 2e-4
  --use_flex False
)

# No backslash line continuations in this script: Run:ai stores the command as one string and drops
# them, which silently cuts a command off at the line break.
case "${mode}" in
  full)
    exec bash scripts/finetune_full.sh "${config}" "${full_args[@]}" "${loader_args[@]}" "$@" ;;
  lora)
    exec bash scripts/finetune_lora.sh "${config}" --output_dir "${SENSENOVA_RUN_DIR}" "${lora_args[@]}" "${loader_args[@]}" "$@" ;;
  tiny)
    if [ ! -f "${SENSENOVA_RUN_DIR}/tiny_model/config.json" ]; then
      python tools/make_tiny_model.py "${SENSENOVA_RUN_DIR}/tiny_model"
    fi
    export SENSENOVA_MODEL_DIR="${SENSENOVA_RUN_DIR}/tiny_model" SENSENOVA_RUN_DIR="${SENSENOVA_RUN_DIR}/run"
    exec bash scripts/finetune_full.sh "${config}" --total_steps 12 "${full_args[@]}" "${loader_args[@]}" "$@" ;;
  eval)
    exec bash scripts/eval_seg_tasks.sh "${config}" "${SENSENOVA_RUN_DIR}" "$@" ;;
  export)
    exec python tools/export_full_checkpoint.py --checkpoint "${SENSENOVA_RUN_DIR}/checkpoint/${config}" --output_dir "${SENSENOVA_RUN_DIR}/export_${config}" "$@" ;;
esac
'

eval_env=()
[ -n "${EVAL_TASKS:-}" ] && eval_env+=(-e "EVAL_TASKS=${EVAL_TASKS}")
[ -n "${SEG2IMG_ARGS:-}" ] && eval_env+=(-e "SEG2IMG_ARGS=${SEG2IMG_ARGS}")
[ -n "${BINSEG_PROMPT:-}" ] && eval_env+=(-e "BINSEG_PROMPT=${BINSEG_PROMPT}")

cmd=(
  runai training standard submit "${job_name}" -p "${project}"
  -i "${image}"
  --image-pull-my-credentials "type=dockerRegistry,name=${pull_cred}"
  --run-as-uid "${uid}" --run-as-gid "${gid}" --create-home-dir
  --compute "${compute}"
  --preemptibility "${preemptibility}"
  --large-shm
  --datasource "type=pvc,name=${project}"
  -e "RUN=${run}"
  -e "JOB=${job_name}"
  -e "REPO_URL=${repo_url}"
  -e "COMMIT=${commit}"
  # Both names: the image sets them to its docker-compose mount (/models/...); training reads
  # SENSENOVA_MODEL_DIR, the evaluation scripts MODEL_PATH.
  -e "SENSENOVA_MODEL_DIR=/data/models/SenseNova-Vision-7B-MoT"
  -e "MODEL_PATH=/data/models/SenseNova-Vision-7B-MoT"
  -e "HF_HOME=/data/hf_cache"
  # The uid has no /etc/passwd entry in the image; torch.compile needs a user name for its cache dir.
  -e "USER=developer"
  -e "TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor"
  -e "TRITON_CACHE_DIR=/data/cache/triton"
  # Evaluation: Spike dataset roots, and the whole model on one B200 (no CPU offload).
  -e "CELEBAMASK_HQ_ROOT=/data/datasets/segmentation/CelebAMask-HQ"
  -e "COCOSTUFF_ROOT=/data/datasets/segmentation"
  -e "SENSENOVA_MAX_MEM_PER_GPU=160GiB"
  "${eval_env[@]}"
  --command -- bash -c "${job_script}" bash "${mode}" "${config}" "$@"
)

echo "run=${run} job=${job_name} mode=${mode} compute=${compute} ${preemptibility} ref=${ref} commit=${commit:0:8} config=${config}"
if [ "${SPIKE_DRY_RUN:-0}" = "1" ]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
"${cmd[@]}"
echo "Follow:  runai training standard logs ${job_name} -p ${project} --follow"
echo "Outputs: /data/runs/${run}"
