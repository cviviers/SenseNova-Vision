#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Submit a fine-tuning run to the TU/e Spike cluster (Run:ai). Run this on your own machine; the job clones
# this repo from GitHub at a pinned commit into /data/code/<run> and trains with the configs in
# data/configs/spike/ (datasets, weights and outputs on the flowmatching PVC mounted at /data).
#
#   bash scripts/spike_submit.sh <run> <full|lora|tiny> <config.yaml> [extra trainer args]
#
#   full  scripts/finetune_full.sh (FSDP, all weights); resubmitting the same <run> resumes from its
#         latest checkpoint
#   lora  scripts/finetune_lora.sh (single GPU, no resume)
#   tiny  smoke test: full recipe on a 2-layer model built from the release (tools/make_tiny_model.py)
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
#   SPIKE_DRY_RUN=1   print the runai command instead of submitting it
set -euo pipefail

run="${1:?run name (lowercase letters, digits and dashes)}"
mode="${2:?mode: full, lora or tiny}"
config="${3:?dataset config yaml, e.g. data/configs/spike/cocostuff_img2palette_seg2img.yaml}"
shift 3

[[ "${run}" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] || { >&2 echo "Error: run name must match [a-z0-9-]"; exit 1; }
case "${mode}" in
  full) default_compute=eight-gpu ;;
  lora | tiny) default_compute=one-gpu ;;
  *) >&2 echo "Error: mode must be full, lora or tiny"; exit 1 ;;
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

[ -f "${repo_dir}/${config}" ] || { >&2 echo "Error: ${config} not found in ${repo_dir}"; exit 1; }

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
if ! git -C "${repo_dir}" cat-file -e "${commit}:${config}" 2>/dev/null; then
  >&2 echo "Warning: ${config} is not in commit ${commit:0:8} as far as the local clone knows; push it first."
fi

# Runs inside the container; $1 = mode, $2 = config, rest = extra trainer args.
job_script='
set -euo pipefail
mode="$1"; config="$2"; shift 2
code_dir="/data/code/${RUN}"
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

case "${mode}" in
  full)
    exec bash scripts/finetune_full.sh "${config}" "$@" ;;
  lora)
    exec bash scripts/finetune_lora.sh "${config}" --output_dir "${SENSENOVA_RUN_DIR}" "$@" ;;
  tiny)
    [ -f "${SENSENOVA_RUN_DIR}/tiny_model/config.json" ] || \
      python tools/make_tiny_model.py "${SENSENOVA_RUN_DIR}/tiny_model"
    SENSENOVA_MODEL_DIR="${SENSENOVA_RUN_DIR}/tiny_model" SENSENOVA_RUN_DIR="${SENSENOVA_RUN_DIR}/run" \
      exec bash scripts/finetune_full.sh "${config}" --total_steps 12 "$@" ;;
esac
'

cmd=(
  runai training standard submit "${run}" -p "${project}"
  -i "${image}"
  --image-pull-my-credentials "type=dockerRegistry,name=${pull_cred}"
  --run-as-uid "${uid}" --run-as-gid "${gid}" --create-home-dir
  --compute "${compute}"
  --large-shm
  --datasource "type=pvc,name=${project}"
  -e "RUN=${run}"
  -e "REPO_URL=${repo_url}"
  -e "COMMIT=${commit}"
  -e "SENSENOVA_MODEL_DIR=/data/models/SenseNova-Vision-7B-MoT"
  -e "HF_HOME=/data/hf_cache"
  --command -- bash -c "${job_script}" bash "${mode}" "${config}" "$@"
)

echo "run=${run} mode=${mode} compute=${compute} ref=${ref} commit=${commit:0:8} config=${config}"
if [ "${SPIKE_DRY_RUN:-0}" = "1" ]; then
  printf '%q ' "${cmd[@]}"; echo
  exit 0
fi
"${cmd[@]}"
echo "Follow:  runai training standard logs ${run} -p ${project} --follow"
echo "Outputs: /data/runs/${run}"
