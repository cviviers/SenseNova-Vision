#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
set -euo pipefail

# docker-compose runs as the host UID with HOME=/tmp/home (torch/triton caches).
mkdir -p "${HOME:-/tmp/home}"

if [ ! -f "${MODEL_PATH:-}/ema.safetensors" ]; then
  echo "[sensenova] Warning: ${MODEL_PATH:-<unset>}/ema.safetensors not found." >&2
  echo "[sensenova] Mount the model directory at /models/SenseNova-Vision-7B-MoT" >&2
  echo "[sensenova] (MODEL_DIR in docker/.env) or set MODEL_PATH." >&2
fi

exec "$@"
