#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# CelebAMask-HQ shortcut for scripts/eval_seg_modes.sh:
#   bash scripts/eval_celebamask_modes.sh <adapter_dir|base> <output_dir> [--limit N] [...]
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/eval_seg_modes.sh" celebamask "$@"
