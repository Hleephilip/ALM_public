#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

python -m alm.motion_completion evaluate \
  --edit-mode first_half \
  --transition-length 10 \
  --w_1 1 \
  --w_2 0.005 \
  "$@"
