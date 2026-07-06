#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

python -m alm.inpainting \
  --input data/images/sample_cat.jpg \
  --mask data/masks/sample_mask.png \
  --prompt "a photo of a cat" \
  --output-dir outputs/inpainting
