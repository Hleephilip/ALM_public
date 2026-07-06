#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

python -m alm.mesh_texturing \
  --mesh data/meshes/suitcase.obj \
  --prompt "a suitcase" \
  --output-dir outputs/mesh_texturing/

