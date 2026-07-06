#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/render_motion_mesh.sh PATH_TO_SAMPLE_MP4 [OPTIONS]" >&2
  exit 2
fi

INPUT_PATH="$1"
shift
python -m alm.motion_completion render --input "${INPUT_PATH}" "$@"
