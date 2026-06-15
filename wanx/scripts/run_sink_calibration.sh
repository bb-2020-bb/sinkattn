#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/runtime_env.sh"

GPU="${WAN_GPU:-0}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  GPU="${1}"
  shift
fi
EXTRA_ARGS=("$@")

MASK_PATH="${WAN_ROOT}/outputs/sink_sparse/wan_sink_mask.pt"
MODEL_ARGS=()
if [[ -n "${WAN_MODEL:-}" ]]; then
  MODEL_ARGS=(--model_path "${WAN_MODEL}")
fi

has_help_flag=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    has_help_flag=true
  fi
done

if [[ "${has_help_flag}" == true ]]; then
  setup_wan_runtime_env 0
else
  setup_wan_runtime_env
  check_wan_python_runtime
fi

PROJECT_ROOT="$(cd "${WAN_ROOT}/.." && pwd)"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m wanx.offline.calibrate_sink_sparse \
  --gpu "${GPU}" \
  "${MODEL_ARGS[@]}" \
  --prompt_file "${WAN_ROOT}/prompts/calibration/sink_calibration_wanx_v1.txt" \
  --output_mask_path "${MASK_PATH}" \
  --num_prompts 4 \
  --seeds 8888,9999 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --output_type latent \
  --offline_direct_coverage 0.85 \
  "${EXTRA_ARGS[@]}"
