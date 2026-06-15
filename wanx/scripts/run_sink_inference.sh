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

OUTPUT_DIR="${WAN_ROOT}/outputs/inference/wanx_batch"
MODEL_ARGS=()
if [[ -n "${WAN_MODEL:-}" ]]; then
  MODEL_ARGS=(--model_path "${WAN_MODEL}")
fi

has_help_flag=false
has_sink_mask_arg=false
has_attn_mode_arg=false
attn_mode="sink"
prev_arg=""
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    has_help_flag=true
  fi
  if [[ "${arg}" == --attn_mode=* ]]; then
    has_attn_mode_arg=true
    attn_mode="${arg#--attn_mode=}"
  fi
  if [[ "${prev_arg}" == "--attn_mode" && -n "${arg}" ]]; then
    has_attn_mode_arg=true
    attn_mode="${arg}"
  fi
  if [[ "${arg}" == --sink_mask_path=* ]]; then
    has_sink_mask_arg=true
  fi
  if [[ "${prev_arg}" == "--sink_mask_path" && -n "${arg}" ]]; then
    has_sink_mask_arg=true
  fi
  prev_arg="${arg}"
done

if [[ "${has_help_flag}" == true ]]; then
  setup_wan_runtime_env 0
else
  setup_wan_runtime_env
fi

if [[ "${has_help_flag}" != true ]]; then
  check_wan_python_runtime

  if [[ "${attn_mode}" == "sink" && "${has_sink_mask_arg}" != true ]]; then
    if [[ -n "${SINK_MASK_PATH:-}" ]]; then
      EXTRA_ARGS+=(--sink_mask_path "${SINK_MASK_PATH}")
    else
      cat >&2 <<EOF
wanx/scripts/run_sink_inference.sh requires an explicit Sink mask package.

The public repository does not ship a guaranteed-ready mask file.
Generate one first with:
  bash wanx/scripts/run_sink_calibration.sh 0 --output_mask_path /path/to/my_wan_sink_mask.pt

Then run inference with either:
  SINK_MASK_PATH=/path/to/my_wan_sink_mask.pt bash wanx/scripts/run_sink_inference.sh ${GPU}
or:
  bash wanx/scripts/run_sink_inference.sh ${GPU} --sink_mask_path /path/to/my_wan_sink_mask.pt
EOF
      exit 1
    fi
  fi

  if [[ "${attn_mode}" == "sink" ]]; then
    check_wan_block_sparse_backend
  fi
fi

ATTN_ARGS=()
if [[ "${has_attn_mode_arg}" != true ]]; then
  ATTN_ARGS=(--attn_mode sink)
fi

PROJECT_ROOT="$(cd "${WAN_ROOT}/.." && pwd)"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m wanx.runtime.inference \
  --gpu "${GPU}" \
  "${MODEL_ARGS[@]}" \
  --prompt_file "${WAN_ROOT}/prompts/inference_prompts.txt" \
  --output_dir "${OUTPUT_DIR}" \
  "${ATTN_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
