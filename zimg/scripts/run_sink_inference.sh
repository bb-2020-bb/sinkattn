#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
source "${ROOT_DIR}/zimg/scripts/runtime_env.sh"

has_help_flag=false
attn_mode="dense"
prev_arg=""
for arg in "$@"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    has_help_flag=true
  fi
  if [[ "${arg}" == --attn_mode=* ]]; then
    attn_mode="${arg#--attn_mode=}"
  fi
  if [[ "${prev_arg}" == "--attn_mode" && -n "${arg}" ]]; then
    attn_mode="${arg}"
  fi
  prev_arg="${arg}"
done

if [[ "${has_help_flag}" == true ]]; then
  setup_zimg_runtime_env 0
else
  setup_zimg_runtime_env
fi

if [[ "${has_help_flag}" != true ]]; then
  check_zimg_python_runtime

  if [[ "${attn_mode}" == "sink" ]]; then
    check_zimg_block_sparse_backend
  fi
fi

"${PYTHON_BIN}" "${ROOT_DIR}/zimg/runtime/inference.py" "$@"
