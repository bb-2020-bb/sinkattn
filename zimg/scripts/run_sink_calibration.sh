#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
source "${ROOT_DIR}/zimg/scripts/runtime_env.sh"

has_help_flag=false
for arg in "$@"; do
  if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
    has_help_flag=true
  fi
done

if [[ "${has_help_flag}" == true ]]; then
  setup_zimg_runtime_env 0
else
  setup_zimg_runtime_env
fi

if [[ "${has_help_flag}" != true ]]; then
  check_zimg_python_runtime
fi

"${PYTHON_BIN}" "${ROOT_DIR}/zimg/offline/calibrate_sink_sparse.py" "$@"
