#!/bin/bash

set -euo pipefail

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    return
  fi

  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi

  cat >&2 <<'EOF'
No usable python interpreter found. Activate your environment or set PYTHON_BIN.
EOF
  return 1
}

setup_wan_runtime_env() {
  local with_torch_lib="${1:-1}"

  export PYTHON_BIN
  resolve_python_bin

  export PYTHONNOUSERSITE=1
  export TOKENIZERS_PARALLELISM=false

  local cache_root="${HF_HOME:-${HOME}/.cache/sinkattention/huggingface}"
  export HF_HOME="${cache_root}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${cache_root}/hub}"
  export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-${cache_root}/diffusers}"
  mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${DIFFUSERS_CACHE}"

  if [[ "${with_torch_lib}" != "1" ]]; then
    return
  fi

  local torch_lib_dir
  torch_lib_dir="$("${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

sys.stdout.write(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
  if [[ -d "${torch_lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${torch_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

check_wan_python_runtime() {
  "${PYTHON_BIN}" - <<'PY'
import importlib

for module_name in ["torch", "huggingface_hub", "transformers", "diffusers"]:
    importlib.import_module(module_name)
PY
}

check_wan_block_sparse_backend() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys

spec = importlib.util.find_spec("block_sparse_attn")
if spec is not None:
    raise SystemExit(0)

hint_path = os.environ.get("BLOCK_SPARSE_ATTN_ROOT", "")
message = [
    "Wan Sink-Sparse runtime requires the external Block-Sparse-Attention backend (`block_sparse_attn`).",
    "The current Python environment cannot import `block_sparse_attn`.",
    "",
    "Install it in the same environment before using wanx/scripts/run_sink_inference.sh or --attn_mode sink.",
]

if hint_path and os.path.isdir(hint_path):
    message.extend(
        [
            "",
            f"Detected local checkout: {hint_path}",
            "Suggested install commands:",
            f"  cd {hint_path}",
            "  pip install packaging ninja",
            "  python setup.py install",
        ]
    )
else:
    message.extend(
        [
            "",
            "Reference upstream project:",
            "  https://github.com/mit-han-lab/Block-Sparse-Attention",
            "",
            "If you already have a checkout, set BLOCK_SPARSE_ATTN_ROOT=/path/to/Block-Sparse-Attention before running this wrapper.",
        ]
    )

sys.stderr.write("\n".join(message) + "\n")
raise SystemExit(1)
PY
}
