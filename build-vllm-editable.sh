#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CCACHE_NOHASHDIR="${CCACHE_NOHASHDIR:-true}"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export NINJA_STATUS="${NINJA_STATUS:-[%f/%t %o/sec] }"

echo "Building vLLM in editable mode"
echo "  Python: $(.venv/bin/python -V)"
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "  MAX_JOBS=${MAX_JOBS}"
echo "  NVCC_THREADS=${NVCC_THREADS}"
echo "  CCACHE_NOHASHDIR=${CCACHE_NOHASHDIR}"
echo "  NINJA_STATUS=${NINJA_STATUS}"

uv pip install --python .venv/bin/python \
    -v \
    --no-build-isolation \
    -e . \
    --torch-backend=auto
