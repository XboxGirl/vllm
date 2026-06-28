#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12
fi

export PATH="$(pwd)/.venv/bin:${PATH}"

missing_modules="$(
    .venv/bin/python - <<'PY'
import importlib.util

modules = (
    "jinja2",
    "packaging",
    "setuptools",
    "setuptools_rust",
    "setuptools_scm",
    "wheel",
)
missing = [module for module in modules if importlib.util.find_spec(module) is None]
print(" ".join(missing))
PY
)"

if [[ -n "${missing_modules}" ]]; then
    uv pip install --python .venv/bin/python \
        "cmake>=3.26.1" \
        ninja \
        packaging \
        "setuptools>=77.0.3,<81.0.0" \
        "setuptools-scm>=8.0" \
        "setuptools-rust>=1.9.0" \
        wheel \
        jinja2
fi

if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python "cmake>=3.26.1" ninja
fi

if ! find vllm -name '*.so' -print -quit | grep -q .; then
    echo "No existing vLLM .so artifacts found under ./vllm." >&2
    echo "Run ./build-vllm-editable.sh first, then rerun this script." >&2
    exit 1
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CCACHE_NOHASHDIR="${CCACHE_NOHASHDIR:-true}"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export NINJA_STATUS="${NINJA_STATUS:-[%f/%t %o/sec] }"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX="${VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX:-1}"

mkdir -p dist
if [[ "${CLEAN_DIST:-1}" != "0" ]]; then
    rm -f dist/vllm*.whl
fi

echo "Building vLLM wheel from the current checkout"
echo "  Python: $(.venv/bin/python -V)"
echo "  VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}"
echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "  MAX_JOBS=${MAX_JOBS}"
echo "  NVCC_THREADS=${NVCC_THREADS}"
echo "  CLEAN_DIST=${CLEAN_DIST:-1}"
echo
echo "This reuses existing CMake/Ninja outputs when they are up to date."

.venv/bin/python setup.py bdist_wheel \
    --dist-dir=dist \
    --py-limited-api=cp38

echo
echo "Built wheel:"
ls -lh dist/vllm*.whl
