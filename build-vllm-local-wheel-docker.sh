#!/usr/bin/env bash
set -euo pipefail

wheel_count="$(find dist -maxdepth 1 -name 'vllm*.whl' 2>/dev/null | wc -l)"
if [[ "${wheel_count}" -ne 1 ]]; then
    echo "Expected exactly one dist/vllm*.whl, found ${wheel_count}." >&2
    echo "Run ./build-vllm-local-wheel.sh first, or set CLEAN_DIST=1 and rebuild the wheel." >&2
    exit 1
fi

image_tag="${IMAGE_TAG:-vllm-local-wheel:latest}"
base_image="${BASE_IMAGE:-nvidia/cuda:13.3.0-devel-ubuntu26.04}"
install_deps="${INSTALL_DEPS:-1}"

echo "Building Docker image from local vLLM wheel"
echo "  Image: ${image_tag}"
echo "  Base image: ${base_image}"
echo "  INSTALL_DEPS=${install_deps}"
echo "  Wheel: $(find dist -maxdepth 1 -name 'vllm*.whl' -print)"

docker build \
    --build-context vllm_wheel=dist \
    --build-arg BASE_IMAGE="${base_image}" \
    --build-arg INSTALL_DEPS="${install_deps}" \
    -t "${image_tag}" \
    -f docker/Dockerfile.prebuilt-wheel \
    .
