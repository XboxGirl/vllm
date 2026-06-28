#!/usr/bin/env bash
set -euo pipefail

docker build \
    --build-arg PYTORCH_NIGHTLY=1 \
    --build-arg torch_cuda_arch_list="12.0" \
    --build-arg max_jobs=8 \
    --build-arg nvcc_threads=2 \
    --build-arg RUN_WHEEL_CHECK=false \
    -t vllm-custom-cuda:latest \
    -f docker/Dockerfile \
    .
