source .venv/bin/activate

export VLLM_LOGGING_LEVEL=DEBUG

export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_KV_CACHE_LAYOUT=HND
# Set to 1 to A/B test applying top-k/top-p to probabilistic MTP draft sampling.
export VLLM_SPEC_DECODE_APPLY_SAMPLING_PARAMS_TO_DRAFT=0

export SCRIPT_DIR=/home/sirius/nvidia-vllm
export VLLM_CACHE_ROOT="$SCRIPT_DIR/cache/test-27b/vllm"
export FLASHINFER_WORKSPACE_BASE="$SCRIPT_DIR/cache/test-27b"
export FLASHINFER_COMPILE_CACHE_DIR="$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$SCRIPT_DIR/cache/test-27b/flashinfer-autotune"
export TRITON_CACHE_DIR="$SCRIPT_DIR/cache/test-27b/triton"
export TORCHINDUCTOR_CACHE_DIR="$SCRIPT_DIR/cache/test-27b/torchinductor"

mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" \
    "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" \
    "$FLASHINFER_COMPILE_CACHE_DIR"

echo "Triton cache:           $TRITON_CACHE_DIR"
echo "vLLM cache:             $VLLM_CACHE_ROOT"
echo "TorchInductor cache:    $TORCHINDUCTOR_CACHE_DIR"
echo "FlashInfer compile:     $FLASHINFER_COMPILE_CACHE_DIR"
echo "FlashInfer cache:       $VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"

# disable_flashinfer_q_quantization works around the FlashInfer XQA dtype
# assertion on Blackwell while upstream XQA decode support is still being
# finalized in vllm-project/vllm#43232.
vllm serve \
    "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP" \
    --served-model-name qwen-3.6-27b \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-model-len 262144 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-config.disable_flashinfer_q_quantization true \
    --language-model-only \
    --gpu-memory-utilization 0.95 \
    --dtype=auto \
    --quantization=modelopt \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --chat-template qwen-3.6-enhanced.jinja \
    --default-chat-template-kwargs '{"preserve_thinking": true, "enable_thinking": true}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
    --enable-prefix-caching \
    --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'