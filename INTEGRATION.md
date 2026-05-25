# Genesis vLLM Patch Integration

## Overview

This branch integrates genesis vLLM patches into the latest vLLM `main`. The patches address Qwen3 reasoning/parser bugs, TurboQuant kernel issues, Marlin quantization alignment, NVFP4 loader failures, and MTP KV-cache conservatism.

---

## Applied Patches

### P3 — BF16→FP8 Cast Fix

- **File:** `vllm/v1/attention/ops/triton_turboquant_store.py`
- **Problem:** `convert_custom_float8_sm80` only accepts fp16/fp32 inputs. On SM < 89 GPUs, the direct BF16→FP8 cast crashes.
- **Fix:** Cast to FP16 first, then to FP8. Safe fallback for older GPU architectures.

### P27 — BEFORE-THINK Fallback

- **File:** `vllm/reasoning/qwen3_reasoning_parser.py`
- **Problem:** Qwen3 outputs text before the `<??` token (BEFORE-THINK). The parser drops this text silently.
- **Fix:** Capture BEFORE-THINK text and prepend it to content delta so it's not lost. Fixes quality regression (vllm#4069).

### P61 — Multi-tool First-Occurrence Fix

- **File:** `vllm/reasoning/qwen3_reasoning_parser.py`
- **Problem:** When multiple tool calls exist, using the **last** `<??` token as content start point causes tool calls after the first one to be lost.
- **Fix:** Use the **first** occurrence of the `<??` token so all tool calls are preserved.

### P65 — TurboQuant Spec-Decode CUDA Graph Downgrade

- **File:** `vllm/v1/attention/backends/turboquant_attn.py`
- **Problem:** TurboQuant + spec-decode + FULL CUDA graphs produces degenerate output.
- **Fix:** Keep `UNIFORM_BATCH` as the ClassVar default, but override `get_cudagraph_support` to downgrade to `UNIFORM_SINGLE_TOKEN_DECODE` when spec-decode is active. Context-aware decision, not a blanket disable.

### P67 — Multi-Query Continuation Prefill Hook

- **File:** `vllm/v1/attention/backends/turboquant_attn.py`
- **Problem:** K+1 spec-verify continuation batches take the CUDA graph bypass path, which is buggy for TurboQuant.
- **Fix:** Detect multi-query + spec-verify batches and route through a custom Triton kernel (`vllm._genesis.kernels.p67_multi_query_kernel`) that handles the edge case correctly.

### P91 — AutoRound Ceiling Division (Re-implemented)

- **Files:** `vllm/model_executor/layers/quantization/auto_gptq.py`, `vllm/model_executor/parameter.py`
- **Problem:** The original vLLM code used floor division (`//`) for `scales_and_zp_size`. When `input_size % group_size != 0`, this miscalculates the number of scale/zero-point values, causing dequantization errors.
- **Fix:** Replace floor division with ceiling division (`cdiv`). Also fixed `load_row_parallel_weight` to compute group-aware `start_idx` for tensor-parallel shard alignment.
- **Note:** The original P91 patch target (`gptq_marlin.py`) was restructured. The fix was re-implemented for the new codebase locations.

### PN61 — qwen3_vl NVFP4 Loader Guard

- **File:** `vllm/_genesis/pn61_guard.py`
- **Problem:** NVFP4/compressed-tensors quantized qwen3_vl checkpoints often have ViT tower weights stripped. The vLLM loader hits a `KeyError` when trying to load missing ViT weights.
- **Fix:** Runtime wrapper around `Qwen3VLForConditionalGeneration.load_weights`. Pre-emptively detects NVFP4 quantization and sets `language_model_only=True` before loading. Falls back to catching `KeyError` on ViT-named keys.
- **Type:** Class-rebind wrapper (applied via `bootstrap.py`), not a text patch.

### PN62 — ViT Scratch Skip

- **File:** `vllm/_genesis/pn62_guard.py`
- **Problem:** When running text-only qwen3_vl models with NVFP4 quantization, the GPUModelRunner still allocates ViT scratch memory, wasting 3-5 GiB.
- **Fix:** Runtime wrapper around `GPUModelRunner._dummy_run`. Detects text-only mode (`--language-model-only` + `mm_limits_all_zero`) and sets `_pn62_skip_vit_scratch=True` marker.
- **Caveat:** The marker is set but no production vLLM code reads it yet. A ViT-alloc hook is needed to actually skip the allocation. This is a preparation step.
- **Type:** Class-rebind wrapper (applied via `bootstrap.py`).

### P83 — MTP Keep-Last-Cached-Block

- **File:** `vllm/v1/core/single_type_kv_cache_manager.py`
- **Problem:** The `pop()` call on computed blocks is overly conservative for MTP drafters. MTP reads KV cache directly (not pre-materialized hidden states), so dropping the last block is unnecessary and reduces hit rates.
- **Fix:** Guard the `pop()` with `GENESIS_ENABLE_P83=1` env var. Applies to both `FullAttentionManager` and `SlidingWindowManager` sites.
- **Warning:** Only enable for MTP. Do NOT enable for true Eagle/Eagle3 — those drafters genuinely need the block drop.

---

## Genesis Module

The `vllm/_genesis/` directory contains:

- **29 Triton kernels** — custom GPU kernels for TurboQuant, multi-query, and other optimizations
- **123 patch wiring files** — the genesis patch system
- **CLI tools** — `genesis doctor`, `genesis init`
- **Tests and middleware** — compatibility detection, guards, and dispatcher

The module is imported via `vllm._genesis.bootstrap` at startup, which auto-applies PN61 and PN62 wrappers.

---

## Skipped Patches

| Patch | Reason |
|-------|--------|
| PN13 | Already fixed upstream — lambda arity fix in `cuda_graph.py` |
| PN54 | `gdn_attn.py` has zero `.contiguous()` calls; upstream restructure removed redundant copies |
| P95 | `max_cudagraph_capture_size` config already caps batch sizes in `flashinfer.py` |
| P100 | `get_cudagraph_support` returns `UNIFORM_BATCH` natively; upstream moved away from PIECEWISE |
| P38b | Superseded by P67 hook — compile-safety concern already covered |
| P34 | Qwen3 tool_call token handling already present in upstream |

---

## Bootstrap

`vllm/_genesis/bootstrap.py` is auto-executed on import and applies:

1. **PN61** — wraps `Qwen3VLForConditionalGeneration.load_weights`
2. **PN62** — wraps `GPUModelRunner._dummy_run`
3. **P83** — text-patch already applied directly to source

---

## Code Review Bot Analysis (PR #43555)

When this branch was accidentally submitted as PR #43555 against `vllm-project/vllm`, the automated code review bot (`gemini-code-assist`) flagged 4 critical issues in `vllm/_genesis/kernels/block_verify_sampler.py` at lines 137, 282, 351, and 436, claiming the cumulative sum tensor indexing was incorrect.

**Verdict: Bot's critique is incorrect.**

The bot assumed `cu_num_draft_tokens` is shape `[batch_size + 1]` starting with 0 (the `cu_seq_lens` pattern). In reality, in this file it is shape `[batch_size]` — a plain cumulative count array `[draft_0, draft_0+draft_1, ..., total]`. The code correctly constructs `cu_start` by prepending 0 and dropping the last element:

```python
cu_start = torch.empty_like(cu_num_draft_tokens)
cu_start[0] = 0
cu_start[1:] = cu_num_draft_tokens[:-1]
num_draft_per_batch = cu_num_draft_tokens - cu_start
```

The Triton kernel indexing (`start_idx = 0 if req_idx == 0 else tl.load(ptr + req_idx - 1)`) is the correct pattern for this data layout. The bot's suggested fix (`ptr[req_idx]` / `ptr[req_idx + 1]`) would only work for arrays that start with 0 and have `batch_size + 1` elements — which is NOT the case here.

**No changes needed.** The bot confused two different cumulative sum conventions used across vLLM.

---

## Notes

- Patches P3, P27, P61, P65, P67, P91, and P83 are **text patches** applied directly to vLLM source files.
- Patches PN61 and PN62 are **runtime wrappers** applied via class-rebind at import time.
- The `_genesis` module is a drop-in Python package that must be in the vLLM source tree for P67 and the bootstrap system to work.
