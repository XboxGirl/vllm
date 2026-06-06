# Genesis Patches Assessment for XPU Branch

> Generated from `Sandermage/genesis-vllm-patches` docs/PATCHES.md
> Branch: `feat/xpu-unified-w4a16` (Intel XPU / vllm-xpu-kernels)
> Also reviewed: `bryanvine/turboquant-xpu` — XPU TurboQuant port (see §6)
>
> **Skip criteria:** NVIDIA-specific, Ampere/Blackwell/SM1XX, consumer-GPU tuned, NVFP4
>
> **IMPORTANT: TurboQuant is NOT CUDA-only.** [bryanvine/turboquant-xpu](https://github.com/bryanvine/turboquant-xpu) demonstrates working end-to-end TQ on Intel Arc Pro B70 (Xe2) via intel-xpu-backend-for-triton (Triton→SPIRV). TQ patches are now re-categorized from SKIP→REVIEW.

---

## REVISIT: TurboQuant Integration (P4, P6, P22, P26, P36, P38, P39a, P44, P98, PN14)
**Previous verdict:** SKIP (assumed TQ = CUDA-only)
**Revised verdict:** REVISIT — TQ works on XPU per `bryanvine/turboquant-xpu` (see §6)
- P4: TQ hybrid model support — **REVIEW** (may need model-arch routing on XPU)
- P6: TQ-aware attention page size — **REVIEW** (check XPU page-size logic)
- P22: TQ shared dequant prealloc — **REVIEW** (memory pool pattern applies)
- P26: TQ prefill output prealloc — **REVIEW**
- P36: TQ shared decode buffers — **REVIEW**
- P38: TQ `_continuation_prefill` persistent workspace — **REVIEW**
- P39a: FLA chunk_scaled_dot_kkt pool — **REVIEW** (FA2 path on XPU?)
- P44: TQ mixed-batch attn_out pool — **REVIEW**
- P98: WorkspaceManager revert — **REVIEW** (workspace lock on XPU?)
- PN14: TQ decode IOOB safe_page_idx clamp — **KEEP** (defensive clamp, platform-agnostic)

## REVISIT: TQ Continuation / Decode (P9, P18b, P20, P32, P33, P65, P67, P78, P101)
**Previous verdict:** SKIP (assumed TQ = CUDA-only)
**Revised verdict:** REVISIT
- P9: TQ WorkspaceManager revert companion — **REVIEW**
- P18b: TQ decode stage1 tuning — **SKIP** (NVIDIA Triton tuning overlay)
- P20: TQ continuation-prefill FP16 rotate — **REVIEW** (continuation path)
- P32/P33: TQ cu_2 + synth_seq_lens preallocs — **REVIEW** (prefill workspace)
- P65: TQ spec-decode cudagraph downgrade — **SKIP** (no CUDA graphs on XPU)
- P67: TQ multi-query kernel for spec-decode K+1 — **REVIEW** (spec-decare logic may apply)
- P78: TQ .tolist() capture-guard — **SKIP** (cudagraph-specific)
- P101: TQ continuation 64-token slicing — **REVIEW** (continuation path)

## SKIP: Kernel Performance — NVIDIA/Ampere/Blackwell (P1, P2, P3, P17, P23, P24, P72, P81, P87, P91, P93, P95, PN64)
**Reason:** P1/P2: FP8 kernel dispatcher (NVIDIA). P3: Ampere FP8 cast. P17/P23/P24: Marlin tuning overlays. P72: CUDA Dynamo workaround. P81/P95/P87: Marlin/CUDA kernels. PN64: SM 12.0 placeholder. P91/P93: AutoRound/AllSpark (NVIDIA kernels). P87 is Marlin W4A16 pad-on-load — **REVIEW** if vllm-xpu-kernels uses Marlin path.

## SKIP: Kernel Performance — Generic (P31, P37)
**Reason:** P31: MoE router softmax (NVIDIA-focused). P37: MoE intermediate cache pool (NVIDIA buffer mgmt).

---

## REVIEW: Spec-decode — Platform-Agnostic

### P60, P60b — GDN+ngram state recovery
**Verdict: KEEP if hybrid models matter on XPU.** Fixes `GatedDeltaNet` state recovery + Triton offset. If you're not running hybrid GDN/Mamba models on XPU, skip.

### P61, P61b — Qwen3 streaming multi-tool + partial-tag overlap
**Verdict: KEEP** — Qwen3 tool-call parser fixes, pure Python. No CUDA dependency.

### P62 — Structured-output spec-decode timing
**Verdict: REVIEW** — vllm#36138 backport. Fixes reasoning→tool_call boundary timing during spec-decode. Platform-agnostic Python fix, but may already be covered by #42875's `is_reasoning_end` fix.

### P64 — Qwen3Coder MTP streaming early-return
**Verdict: KEEP** — vllm#39598 backport. Fixes MTP streaming when Qwen3Coder emits partial tool calls. Pure Python parser fix.

### P66 — cudagraph capture sizes spec-decode divisibility filter
**Verdict: SKIP** — CUDA-specific cudagraph size filtering. XPU may not use same cudagraph path.

### P68, P69 — Long-context tool-call adherence
**Verdict: KEEP** — Platform-agnostic middleware for tool schema enforcement.

### P70 — Auto strict ngram minimum
**Verdict: SKIP** — Spec-decode config enforcement; opinionated, not bug-fix.

### P71 — Block-verify rejection sampler
**Verdict: REVIEW** — vllm#40819. Solid spec-decode fix, but research-oriented. Skip unless you're benchmarking spec-decode on XPU.

### P77 — Adaptive ngram K controller
**Verdict: SKIP** — Research/experimental EMA+hysteresis controller.

### P79b — Async × spec-decode proposer sync
**Verdict: REVIEW** — vllm#40610 backport. GPU sync timing fix. May help XPU async paths but needs testing.

### P82 — SGLang threshold_single OR-clause
**Verdict: SKIP** — SGLang acceptance heuristic port. Changes spec-decode behavior, not a bug fix.

### P83, P84, P85 — MTP keep-last-cached-block, hash_block_size, fine-shadow prefix cache
**Verdict: SKIP** — Research artifacts. P83 disproven. P84/P85 are prefix-cache experiments.

### P86 — ngram batch_propose O(N*K) → O(N+K)
**Verdict: HOLD / LOW-PRIORITY** — vllm#40876 was closed unmerged with `closed-as-slop`; vllm#42044 is unrelated and not a valid superseding fix. The local branch still has the O(N*K) membership-scan pattern, so the optimization is real and platform-agnostic, but it is too small for a standalone PR under vLLM's contribution policy. Only consider if bundled with substantive spec-decode work.

### P94 — Spec-decode prepare_next_token_ids_padded zero-alloc
**Verdict: KEEP** — vllm#41043. Removes GPU→CPU sync in spec-decode hot path. Pure Python allocation fix.

### PN9 — Independent drafter attention backend
**Verdict: SKIP** — Self-retired (upstream merged via `--speculative-config.attention_backend`)

---

## REVIEW: Qwen3 Parser / Structured Output

### P12 — Qwen3 implicit reasoning end
**Verdict: KEEP** — vllm#39055 companion. Qwen3 parser fix for implicit reasoning→tool-call transitions.

### P15 — Qwen3 None/null tool arg parser
**Verdict: KEEP** — Fixes Qwen3 tool calls with None/null arguments. Pure Python.

### P29 — Tool parser IndexError guard
**Verdict: KEEP** — Defensive IndexError handling in tool parsers. Platform-agnostic.

### P59 — Qwen3 reasoning embedded tool_call recovery
**Verdict: KEEP** — vllm#39055. Recovers Qwen3 tool calls embedded in reasoning blocks. Pure Python.

### PN51 — Qwen3 streaming enable_thinking=false content routing
**Verdict: NO ACTION LOCALLY / KEEP IF MISSING** — Fixes Open WebUI/LibreChat/Cline content routing when thinking is disabled. vllm#40816 is closed, but vllm#41561 was closed unmerged, so do not cite it as a merged upstream fix. The equivalent `_in_reasoning_phase` fix is already present in this branch; verify on any target branch before skipping.

### PN56 — Qwen3Coder XML parse-failure fallback
**Verdict: KEEP** — vllm#41466. Fallback for Qwen3Coder XML parse failures / stale `prev_tool_call_arr` state. Pure Python.

### PN66 — Multiturn `</think>` leak fix
**Verdict: SKIP FOR NOW / REVIEW IF REPRODUCED** — vllm#41696 was closed as a draft and not merged. The reported multiturn `</think>` leak is platform-agnostic, but this should not be applied blindly; revisit only with a local repro or a cleaner upstream successor.

### PN67 — thinking_token_budget inverted-bool fix
**Verdict: KEEP** — vllm#41674. 1-line fix for `gpu_input_batch.py`. Platform-agnostic.

### P107 — MTP truncation at reasoning→tool_call boundary
**Verdict: REVIEW** — vllm#41467. Detects MTP truncation during reasoning→tool_call transitions. May overlap with #42006's MTP fixes.

---

## SKIP: Hybrid / GDN / Mamba (P7, P7b, P28, P34, P46, P79, PN11, PN32, PN34, PN50, PN59)
**Reason:** GDN/Mamba patches. P7/P7b/P28 are dual-stream parallelism. P34 is zero-collapse deadlock guard. P46 is gating buffer pool. PN11 is GDN contiguity. PN32 is GDN chunked-prefill OOM. PN50 is GDN proj fusion kernel. PN54/PN59 are GDN OOM mitigations. Hybrid models aren't the focus for XPU work (yet).

---

## REVIEW: Cudagraph / Scheduler / Buffers

### P5, P5b — KV cache page size unification + pad-smaller-to-max
**Verdict: REVIEW** — P5 is KV page size unification. May apply to XPU. P5b is padding env-opt-in. Review if XPU uses same page-size logic.

### P14 — block_table tail zero-fill
**Verdict: REVIEW** — Defensive zero-fill on block table. Platform-agnostic in theory, but may be XPU-specific. Check if XPU block tables have same issue.

### P74 — Auto chunk-clamp via long_prefill_token_threshold
**Verdict: SKIP** — Consumer-GPU safety net for prealloc overflow. Production XPU likely doesn't hit this.

### PN12 — FFN intermediate scratch pool (Cliff 1 fix)
**Verdict: REVIEW** — Closes 138 MiB OOM at 192K context. Platform-agnostic memory pool. May help XPU long-context.

### PN17 — FA2 softmax_lse runtime clamp
**Verdict: REVIEW** — Clamps `max_seqlen_k` from capture path to actual chunk max. Fixes FA2 OOM. If XPU uses flash-attention, this applies. Check `vllm-xpu-kernels` FA path.

### PN19 — Scoped max_split_size_mb during model load
**Verdict: SKIP** — PyTorch 2.10+ load-time allocator fix. NVIDIA-specific PyTorch allocator path.

### PN25 — SiluAndMul forward_native opaque-op pool
**Verdict: REVIEW** — Inductor-opaque registration for FFN intermediate buffer pool. If XPU uses torch.compile for SiluAndMul, this applies.

### PN31 — FA varlen persistent out buffer
**Verdict: REVIEW** — Per-shape persistent `out` buffer for flash_attn_varlen_func. Memory optimization, platform-agnostic.

### PN33 — Spec-decode warmup K-aware
**Verdict: REVIEW / KEEP IF SPEC-DECODE WARMUP OOM MATTERS** — vllm#37521 is open and not merged. The local branch still warms up with one dummy draft token per request, so the K-aware warmup fix is not already present. Platform-agnostic correctness fix; apply only with targeted spec-decode profiling/tests.

---

## SKIP: DFlash (PN21, PN22, PN23, PN24)
**Reason:** DFlash-specific spec-decode patches. DFlash isn't enabled on XPU. All DFlash patches can be skipped.

## SKIP: TurboQuant Unified Pack (PN26, PN26b)
**Reason:** The sparse-V tile-skip unified-pack path remains NVIDIA-oriented and unverified on XPU. Do not skip because “TQ is CUDA-only” — TurboQuant itself works on XPU; only this specific pack needs separate XPU validation.

## SKIP: Community-Issue Fixes — NVFP4/Consumer (PN38, PN61, PN62, PN63)
**Reason:** PN38 is DFlash quant drafter (skip per DFlash). PN61 is NVFP4 Qwen3VL loader (NVIDIA-only). PN62 is text-only ViT scratch skip (NVIDIA memory). PN63 is FP8_e5m2 Blackwell advisory.

## SKIP: MoE/Numerics (PN27, PN28, PN29)
**Reason:** PN27 reverts MoE interface (NVIDIA MoE path). PN28 is merge_attn_states NaN guard (NVIDIA-focused). PN29 is GDN chunk_o scale-fold (hybrid model skip).

## SKIP: Middleware/UX (PN60, PN65)
**Reason:** PN60 is quant arg vs config.json validator (DX convenience, not bug fix). PN65 is structured API access log middleware (UX/monitoring, not correctness).

---

## SUMMARY

### KEEP / REVIEW after verification
| ID | Description | Upstream Status | Verdict |
|---|---|---|---|
| P61 | Qwen3 multi-tool first-occurrence | vllm#40783 OPEN | ✅ Qwen3 parser fix — still needed |
| P61b | Qwen3 streaming partial-tag overlap | vllm#40783 OPEN | ✅ Companion to P61 — still needed |
| P94 | Spec-decode zero-alloc | vllm#41043 **MERGED** | ✅ Already present locally — no cherry-pick needed |
| P12 | Qwen3 implicit reasoning end | vllm#39055 OPEN | ✅ Qwen3 parser — still needed |
| P15 | Qwen3 None/null tool arg | — (Genesis-original) | ✅ Defensive fix — apply if not already covered by schema coercion |
| P29 | Tool parser IndexError guard | — (Genesis-original) | ✅ Defensive — apply if missing |
| P59 | Qwen3 reasoning embedded tool_call | vllm#39055 OPEN | ✅ Companion to P12 — still needed |
| P64 | Qwen3Coder MTP streaming early-return | vllm#39598 OPEN | ✅ MTP parser — still needed |
| P68/P69 | Long-context tool adherence | — (Genesis-original) | ✅ Middleware — apply after policy/behavior review |
| PN51 | Qwen3 streaming thinking=false | vllm#40816 CLOSED; vllm#41561 CLOSED unmerged | ✅ Equivalent fix appears present locally — no action on this branch |
| PN56 | Qwen3Coder XML parse fallback / `prev_tool_call_arr` fallback | vllm#41466 OPEN draft | ✅ Parser fix — backport/review likely needed |
| PN66 | Multiturn `</think>` leak fix | vllm#41696 CLOSED draft | ❌ **SKIP for now** — revisit only with local repro or successor PR |
| PN67 | Inverted-bool thinking_budget | vllm#41674 **MERGED** | ✅ Already present locally — no cherry-pick needed |
| PN33 | Spec-decode warmup K-aware | vllm#37521 OPEN | ⚠️ Local branch lacks it — review/apply only if spec-decode warmup OOM matters |
| P86 | ngram batch_propose O(N*K) → O(N+K) | vllm#40876 CLOSED unmerged (`closed-as-slop`) | ⚠️ HOLD — valid micro-optimization, but avoid standalone PR |

### REDUCED KEEP (after upstream/local check — only these likely need work)
| ID | Description | Reason |
|---|---|---|
| P61/P61b | Qwen3 multi-tool + streaming overlap | vllm#40783 still OPEN — cherry-pick or wait |
| P12/P59 | Qwen3 reasoning/tool_call embedded recovery | vllm#39055 still OPEN — needed |
| P15 | Qwen3 None/null tool arg | Genesis-original — apply |
| P29 | Tool parser IndexError guard | Genesis-original — apply |
| P64 | Qwen3Coder MTP streaming early-return | vllm#39598 OPEN — needed |
| P68/P69 | Long-context tool adherence | Genesis-original — apply |
| PN56 | Qwen3Coder XML parse fallback | vllm#41466 draft — backport likely needed |
| PN33 | Spec-decode warmup K-aware | vllm#37521 OPEN and local branch lacks K-aware warmup — review if spec-decode is in scope |

### ALREADY IN UPSTREAM OR LOCAL BRANCH (no action needed here)
| ID | Description | Status |
|---|---|---|
| P94 | Spec-decode zero-alloc | ✅ Merged (vllm#41043) and present locally |
| PN67 | Inverted-bool thinking_budget | ✅ Merged (vllm#41674) and present locally |
| PN51 | Qwen3 streaming thinking=false | ✅ Equivalent fix present locally; vllm#41561 itself was closed unmerged |

### HOLD / NO LONGER NEEDED
| ID | Description | Reason |
|---|---|---|
| P86 | ngram batch_propose O(N*K)→O(N+K) | vllm#40876 closed unmerged as `closed-as-slop`; vllm#42044 is unrelated, not a superseder. Bundle only with substantive spec-decode work. |
| PN66 | DelegatingParser multiturn `</think>` leak fix | vllm#41696 closed draft; revisit only with repro or upstream successor. |

### REVIEW (evaluate per XPU stack)
| ID | Description | Notes |
|---|---|---|
| P60/P60b | GDN+ngram state recovery | Only if hybrid GDN/Mamba on XPU |
| P5/P5b | KV cache page size unification | Check XPU page-size logic |
| P14 | block_table tail zero-fill | Check XPU block table |
| P62 | Structured-output spec-decode timing | May overlap with #42875 |
| P79b | Async × spec proposer sync | XPU async path check |
| PN12 | FFN intermediate scratch pool | Long-context OOM fix |
| PN17 | FA2 softmax_lse runtime clamp | Check if XPU uses FA2 |
| PN25 | SiluAndMul opaque-op pool | Check XPU torch.compile path |
| PN31 | FA varlen persistent out buffer | Memory opt, check FA on XPU |
| P107 | MTP truncation detector | May overlap with #42006 |
| PN70 | Tool schema subset filter | Companion to P68, review later |

### SKIP (NVIDIA/CUDA/Consumer-specific)
All remaining patches. Primary skip reasons:
- **TurboQuant:** Do **not** blanket-skip TQ as CUDA-only. TQ works on XPU; review TQ patches individually. Skip only CUDA-graph-specific or NVIDIA-tuned TQ items (for example P65/P78/PN26/PN26b) unless XPU validation says otherwise.
- **NVIDIA/Ampere/Blackwell kernels:** P1/P2/P17/P18/P23/P24/P81/P87/P91/P93/P95/PN19/PN63/PN64
- **Hybrid/GDN/Mamba:** P7/P7b/P28/P34/P46/P79/PN11/PN30/PN32/PN34/PN50/PN54/PN59
- **DFlash:** PN21/PN22/PN23/PN24/PN38
- **Consumer/driver:** P70/P71/P72/P74/P77/P82/P83/P84/P85/PN60/PN61/PN62/PN65/PN9
- **MoE/Numerics:** PN27/PN28/PN29
- **Library/diagnostic:** P51/P79d/P102

### Files to target for cherry-picks
The likely KEEP patches are mostly pure-Python parser/spec-decode fixes that live in:
- `vllm/tool_parsers/` (Qwen3/Gemma2/DeepSeek parser fixes)
- `vllm/parser/` (DelegatingParser, abstract parser infrastructure)
- `vllm/reasoning/` (reasoning boundary detection)
- `vllm/v1/` (spec-decode metadata, warmup, input_batch)

## Appendix A: TurboQuant on XPU — Revised Assessment

**Project:** [bryanvine/turboquant-xpu](https://github.com/bryanvine/turboquant-xpu) — Apache-2.0
**Status:** Working end-to-end on Intel Arc Pro B70 (Xe2 Battlemage G31) via vLLM 0.19.0 + Intel Triton XPU backend
**Upstream:** vLLM PR [#38479](https://github.com/vllm-project/vllm/pull/38479)

### Key findings from `docs/FINDINGS_SUMMARY.md`

| Finding | Implication for XPU branch |
|---|---|
| All 6 TQ Triton kernels compile/run on Intel XPU with **zero modifications** | TQ presets can be enabled on XPU immediately via bind-mount patches |
| `k8v4` preset (FP8 keys + 4-bit values) is **3× faster** than `k3v4_nc` on XPU | Prefer k8v4 on XPU — skips Lloyd-Max centroid gather + WHT rotation GEMM |
| Architecture determines TQ worth: MoE + small uniform head_dim = good | Qwen3-30B-A3B (MoE, head_dim=128): 8.5× KV capacity, 0.47× throughput. Gemma4 (head_dim 256/512): 0.27× throughput on k3v4, 0.62× on k8v4 |
| Memory is NOT the binding constraint past 262K context (RoPE limit) | TQ value: concurrency throughput at moderate context (16-32× concurrent users) |

### TQ patch integration strategy for XPU

1. **Bind-mount approach** (`patches/vllm_mounts/`): Mount TQ patches into vLLM container at runtime. Already works on XPU.
2. **Upstream approach**: Backport Genesis TQ patches (P9, P18b, P20, P22, P26, P32, P33, P36, P38, P39a, P44, P65, P67, P78, P98, PN14) as standalone commits, cherry-picking from upstream vLLM PRs where applicable.
3. **SYCL kernel path**: Long-term native SYCL kernels (`sycl/`) are proposed but not ready. Triton→SPIRV path is the current working integration.

### Which TQ Genesis patches to REVISIT for XPU

| Patch | XPU relevance | Action |
|---|---|---|
| P4 (TQ hybrid model support) | Medium — check if XPU models hit hybrid path | Review against `patches/` mount points |
| P9 (WorkspaceManager revert) | Medium — workspace lock on XPU? | Review + test |
| P101 (continuation slicing) | High — continuation prefill is the hot path | Extract & apply |
| PN14 (decode IOOB clamp) | High — defensive clamp, applies everywhere | Extract & apply |
| P98 (WorkspaceManager revert v2) | Medium — companion to P9 | Review + test |
| P67 (multi-query spec-decode kernel) | Low-Medium — spec-decode on XPU? | Hold for spec-decode plan |
| P20 (continuation-prefill FP16 rotate) | High — continuation path | Extract & apply |

## Appendix B: turboquant-xpu Integration Plan

### Phase 1 — Verify bind-mount TQ on current branch
```bash
# Mount bryanvine/turboquant-xpu patches into vllm-xpu container
# Verify --kv-cache-dtype turboquant_k8v4 works on Gemma4 / Qwen3-30B
docker compose --profile gpu up -d
```

### Phase 2 — Stand up TQ patches from Genesis
1. Extract Genesis TQ logic that's NOT already in `bryanvine/turboquant-xpu`
2. Cherry-pick from upstream vLLM TQ PRs where applicable
3. Test against XPU Triton → SPIRV path

### Phase 3 — SYCL kernel work (future)
- `sycl/` directory contains proposed native SYCL decode kernels
- Blocked on vllm-xpu-kernels issue #271 feasibility review
- Expected 2-3× additional decode speedup over Triton path

### Files in turboquant-xpu to integrate
| Directory | Relevance | Status |
|---|---|---|
| `patches/vllm_mounts/` (14 files recursively; several mount groups) | Container mount points for TQ enablement | ✅ Working |
| `src/turboquant_xpu/kernels/` | Standalone kernel launcher wrappers | 📋 Review |
| `src/turboquant_xpu/quantizer/` | Pure Python, no port needed | ✅ Already works |
| `sycl/` (CMake, .cpp) | Native SYCL decode path | 🚧 Proposed, not merged |
| `tests/` | Kernel + integration tests | 📋 Adopt |
| `docs/XPU_PORTING_ANALYSIS.md` | Kernel-by-kernel XPU risk assessment | 📋 Review for Gemini TQ patches |

---

## Appendix C: TurboQuant XPU Status Update (2026-06)

### Key correction: TurboQuant works on XPU

[bryanvine/turboquant-xpu](https://github.com/bryanvine/turboquant-xpu) demonstrates working end-to-end TQ on Intel Arc Pro B70 (Xe2) via vLLM 0.19.0 + Intel Triton XPU backend. All 6 TQ Triton kernels compile/run on Intel XPU with **zero modifications**.

**Revised verdict for TQ patches:** REVISIT (was: SKIP)
- PN14: TQ decode IOOB safe_page_idx clamp — **KEEP** (defensive clamp, platform-agnostic)
- P101: TQ continuation 64-token slicing — **REVISIT** (continuation path, high relevance)
- P20: TQ continuation-prefill FP16 rotate — **REVISIT**
- P67: TQ multi-query kernel for spec-decode K+1 — **REVISIT**
- P9/P98: WorkspaceManager revert — **REVISIT** (workspace lock on XPU?)
- P65/P78: cudagraph-specific — **SKIP** (no CUDA graphs on XPU)

**TQ integration path:**
1. Phase 1: Mount `bryanvine/turboquant-xpu/patches/` into vllm-xpu container — works today
2. Phase 2: Extract relevant Genesis TQ patches as standalone commits
3. Phase 3: Pursue native SYCL decode kernels (`sycl/` directory) — 2-3× projected speedup

**Recommended preset:** `turboquant_k8v4` (FP8 keys + 4-bit values) — 3× faster than `k3v4_nc` on XPU, skips Lloyd-Max centroid gather + WHT rotation GEMM.

See `docs/XPU_PORTING_ANALYSIS.md` for kernel-by-kernel XPU risk assessment.
