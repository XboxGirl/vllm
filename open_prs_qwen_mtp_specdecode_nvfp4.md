# Suitable Patches: Qwen Tool Calls, MTP, Speculative Decode, NVFP4

Generated: 2026-06-28
Sources: vllm-project/vllm PRs + genesis-vllm-patches (Genesis-original only)

## MTP / Speculative Decode (15 total)

### Upstream PRs (8)

| PR | Title | Status |
|---|---|---|
| [#44943](https://github.com/vllm-project/vllm/pull/44943) | [Bugfix][MoE][SpecDecode] Fix Qwen3.5/3.6 MTP loader for pre-fused expert checkpoints | Active |
| [#44644](https://github.com/vllm-project/vllm/pull/44644) | [Bugfix][Model] Avoid duplicate Qwen3.5 MTP backbone allocations | Active |
| [#44927](https://github.com/vllm-project/vllm/pull/44927) | fix(structured_output): pass new_token_ids to should_advance() to fix MTP spec-decode off-by-one | Active |
| [#43650](https://github.com/vllm-project/vllm/pull/43650) | [Bugfix][Core] MTP + enable prefix caching + mamba accuracy fix (Qwen3.5) | Active |
| [#45985](https://github.com/vllm-project/vllm/pull/45985) | [Feature] Support Eagle3 speculative decoding with Pipeline Parallelism | Active |
| [#46399](https://github.com/vllm-project/vllm/pull/46399) | [Bugfix][Spec Decode] Disable dynamic speculative decoding for fixed-K proposers | Active |
| [#43310](https://github.com/vllm-project/vllm/pull/43310) | Add per-request speculative decode metrics | Active |
| [#45963](https://github.com/vllm-project/vllm/pull/45963) | Disable dynamic speculative decoding when DP is enabled | Active |

### Genesis-Original Patches (7)

| ID | Title | Status | Env Flag |
|---|---|---|---|
| P65 | TurboQuant spec-decode cudagraph downgrade | **MERGED** (eb7eb8331) | GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE |
| P66 | cudagraph_capture_sizes spec-decode divisibility filter | **MERGED** (ebe26d141) | GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER |
| P67 | TurboQuant multi-query kernel for spec-decode K+1 | **MERGED** (891e8a0e3) | GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL |
| P83 | MTP keep-last-cached-block (vllm#38182 symptom) | **MERGED** (495adef37) | VLLM_MTP_KEEP_LAST_CACHED_BLOCK |
| ~~P84~~ | ~~hash_block_size override (vllm#38182 root cause)~~ | ~~Already upstream as `hash_block_size` config option~~ | — |
| P85 | Hybrid fine-shadow prefix cache (MambaManager fix) | **MERGED** (495adef37) | VLLM_HYBRID_FINE_SHADOW_CACHE |
| ~~PN102~~ | ~~Unified spec-decode metadata + disagreement tracker~~ | ~~Requires Genesis kernel infrastructure~~ | — |

**Notes**:
- P65/P66/P67: Standalone patches, applied directly
- P83: Applied — is_mtp flag threaded through scheduler → KVCacheManager → coordinator
- P85: Applied — fine-shadow cache in MambaManager.cache_blocks + find_longest_cache_hit
- ~~P84~~: ~~retired~~ — upstream-native `--hash-block-size` config option
- ~~PN102~~: ~~skipped~~ — requires Genesis kernel infrastructure

## NVFP4 (12 total)

### Upstream PRs (9)

| PR | Title | Status |
|---|---|---|
| [#45735](https://github.com/vllm-project/vllm/pull/45735) | [Quantization] Extend ModelOpt mixed precision and NVFP4 runtime formats | Active |
| [#45738](https://github.com/vllm-project/vllm/pull/45738) | [NVFP4] Support clamped SwiGLU-OAI on FlashInfer-CUTLASS MoE | Active |
| [#44389](https://github.com/vllm-project/vllm/pull/44389) | [KVCache] Add Triton software NVFP4 KV cache support | Active |
| [#46880](https://github.com/vllm-project/vllm/pull/46880) | [Bugfix][NVFP4 MoE] Pad gated intermediate to 64 for FlashInfer TRT-LLM shuffle | Active |
| [#45187](https://github.com/vllm-project/vllm/pull/45187) | Add NVFP4 KV 4-over-6 scale search | Active |
| [#44851](https://github.com/vllm-project/vllm/pull/44851) | Add SM120 NVFP4 KV cache support | Active |
| [#42984](https://github.com/vllm-project/vllm/pull/42984) | [NVFP4] Fix NVFP4 quant padding for FP4 experts | Active |
| [#46481](https://github.com/vllm-project/vllm/pull/46481) | [Kernel][SM120] NVFP4 grouped MoE: pingpong schedule | Active |
| [#46872](https://github.com/vllm-project/vllm/pull/46872) | Remove redundant _pack_topk_ids_weights_kernel in TrtLLM NvFP4 MoE | Active |

### Genesis-Original Patches (3)

| ID | Title | Status | Env Flag |
|---|---|---|---|
| PN61 | qwen3_vl loader KeyError to text-only auto-fallback | **MERGED** (84855f019) | GENESIS_ENABLE_PN61 |
| ~~PN62~~ | ~~Text-only ViT scratch skip (3-5 GiB on 27B-NVFP4)~~ | ~~Opt-in~~ | ~~GENESIS_ENABLE_PN62~~ |
| PN77 | FP8 lm_head compression (BF16 to FP8 e4m3) | **SKIPPED** — not needed | — |

**Note**: PN62 superseded by upstream `MultiModalConfig.skip_mm_profiling` config option.

## Qwen Tool Calls (9 total)

### Genesis-Original Patches (9)

| ID | Title | Status | Env Flag |
|---|---|---|---|
| P15 | Qwen3 None/null tool arg parser | **MERGED** (ac2b0f2d6) | — |
| ~~P12~~ | ~~Qwen3 `` implicit reasoning end~~ | ~~Already upstream in new parser~~ | — |
| ~~P27~~ | ~~Qwen3 BEFORE-THINK fallback~~ | ~~Old parser architecture superseded~~ | — |
| ~~P29~~ | ~~tool parser IndexError guard~~ | ~~Already upstream in new parser~~ | — |
| ~~P61c~~ | ~~Qwen3Coder deferred-commit until `<function=` header~~ | ~~Version-gated (<0.23.0)~~ | — |
| P68 | Auto force tool_choice=required for long-context | **MERGED** (99be29ff2) | GENESIS_ENABLE_P68_AUTO_FORCE_TOOL |
| P69 | Long-context tool-format reminder injection | **MERGED** (99be29ff2) | GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER |
| PN70 | Tool schema subset filter | **MERGED** (495adef37) | VLLM_TOOL_SCHEMA_SUBSET_FILTER |
| PN72 | Frequency-based ngram draft post-filter | **MERGED** (8d81ba4e7) | GENESIS_ENABLE_PN72_FREQUENCY_NGRAM_DRAFTER |

**Notes**:
- P12/P27/P29: Superseded by new parser architecture (vLLM >=0.23.0)
- P61c/P64/PN56: Version-gated (<0.23.0), qwen3coder_tool_parser.py removed in dev148-era engine
- P68/P69: Applied as standalone middleware (long_ctx_tool_adherence.py)
- PN70: Applied as standalone filter (pn70_tool_schema_subset_filter.py)
- PN72: Applied as standalone filter (ngram_frequency_filter.py)
- **Upstream verification (2026-06-28)**: Checked vLLM GitHub for PRs addressing these issues. Issue #38182 (MTP prefix cache) is OPEN. PR #45614 (Mamba prefix cache EAGLE hit) is OPEN. PR #42904 (xgrammar patternProperties) is OPEN. No merged PRs supersede the Genesis patches.

## Abandoned

| ID | Reason |
|---|---|
| #43012 | 41d draft, 1 comment, 17d idle, CI failures |
| #41396 | 59d old, 2 comments, no maintainer engagement |

## Excluded (not Qwen-relevant)

XPU (#45779), Gemma (#46426, #46443), DeepSeek (#45452, #45149), MiniMax (#46816), GLM (#45144), gpt-oss (#46645), DDTree (#42910), Gumbel benchmark (#46499).

## Counts

| Category | Upstream PRs | Genesis Patches | Total |
|---|---|---|---|
| MTP / Spec Decode | 8 | ~~7~~ | 15 |
| NVFP4 | 9 | ~~3~~ | 12 |
| Qwen Tool Calls | 0 | ~~9~~ | 9 |
| **Total** | **17** | **19** | **36** |

**Active Genesis patches** (standalone-applicable): P15, P65, P66, P67, P68, P69, P83, P85, PN61, PN70, PN72 (11 of 19)
**Struck-off**: Version-gated (<0.23.0), superseded by upstream, or skipped (PN77)