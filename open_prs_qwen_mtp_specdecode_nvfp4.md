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
| P65 | TurboQuant spec-decode cudagraph downgrade | Opt-in | GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE |
| P66 | cudagraph_capture_sizes spec-decode divisibility filter | Opt-in | GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER |
| P67 | TurboQuant multi-query kernel for spec-decode K+1 | Opt-in | GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL |
| P83 | MTP keep-last-cached-block (vllm#38182 symptom) | Research | GENESIS_ENABLE_P83 |
| ~~P84~~ | ~~hash_block_size override (vllm#38182 root cause)~~ | ~~Already upstream as `hash_block_size` config option~~ | — |
| P85 | Hybrid fine-shadow prefix cache (MambaManager fix) | Research | GENESIS_ENABLE_P85 |
| ~~PN102~~ | ~~Unified spec-decode metadata + disagreement tracker~~ | ~~Requires Genesis kernel infrastructure~~ | — |

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
| PN77 | FP8 lm_head compression (BF16 to FP8 e4m3) | Opt-in | GENESIS_ENABLE_PN77_FP8_LM_HEAD |

**Note**: PN62 superseded by upstream `MultiModalConfig.skip_mm_profiling` config option.

## Qwen Tool Calls (9 total)

### Genesis-Original Patches (9)

| ID | Title | Status | Env Flag |
|---|---|---|---|
| P15 | Qwen3 None/null tool arg parser | **MERGED** (ac2b0f2d6) | — |
| ~~P12~~ | ~~Qwen3 `` implicit reasoning end~~ | ~~Default ON~~ | ~~—~~ |
| ~~P27~~ | ~~Qwen3 BEFORE-THINK fallback~~ | ~~Default ON~~ | ~~—~~ |
| ~~P29~~ | ~~tool parser IndexError guard~~ | ~~Default ON~~ | ~~—~~ |
| ~~P61c~~ | ~~Qwen3Coder deferred-commit until `<function=` header~~ | ~~Opt-in~~ | ~~GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT~~ |
| ~~P68~~ | ~~Auto force tool_choice=required for long-context~~ | ~~Opt-in~~ | ~~GENESIS_ENABLE_P68_AUTO_FORCE_TOOL~~ |
| ~~P69~~ | ~~Long-context tool-format reminder injection~~ | ~~Opt-in~~ | ~~GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER~~ |
| ~~PN70~~ | ~~Tool schema subset filter~~ | ~~Opt-in~~ | ~~GENESIS_ENABLE_PN70_TOOL_SCHEMA_FILTER~~ |
| ~~PN72~~ | ~~Frequency-based ngram draft post-filter~~ | ~~Opt-in~~ | ~~GENESIS_ENABLE_PN72_FREQUENCY_NGRAM_DRAFTER~~ |

**Notes**:
- P12/P27/P29: Superseded by new parser architecture (vLLM >=0.23.0)
- P61c/P64/PN56: Version-gated (<0.23.0), qwen3coder_tool_parser.py removed in dev148-era engine
- P68/P69: Require Genesis middleware infrastructure (not standalone-applicable)
- PN70/PN72: Require Genesis kernel modules (not standalone-applicable)

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

**Active Genesis patches** (standalone-applicable): P15, PN61 (2 of 19)
**Struck-off**: Version-gated (<0.23.0), superseded by upstream, or require Genesis infrastructure