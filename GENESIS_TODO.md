# Genesis Patches — Implementation Tracker

> Source: `GENESIS.md` "KEEP / REVIEW after verification" → "REDUCED KEEP"
> Branch: `feat/xpu-unified-w4a16`
> Started: 2026-06-04
> Status: All REDUCED KEEP patches applied

---

## Applied Patches

### Reasoning Parser (`vllm/reasoning/qwen3_reasoning_parser.py`)
- **P12** — Already present: `_tool_call_token_id`, `is_reasoning_end()`, `is_reasoning_end_streaming()`
- **P61** — `extract_content_ids`: LAST → FIRST occurrence (`input_ids.index()`)
- **P61b** — Added `partial_tag_overlap` import and guard in `extract_reasoning_streaming`
- **P59** — Added `import re`, `_EMBEDDED_TOOL_CALL_RE`, `_split_embedded_tool_calls()`; wrapped `extract_reasoning` returns

### Tool Parsers (`vllm/tool_parsers/`)
- **P15** — `utils.py`: `coerce_to_schema_type` now accepts `"none"` in addition to `"null"`
- **P29** — `qwen3coder_tool_parser.py`: Added IndexError guard for `streamed_args_for_tool[current_tool_index]`
- **P64** — `qwen3coder_tool_parser.py`: Removed early return from `json_fragments` block; unified emit with `combined` variable
- **PN56** — `qwen3coder_tool_parser.py`: Added `_pn56_parse_succeeded` flag and fallback for `prev_tool_call_arr["arguments"]`

### Spec-decode (`vllm/v1/worker/gpu_model_runner.py`)
- **PN33** — K-aware warmup: `draft_token_ids` now uses real `num_speculative_tokens` instead of dummy `[0]`

---

## Deferred

- **P68/P69** — Long-context tool adherence middleware: Requires new middleware module; deferred for policy review

---

## Already Present (no action needed)
- **P94** — Spec-decode zero-alloc (vllm#41043 merged)
- **PN51** — Qwen3 streaming thinking=false (equivalent fix present)
- **PN67** — Inverted-bool thinking_budget (vllm#41674 merged)

---

## Verification
All modified files pass `py_compile` syntax check.
