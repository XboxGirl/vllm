# XPU Gemma 4 Tool Calling Fix Plan

## PRs to Merge (in order)

### 1. #42776 — [Bugfix] Gemma 4 Template Content + Tool Rendering
- **Author:** @yzong-rh
- **Changed files:** 2 (+190/-48)
- **Files touched:** `examples/tool_chat_template_gemma4.jinja`, `tests/renderers/test_gemma4_chat_template.py`
- **Summary:** Fixes 4 bugs in the Gemma4 chat template:
  - Content/tool_calls ordering (content before tool_calls in assistant turns)
  - Extra `<|turn|>` marker after tool call + content turns
  - Empty reasoning blocks (`reasoning=""`) being dropped
  - Duplicate `<|tool_response|>` opener for legacy assistant-embedded tool_responses
- **Tests:** 4 regression tests in `tests/renderers/test_gemma4_chat_template.py`
- **Commit SHA:** 0f710694409913b7242ec64b378a1221ebe2b28b (3 commits)

### 2. #42006 — [Bugfix] Fix Gemma4 MTP streaming multi-tool calls
- **Author:** @whytem
- **Changed files:** 2 (+315/-1)
- **Files touched:** `vllm/tool_parsers/gemma4_tool_parser.py`, `tests/tool_parsers/test_gemma4_tool_parser.py`
- **Summary:** Fixes tool-call argument dropping when MTP speculative decoding produces deltas spanning multiple tool-call delimiters. Splits deltas on `<|tool_call|>` / `<|tool_call|>` boundaries before parser state machine.
- **Tests:** 51 tests in `tests/tool_parsers/test_gemma4_tool_parser.py`
- **Commit SHA:** fe026ca4547295936c8bac39b4e0ed8885d07ea7 (1 commit)
- **Note:** Superseded by #42300 (accumulated parser rewrite), but maintainer chose this targeted approach

### 3. #43609 — [Bugfix] Enable audio transcription endpoint for Gemma 4
- **Author:** @SoluMilken
- **Changed files:** 5 (+123/-7)
- **Files touched:** Audio-related + model registry changes for Gemma4 multimodal
- **Summary:** Enables `/v1/audio/transcriptions` and `/v1/audio/translations` endpoints for Gemma4 E2B/E4B models.
- **Tests:** Manual test with wav files
- **Commit SHA:** 24652b44feb55d0287953db5a43c8382871d1862 (4 commits)

## PR to Evaluate: #42875 — [Bugfix] Fix Gemma4 streaming tool calls lost when entire call arrives in one delta
- **Author:** @alexbi29
- **Changed files:** 6 (+768/-47)
- **Files touched:** `vllm/tool_parsers/gemma4_tool_parser.py`, `vllm/reasoning/gemma4_reasoning_parser.py`, `vllm/parser/abstract_parser.py`, tests
- **Summary:** Fixes 5 coupled bugs that cause tool calls to be silently dropped when `--stream-interval` is large enough that the entire tool call arrives in one delta:
  1. `<|tool_response|>` before `<|tool_call|>` in same delta → `is_reasoning_end` returns False
  2. `_extract_streaming` Case 2 skipped when start/end tokens arrive together
  3. `DelegatingParser.parse_delta` drops reasoning field when reasoning ends + tool call begins in same delta
  4. Multi-turn reasoning leak after completed tool-response exchange
  5. Empty `reasoning_content` emitted when thinking block is empty
- **Assessment needed:** Check if these 5 fixes overlap with #42006 or #42300 (the full parser rewrite). If #42006/#42776/#43609 already cover these cases, we can skip #42875.

## Execution Order
1. ~~Apply #42776~~ ✅ Reapplied on `xboxgirl/xpu-unified-w4a16-merged`: accepted the PR template/test content for `examples/tool_chat_template_gemma4.jinja` and `tests/renderers/test_gemma4_chat_template.py`.
2. ~~Apply #42006~~ ✅ Reapplied/adapted: added Gemma4 delimiter-segment replay for MTP-sized deltas crossing tool-call boundaries and targeted parser regressions.
3. ~~Apply #43609~~ ✅ Reapplied/adapted: added `SupportsTranscription` hooks to `vllm/model_executor/models/gemma4_mm.py`, prompt/config tests, and the Speech2Text docs row.
4. ~~Evaluate #42875~~ ✅ Reapplied/adapted: kept #42006's segmented replay while adding single-delta tool-call handling, reasoning-boundary preservation, `<|tool_response>` stop-token handling, and empty-reasoning suppression.

## Conflicts Resolved
- **#42776**: `examples/tool_chat_template_gemma4.jinja` — accepted PR version (multiline multimodal placeholders)
- **#43609**: `vllm/model_executor/models/gemma4_mm.py` — kept both `SupportsQuant` + `SupportsTranscription` imports
- **#42875**: `vllm/reasoning/gemma4_reasoning_parser.py` — kept HEAD's multi-turn reasoning leak fix over simplified version
- **#42875**: `vllm/tool_parsers/gemma4_tool_parser.py` — kept HEAD's MTP delta-split method from #42006

## 2026-06-06 Current Branch Verification
- The old local cherry-pick hashes previously listed here (`8e93d76`, `2fd6056`, `b7d9c4c`, `7cc9387`, `d9e3d82`, `89d8dab`, `19b02bd`, `bbee763`, `c4cf6aa`, `a47c1e5`, `8d7ee60`) were **not present** in this checkout and were not ancestors of `HEAD`.
- The upstream PR commit objects for #42776 (`0f710694...`), #42006 (`fe026ca4...`), #43609 (`24652b44...` / `e7b07213...`), and #42875 (`0629385a...`) were available locally, but also were not ancestors of `HEAD`.
- The fixes were therefore reapplied/adapted directly onto `xboxgirl/xpu-unified-w4a16-merged` instead of relying on the stale TODO status.
- Dependency-free validation run after the reapply: `python -m py_compile` over the modified parser/model/test Python files. Full pytest was attempted but the local `.venv` is missing `pytest`; a template smoke was also attempted but the local `.venv` is missing `jinja2`.

## Notes
- All 4 PRs are now represented in the current branch, but the final implementation was an adapted reapply rather than the stale cherry-pick hashes that were previously listed.
- #42875 fixes are **complementary** to #42006 (different parser paths: single-delta vs MTP multi-delta), so both sets were kept.
- Recommended full validation after installing test dependencies: `uv run pytest tests/tool_parsers/test_gemma4_tool_parser.py tests/reasoning/test_gemma4_reasoning_parser.py tests/renderers/test_gemma4_chat_template.py tests/models/multimodal/processing/test_gemma4.py -v`
