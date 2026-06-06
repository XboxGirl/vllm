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
1. ~~Cherry-pick #42776~~ ✅ Done → `8e93d76`, `2fd6056`
2. ~~Cherry-pick #42006~~ ✅ Done → `b7d9c4c`
3. ~~Cherry-pick #43609~~ ✅ Done → `7cc9387`, `d9e3d82`, `89d8dab`, `19b02bd`
4. ~~Evaluate #42875~~ ✅ Merged — all 4 commits cherry-picked → `bbee763`, `c4cf6aa`, `a47c1e5`, `8d7ee60`

## Conflicts Resolved
- **#42776**: `examples/tool_chat_template_gemma4.jinja` — accepted PR version (multiline multimodal placeholders)
- **#43609**: `vllm/model_executor/models/gemma4_mm.py` — kept both `SupportsQuant` + `SupportsTranscription` imports
- **#42875**: `vllm/reasoning/gemma4_reasoning_parser.py` — kept HEAD's multi-turn reasoning leak fix over simplified version
- **#42875**: `vllm/tool_parsers/gemma4_tool_parser.py` — kept HEAD's MTP delta-split method from #42006

## Notes
- All 4 PRs merged cleanly onto our branch
- #42875 fixes are **complementary** to #42006 (different parser paths: single-delta vs MTP multi-delta)
- Test command after all merges: `uv pip install -e . && uv run pytest tests/tool_parsers/test_gemma4_tool_parser.py tests/renderers/test_gemma4_chat_template.py -v`
