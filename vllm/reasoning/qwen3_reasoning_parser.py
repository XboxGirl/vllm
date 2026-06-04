# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tool_parsers.utils import partial_tag_overlap

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


# Regex for extracting nested tool_call blocks (vllm#39055).
_EMBEDDED_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)<\/tool_call>|<tool_call>.*$",
    re.DOTALL,
)


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3/Qwen3.5 model family.

    The Qwen3 model family uses <think>...</think> tokens to denote reasoning
    text. Starting with Qwen3.5, the chat template places <think> in the
    prompt so only </think> appears in the generated output. The model
    provides a strict switch to disable reasoning output via the
    'enable_thinking=False' parameter.

    When thinking is disabled, the template places <think>\\n\\n</think>\\n\\n
    in the prompt. The serving layer detects this via prompt_is_reasoning_end
    and routes deltas as content without calling the streaming parser.

    NOTE: Models up to the 2507 release (e.g., Qwen/Qwen3-235B-A22B-Instruct-2507)
    use an older chat template where the model generates <think> itself.
    This parser handles both styles: if <think> appears in the generated output
    it is stripped before extraction (non-streaming) or skipped (streaming).

    NOTE: Qwen3.5 models may emit <tool_call> inside the thinking block
    without closing </think> first. <tool_call> is treated as an implicit
    end of reasoning, matching the approach in KimiK2ReasoningParser.
    """

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Qwen3 defaults to thinking enabled; only treat output as
        # pure content when the user explicitly disables it.
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

        self._tool_call_tag = "<tool_call>"
        self._tool_call_token_id = self.vocab.get(self._tool_call_tag)
        self._tool_call_end_tag = "</tool_call>"
        self._tool_call_end_token_id = self.vocab.get(self._tool_call_end_tag)

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    @staticmethod
    def _split_embedded_tool_calls(
        reasoning: str | None,
        content: str | None,
    ) -> tuple[str | None, str | None]:
        """Extract nested tool_call XML blocks out of reasoning content.

        Qwen3.5/3.6 models can emit XML tool calls before the thinking end
        tag. The downstream tool parser only inspects content, so embedded
        tool calls would otherwise be lost (vllm#39055).
        """
        if not reasoning or "<tool_call" not in reasoning or "<function=" not in reasoning:
            return reasoning, content

        extracted_blocks: list[str] = []

        def _collect_or_keep(match):
            block = match.group(0)
            if "<function=" not in block:
                return block
            extracted_blocks.append(block.strip())
            return ""

        remaining_reasoning = _EMBEDDED_TOOL_CALL_RE.sub(_collect_or_keep, reasoning)
        remaining_reasoning = remaining_reasoning.strip() or None

        if not extracted_blocks:
            return reasoning, content

        content_parts = ["\n\n".join(extracted_blocks)]
        if content:
            content_parts.append(content)
        merged_content = "\n\n".join(part for part in content_parts if part) or None
        return remaining_reasoning, merged_content

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        tool_call_token_id = self._tool_call_token_id
        tool_call_end_token_id = self._tool_call_end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[i]
            if token_id == start_token_id:
                # Found <think> before </think> or <tool_call>
                return False
            if token_id == end_token_id:
                return True
            if tool_call_token_id is not None and token_id == tool_call_token_id:
                # Only treat as implicit reasoning end if this <tool_call>
                # is NOT followed by </tool_call>.  Paired occurrences are
                # template examples in the prompt, not model output.
                if tool_call_end_token_id is not None and any(
                    input_ids[j] == tool_call_end_token_id
                    for j in range(i + 1, len(input_ids))
                ):
                    continue
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if super().is_reasoning_end_streaming(input_ids, delta_ids):
            return True
        if self._tool_call_token_id is not None:
            return self._tool_call_token_id in delta_ids
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        result = super().extract_content_ids(input_ids)
        if result:
            return result
        # Fall back: content starts at FIRST <tool_call> (implicit reasoning end).
        # Use first occurrence so multi-tool requests with multiple <tool_call> blocks
        # preserve all tool calls (vllm#40783).
        if (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in input_ids
        ):
            tool_call_index = input_ids.index(self._tool_call_token_id)
            return input_ids[tool_call_index:]
        return []

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        The <think> token is placed in the prompt by the chat template,
        so typically only </think> appears in the generated output.
        If <think> is present (e.g. from a different template), it is
        stripped before extraction.

        When thinking is explicitly disabled and no </think> appears,
        returns (None, model_output) — all output is content.
        Otherwise (thinking enabled, default), a missing </think> means
        the output was truncated and everything is reasoning:
        returns (model_output, None).

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Strip <think> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            return self._split_embedded_tool_calls(reasoning, content or None)

        if not self.thinking_enabled:
            # Thinking explicitly disabled — treat everything as content.
            return None, model_output

        # No </think> — check for implicit reasoning end via <tool_call>.
        tool_call_index = model_output.find(self._tool_call_tag)
        if tool_call_index != -1:
            reasoning = model_output[:tool_call_index]
            content = model_output[tool_call_index:]
            return self._split_embedded_tool_calls(reasoning or None, content or None)
        # Thinking enabled but no </think>: output was truncated.
        # Everything generated so far is reasoning.
        return self._split_embedded_tool_calls(model_output, None)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a streaming delta.

        Since <think> is placed in the prompt by the chat template, all
        generated tokens before </think> are reasoning and tokens after
        are content.

        NOTE: When thinking is disabled, no think tokens appear in the
        generated output. The serving layer detects this via
        prompt_is_reasoning_end and routes deltas as content without
        calling this method.
        """
        # Strip <think> from delta if present (old template / edge case
        # where the model generates <think> itself).
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]

        if self.end_token_id in delta_token_ids:
            # End token in this delta: split reasoning from content.
            end_index = delta_text.find(self.end_token)
            if end_index >= 0:
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )
            # end_token_id in IDs but not in text (already stripped)
            return None

        # Implicit reasoning end via <tool_call>.
        if (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in delta_token_ids
        ):
            tool_index = delta_text.find(self._tool_call_tag)
            if tool_index >= 0:
                reasoning = delta_text[:tool_index]
                content = delta_text[tool_index:]
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )

        # No end token in this delta.
        if not delta_text:
            # Nothing left after stripping start token.
            return None
        elif self.end_token_id in previous_token_ids:
            # End token already passed: everything is content now.
            return DeltaMessage(content=delta_text)
        elif (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in previous_token_ids
        ):
            return DeltaMessage(content=delta_text)
        else:
            # Partial-tag overlap guard: avoid emitting half-formed <tool_call>
            # as reasoning if the tag is being assembled across deltas.
            overlap = partial_tag_overlap(current_text, self._tool_call_tag)
            if overlap > 0:
                send_len = len(delta_text) - overlap
                if send_len > 0:
                    return DeltaMessage(reasoning=delta_text[:send_len])
                return DeltaMessage()
            # No end token yet: still in reasoning phase.
            return DeltaMessage(reasoning=delta_text)
