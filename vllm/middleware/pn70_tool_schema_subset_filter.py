from __future__ import annotations

import logging
import os

from vllm.middleware.long_ctx_tool_adherence import (
    _scan_schema_for_unsupported_key,
)

log = logging.getLogger("vllm.middleware.pn70_tool_schema_subset_filter")

_ENV_FLAG = "VLLM_TOOL_SCHEMA_SUBSET_FILTER"


def _is_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "").strip().lower() in (
        "1", "true", "yes", "y", "on",
    )


def _tool_is_xgrammar_compat(tool: object) -> bool:
    if isinstance(tool, dict):
        fn = tool.get("function") or {}
        params = fn.get("parameters")
    else:
        fn = getattr(tool, "function", None)
        if fn is None:
            return True
        params = getattr(fn, "parameters", None)
    if params is None:
        return True
    try:
        return _scan_schema_for_unsupported_key(params) is None
    except Exception:
        return True


def _tool_name(tool: object) -> str:
    if isinstance(tool, dict):
        return (tool.get("function") or {}).get("name") or "<anonymous>"
    fn = getattr(tool, "function", None)
    if fn is None:
        return "<anonymous>"
    return getattr(fn, "name", None) or "<anonymous>"


def _wrap_get_json_schema_from_tools(original):
    def wrapped(tools):
        if not _is_enabled():
            return original(tools)
        if not tools:
            return original(tools)
        try:
            compat = [t for t in tools if _tool_is_xgrammar_compat(t)]
            incompat_count = len(tools) - len(compat)
            if incompat_count == 0:
                return original(tools)
            incompat_names = [
                _tool_name(t) for t in tools if not _tool_is_xgrammar_compat(t)
            ]
            if not compat:
                log.warning(
                    "[vllm] all %d tools have xgrammar-unsupported "
                    "schema keys — returning None (no grammar enforcement). "
                    "Filtered tools: %s. Set "
                    "%s=0 to disable.",
                    len(tools), incompat_names, _ENV_FLAG,
                )
                return None
            log.warning(
                "[vllm] filtered %d/%d xgrammar-incompat tools from "
                "combined `anyOf` schema. Compat subset: %d tool(s). "
                "Filtered: %s. Set %s=0 to disable.",
                incompat_count, len(tools), len(compat), incompat_names,
                _ENV_FLAG,
            )
            return original(compat)
        except Exception as e:
            log.warning(
                "[vllm] tool schema filter raised %s: %s — falling back to "
                "stock _get_json_schema_from_tools behavior",
                type(e).__name__, e,
            )
            return original(tools)
    wrapped.__wrapped__ = original
    wrapped.__pn70_wrapped__ = True
    return wrapped


def apply_tool_schema_filter():
    from vllm.tool_parsers import utils as _u

    if not hasattr(_u, "_get_json_schema_from_tools"):
        return False

    if getattr(_u._get_json_schema_from_tools, "__pn70_wrapped__", False):
        return True

    original = _u._get_json_schema_from_tools
    _u._get_json_schema_from_tools = _wrap_get_json_schema_from_tools(original)
    return True
