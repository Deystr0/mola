from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from optimizer.cache.policy import CacheControlError
from optimizer.context import OptimizationEvent

_TASK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_FRESHNESS_RE = re.compile(
    r"(?i)\b(?:today|tonight|tomorrow|yesterday|now|current(?:ly)?|latest|recent|breaking|live|"
    r"news|weather|forecast|price|exchange rate|stock|score|schedule|availability|in stock|"
    r"release date|this (?:week|month|year))\b"
)
_QUERY_SENTINEL = "__LLM_OPTIMIZER_SEMANTIC_QUERY__"


class SemanticCacheControlError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SemanticRequest:
    query: str
    compatibility_body: bytes
    task: str


@dataclass(frozen=True, slots=True)
class SemanticCachePlan:
    mode: Literal["disabled", "bypass", "lookup", "refresh"]
    read: bool
    write: bool
    reason: str
    request: SemanticRequest | None
    event: OptimizationEvent | None


def plan_semantic_cache(
    *,
    enabled: bool,
    body: bytes,
    streaming: bool,
    cache_control: str | None,
    freshness: str | None,
    task: str | None,
) -> SemanticCachePlan:
    if not enabled:
        return SemanticCachePlan("disabled", False, False, "disabled", None, None)
    if cache_control not in {None, "no-store", "refresh"}:
        raise CacheControlError(
            "Unknown cache control. Expected 'no-store', 'refresh', or no header."
        )
    if freshness not in {None, "auto", "stable", "sensitive"}:
        raise SemanticCacheControlError(
            "Unknown semantic freshness mode. Expected 'auto', 'stable', or 'sensitive'."
        )
    if task is not None and not _TASK_RE.fullmatch(task):
        raise SemanticCacheControlError(
            "Invalid semantic task. Use 1-128 letters, numbers, dots, colons, slashes, "
            "underscores, or hyphens."
        )
    if cache_control == "no-store":
        return _bypass("no_store")
    if streaming:
        return _bypass("streaming")

    prepared = _prepare_request(body, task or "")
    if prepared is None:
        return _bypass("incompatible_request")
    if freshness == "sensitive" or _FRESHNESS_RE.search(prepared.query):
        return _bypass("freshness_sensitive")
    if cache_control == "refresh":
        return SemanticCachePlan(
            mode="refresh",
            read=False,
            write=True,
            reason="refresh",
            request=prepared,
            event=_event("semantic_cache_refresh", "refresh"),
        )
    return SemanticCachePlan("lookup", True, True, "eligible", prepared, None)


def _prepare_request(body: bytes, task: str) -> SemanticRequest | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or _has_tool_context(payload):
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        return None

    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        extracted = _extract_and_replace_query(message)
        if extracted is None:
            continue
        query = extracted.strip()
        if not query:
            return None
        compatibility_body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return SemanticRequest(query=query, compatibility_body=compatibility_body, task=task)
    return None


def _extract_and_replace_query(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = _QUERY_SENTINEL
        return content
    if not isinstance(content, list) or not content:
        return None
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            return None
        text = block.get("text")
        if not isinstance(text, str):
            return None
        text_parts.append(text)
    for block in content:
        block["text"] = _QUERY_SENTINEL
    return "\n".join(text_parts)


def _has_tool_context(payload: dict[str, Any]) -> bool:
    if any(field in payload for field in ("tools", "functions", "tool_choice")):
        return True
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" or "tool_calls" in message:
            return True
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result"}
            for block in content
        ):
            return True
    return False


def _bypass(reason: str) -> SemanticCachePlan:
    return SemanticCachePlan(
        mode="bypass",
        read=False,
        write=False,
        reason=reason,
        request=None,
        event=_event("semantic_cache_bypassed", reason),
    )


def _event(kind: str, reason: str) -> OptimizationEvent:
    return OptimizationEvent(
        kind=kind,
        path="$",
        before_estimated_tokens=0,
        after_estimated_tokens=0,
        detail=f"reason={reason}",
    )
