from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from optimizer.context import OptimizationEvent


class CacheControlError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CachePlan:
    mode: Literal["disabled", "bypass", "lookup", "refresh"]
    read: bool
    write: bool
    reason: str
    event: OptimizationEvent | None


def plan_exact_cache(
    *,
    enabled: bool,
    body: bytes,
    streaming: bool,
    cache_control: str | None,
) -> CachePlan:
    if not enabled:
        return CachePlan("disabled", False, False, "disabled", None)
    if cache_control not in {None, "no-store", "refresh"}:
        raise CacheControlError(
            "Unknown cache control. Expected 'no-store', 'refresh', or no header."
        )
    if cache_control == "no-store":
        return _bypass("no_store")
    if streaming:
        return _bypass("streaming")

    payload = _json_object(body)
    if payload is None:
        return _bypass("invalid_json")
    if _has_tool_context(payload):
        return _bypass("tool_bearing_request")
    if cache_control == "refresh":
        return CachePlan(
            mode="refresh",
            read=False,
            write=True,
            reason="refresh",
            event=_cache_event("exact_cache_refresh", "refresh"),
        )
    return CachePlan("lookup", True, True, "eligible", None)


def _bypass(reason: str) -> CachePlan:
    return CachePlan(
        mode="bypass",
        read=False,
        write=False,
        reason=reason,
        event=_cache_event("exact_cache_bypassed", reason),
    )


def _cache_event(kind: str, reason: str) -> OptimizationEvent:
    return OptimizationEvent(
        kind=kind,
        path="$",
        before_estimated_tokens=0,
        after_estimated_tokens=0,
        detail=f"reason={reason}",
    )


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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
