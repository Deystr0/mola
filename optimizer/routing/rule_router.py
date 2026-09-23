from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from optimizer.config import RoutingOptimizationConfig
from optimizer.context import OptimizationEvent, estimate_tokens

RouteName = Literal["cache", "tool", "local", "cheap", "mid", "frontier"]
ROUTE_NAMES: tuple[str, ...] = get_args(RouteName)


class RouterControlError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RouteFeatures:
    estimated_input_tokens: int
    message_count: int
    tool_count: int
    has_media: bool
    streaming: bool


@dataclass(frozen=True, slots=True)
class RouteDecision:
    enabled: bool
    body: bytes
    route: RouteName | None
    source: str
    reason: str
    original_model: str | None
    routed_model: str | None
    model_configured: bool
    model_changed: bool
    cache_only: bool
    features: RouteFeatures
    event: OptimizationEvent | None
    rule_route: RouteName | None = None
    model_route: RouteName | None = None
    model_confidence: float | None = None
    model_mode: str | None = None
    model_applied: bool = False
    model_fallback_reason: str | None = None


class RuleRouter:
    """Select an explainable execution class from content-free structural features."""

    def __init__(self, config: RoutingOptimizationConfig) -> None:
        self.config = config

    def route(
        self,
        *,
        provider: str,
        body: bytes,
        requested_route: str | None,
    ) -> RouteDecision:
        features, payload = _features(body)
        original_model = _model(payload)
        if not self.config.enabled:
            return RouteDecision(
                enabled=False,
                body=body,
                route=None,
                source="disabled",
                reason="disabled",
                original_model=original_model,
                routed_model=original_model,
                model_configured=False,
                model_changed=False,
                cache_only=False,
                features=features,
                event=None,
            )

        selected, source, reason = self._select(requested_route, features)
        cache_only = selected == "cache"
        model_route = selected
        rule_route = selected
        if cache_only:
            model_route, _, _ = self._select(None, features)
            rule_route = model_route
        target_model = self._target_model(provider, model_route)
        model_configured = cache_only or target_model is not None
        can_apply_model = target_model is not None and payload is not None
        routed_model = target_model if can_apply_model else original_model
        changed = can_apply_model and target_model != original_model
        routed_body = body
        if changed and payload is not None:
            payload["model"] = target_model
            routed_body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")

        if not cache_only and target_model is None:
            reason = f"{reason};model_unconfigured"
        event = OptimizationEvent(
            kind="route_selected",
            path="model",
            before_estimated_tokens=features.estimated_input_tokens,
            after_estimated_tokens=features.estimated_input_tokens,
            detail=(
                f"route={selected};source={source};reason={reason};"
                f"model_configured={str(model_configured).lower()};"
                f"model_changed={str(changed).lower()}"
            ),
        )
        return RouteDecision(
            enabled=True,
            body=routed_body,
            route=selected,
            source=source,
            reason=reason,
            original_model=original_model,
            routed_model=routed_model,
            model_configured=model_configured,
            model_changed=changed,
            cache_only=cache_only,
            features=features,
            event=event,
            rule_route=rule_route,
        )

    def _select(
        self,
        requested_route: str | None,
        features: RouteFeatures,
    ) -> tuple[RouteName, str, str]:
        if requested_route is not None:
            if requested_route not in ROUTE_NAMES:
                choices = ", ".join(ROUTE_NAMES)
                raise RouterControlError(
                    f"Unknown route '{requested_route}'. Expected one of: {choices}."
                )
            return cast(RouteName, requested_route), "explicit", "request_header"
        if features.tool_count:
            return "tool", "heuristic", "tool_context"
        if features.has_media:
            return "frontier", "heuristic", "media_content"
        if features.estimated_input_tokens >= self.config.frontier_min_estimated_tokens:
            return "frontier", "heuristic", "input_size"
        if features.message_count >= self.config.frontier_min_messages:
            return "frontier", "heuristic", "conversation_length"
        if (
            features.estimated_input_tokens <= self.config.cheap_max_estimated_tokens
            and features.message_count <= 4
        ):
            return "cheap", "heuristic", "small_simple_request"
        return self.config.default_route, "heuristic", "default"

    def _target_model(self, provider: str, route: RouteName) -> str | None:
        if route == "cache" or not hasattr(self.config.models, provider):
            return None
        provider_models = getattr(self.config.models, provider)
        return cast(str | None, getattr(provider_models, route))


def _features(body: bytes) -> tuple[RouteFeatures, dict[str, Any] | None]:
    estimated = estimate_tokens(body)
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = None
    if not isinstance(payload, dict):
        return RouteFeatures(estimated, 0, 0, False, False), None

    raw_messages = payload.get("messages")
    messages = raw_messages if isinstance(raw_messages, list) else []
    top_level_tools = payload.get("tools")
    tool_count = len(top_level_tools) if isinstance(top_level_tools, list) else 0
    legacy_functions = payload.get("functions")
    if isinstance(legacy_functions, list):
        tool_count += len(legacy_functions)
    if "tool_choice" in payload:
        tool_count += 1
    has_media = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            tool_count += len(calls)
        if message.get("role") == "tool":
            tool_count += 1
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in {"tool_use", "tool_result"}:
                tool_count += 1
            elif block_type in {
                "image",
                "image_url",
                "input_image",
                "document",
                "audio",
                "input_audio",
            }:
                has_media = True
    return (
        RouteFeatures(
            estimated_input_tokens=estimated,
            message_count=len(messages),
            tool_count=tool_count,
            has_media=has_media,
            streaming=payload.get("stream") is True,
        ),
        payload,
    )


def _model(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    value = payload.get("model")
    return value if isinstance(value, str) else None
