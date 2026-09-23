from __future__ import annotations

import json

import pytest

from optimizer.config import ProviderRouteModels, RoutingModelsConfig, RoutingOptimizationConfig
from optimizer.routing import RouterControlError, RuleRouter


def _router(**overrides) -> RuleRouter:
    config = RoutingOptimizationConfig(
        enabled=True,
        cheap_max_estimated_tokens=120,
        frontier_min_estimated_tokens=500,
        frontier_min_messages=8,
        models=RoutingModelsConfig(
            openai=ProviderRouteModels(
                local="local-model",
                tool="tool-model",
                cheap="cheap-model",
                mid="mid-model",
                frontier="frontier-model",
            )
        ),
        **overrides,
    )
    return RuleRouter(config)


def _body(content: str, **extra) -> bytes:
    return json.dumps(
        {
            "model": "caller-model",
            "messages": [{"role": "user", "content": content}],
            **extra,
        }
    ).encode()


@pytest.mark.parametrize(
    ("requested", "expected_model"),
    [
        ("local", "local-model"),
        ("tool", "tool-model"),
        ("cheap", "cheap-model"),
        ("mid", "mid-model"),
        ("frontier", "frontier-model"),
    ],
)
def test_explicit_routes_rewrite_to_configured_model(
    requested: str,
    expected_model: str,
) -> None:
    decision = _router().route(
        provider="openai",
        body=_body("hello"),
        requested_route=requested,
    )

    assert decision.route == requested
    assert decision.source == "explicit"
    assert decision.model_configured
    assert decision.model_changed
    assert decision.routed_model == expected_model
    assert json.loads(decision.body)["model"] == expected_model


def test_explicit_cache_route_uses_rule_model_for_compatible_lookup() -> None:
    body = _body("hello")

    decision = _router().route(
        provider="openai",
        body=body,
        requested_route="cache",
    )

    assert decision.route == "cache"
    assert decision.cache_only
    assert decision.model_configured
    assert decision.rule_route == "cheap"
    assert decision.routed_model == "cheap-model"
    assert json.loads(decision.body)["model"] == "cheap-model"


def test_structural_heuristics_cover_tool_cheap_mid_and_frontier() -> None:
    router = _router()
    tool = router.route(
        provider="openai",
        body=_body("Use a tool", tools=[{"type": "function", "function": {"name": "x"}}]),
        requested_route=None,
    )
    cheap = router.route(provider="openai", body=_body("short"), requested_route=None)
    mid = router.route(provider="openai", body=_body("m" * 700), requested_route=None)
    frontier = router.route(
        provider="openai",
        body=_body("f" * 2200),
        requested_route=None,
    )

    assert (tool.route, tool.reason) == ("tool", "tool_context")
    assert (cheap.route, cheap.reason) == ("cheap", "small_simple_request")
    assert (mid.route, mid.reason) == ("mid", "default")
    assert (frontier.route, frontier.reason) == ("frontier", "input_size")


def test_media_and_long_conversation_route_to_frontier() -> None:
    router = _router()
    media = router.route(
        provider="openai",
        body=json.dumps(
            {
                "model": "caller-model",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "data:"}}],
                    }
                ],
            }
        ).encode(),
        requested_route=None,
    )
    conversation = router.route(
        provider="openai",
        body=json.dumps(
            {
                "model": "caller-model",
                "messages": [{"role": "user", "content": "x"} for _ in range(8)],
            }
        ).encode(),
        requested_route=None,
    )

    assert (media.route, media.reason) == ("frontier", "media_content")
    assert (conversation.route, conversation.reason) == ("frontier", "conversation_length")


def test_unconfigured_route_preserves_caller_model_and_records_fallback() -> None:
    router = RuleRouter(RoutingOptimizationConfig(enabled=True))
    body = _body("short")

    decision = router.route(provider="openai", body=body, requested_route=None)

    assert decision.route == "cheap"
    assert not decision.model_configured
    assert not decision.model_changed
    assert decision.routed_model == "caller-model"
    assert decision.body == body
    assert decision.reason.endswith("model_unconfigured")


def test_disabled_router_is_transparent_and_ignores_control() -> None:
    body = _body("short")
    decision = RuleRouter(RoutingOptimizationConfig()).route(
        provider="openai",
        body=body,
        requested_route="unknown",
    )

    assert not decision.enabled
    assert decision.route is None
    assert decision.body == body


def test_unknown_explicit_route_is_rejected() -> None:
    with pytest.raises(RouterControlError):
        _router().route(
            provider="openai",
            body=_body("short"),
            requested_route="fastest",
        )


def test_malformed_request_never_claims_configured_model_was_applied() -> None:
    decision = _router().route(
        provider="openai",
        body=b"not-json",
        requested_route="cheap",
    )

    assert decision.route == "cheap"
    assert decision.model_configured
    assert not decision.model_changed
    assert decision.original_model is None
    assert decision.routed_model is None
    assert decision.body == b"not-json"
