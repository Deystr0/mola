from __future__ import annotations

import json

import pytest

from optimizer.cache import CacheControlError, SemanticCacheControlError, plan_semantic_cache


def _body(query: str, *, system: str = "Answer accurately.") -> bytes:
    return json.dumps(
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
        }
    ).encode()


def test_semantic_plan_extracts_only_query_and_keeps_context_compatibility() -> None:
    first = plan_semantic_cache(
        enabled=True,
        body=_body("How do I reset my password?"),
        streaming=False,
        cache_control=None,
        freshness=None,
        task="support.password",
    )
    paraphrase = plan_semantic_cache(
        enabled=True,
        body=_body("How can I reset my password?"),
        streaming=False,
        cache_control=None,
        freshness="stable",
        task="support.password",
    )
    different_context = plan_semantic_cache(
        enabled=True,
        body=_body("How can I reset my password?", system="Answer as JSON."),
        streaming=False,
        cache_control=None,
        freshness=None,
        task="support.password",
    )

    assert first.mode == "lookup"
    assert first.request is not None
    assert paraphrase.request is not None
    assert different_context.request is not None
    assert first.request.query == "How do I reset my password?"
    assert first.request.compatibility_body == paraphrase.request.compatibility_body
    assert first.request.compatibility_body != different_context.request.compatibility_body
    assert b"reset my password" not in first.request.compatibility_body


@pytest.mark.parametrize(
    ("body", "streaming", "control", "freshness", "reason"),
    [
        (_body("Stable documentation question"), True, None, None, "streaming"),
        (_body("Stable documentation question"), False, "no-store", None, "no_store"),
        (_body("What is the weather today?"), False, None, None, "freshness_sensitive"),
        (_body("Stable documentation question"), False, None, "sensitive", "freshness_sensitive"),
        (b"not-json", False, None, None, "incompatible_request"),
        (
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "question"},
                        {"role": "tool", "content": "result"},
                    ]
                }
            ).encode(),
            False,
            None,
            None,
            "incompatible_request",
        ),
    ],
)
def test_semantic_cache_bypass_rules(
    body: bytes,
    streaming: bool,
    control: str | None,
    freshness: str | None,
    reason: str,
) -> None:
    plan = plan_semantic_cache(
        enabled=True,
        body=body,
        streaming=streaming,
        cache_control=control,
        freshness=freshness,
        task=None,
    )

    assert plan.mode == "bypass"
    assert not plan.read
    assert not plan.write
    assert plan.reason == reason
    assert plan.event is not None


def test_semantic_refresh_skips_read_and_allows_write() -> None:
    plan = plan_semantic_cache(
        enabled=True,
        body=_body("Stable documentation question"),
        streaming=False,
        cache_control="refresh",
        freshness=None,
        task=None,
    )

    assert plan.mode == "refresh"
    assert not plan.read
    assert plan.write
    assert plan.request is not None


def test_disabled_semantic_cache_does_not_validate_headers() -> None:
    plan = plan_semantic_cache(
        enabled=False,
        body=b"not-json",
        streaming=True,
        cache_control="unknown",
        freshness="unknown",
        task="invalid task",
    )

    assert plan.mode == "disabled"


def test_invalid_semantic_controls_are_rejected() -> None:
    with pytest.raises(CacheControlError):
        plan_semantic_cache(
            enabled=True,
            body=_body("question"),
            streaming=False,
            cache_control="reload",
            freshness=None,
            task=None,
        )
    with pytest.raises(SemanticCacheControlError):
        plan_semantic_cache(
            enabled=True,
            body=_body("question"),
            streaming=False,
            cache_control=None,
            freshness="fresh",
            task=None,
        )
    with pytest.raises(SemanticCacheControlError):
        plan_semantic_cache(
            enabled=True,
            body=_body("question"),
            streaming=False,
            cache_control=None,
            freshness=None,
            task="invalid task",
        )
