from __future__ import annotations

import pytest

from optimizer.cache import CacheControlError, plan_exact_cache


def test_disabled_cache_does_not_validate_control_header() -> None:
    plan = plan_exact_cache(
        enabled=False,
        body=b"not-json",
        streaming=True,
        cache_control="unknown",
    )

    assert plan.mode == "disabled"
    assert not plan.read
    assert not plan.write
    assert plan.event is None


def test_eligible_request_reads_and_writes() -> None:
    plan = plan_exact_cache(
        enabled=True,
        body=b'{"model":"test","messages":[]}',
        streaming=False,
        cache_control=None,
    )

    assert plan.mode == "lookup"
    assert plan.read
    assert plan.write


@pytest.mark.parametrize(
    ("body", "streaming", "control", "reason"),
    [
        (b"{}", False, "no-store", "no_store"),
        (b"{}", True, None, "streaming"),
        (b"not-json", False, None, "invalid_json"),
        (b"[]", False, None, "invalid_json"),
        (b'{"tools":[]}', False, None, "tool_bearing_request"),
        (b'{"functions":[]}', False, None, "tool_bearing_request"),
        (b'{"tool_choice":"none"}', False, None, "tool_bearing_request"),
        (
            b'{"messages":[{"role":"tool","content":"result"}]}',
            False,
            None,
            "tool_bearing_request",
        ),
        (
            b'{"messages":[{"role":"user","content":[{"type":"tool_result"}]}]}',
            False,
            None,
            "tool_bearing_request",
        ),
    ],
)
def test_unsafe_requests_bypass_cache(
    body: bytes,
    streaming: bool,
    control: str | None,
    reason: str,
) -> None:
    plan = plan_exact_cache(
        enabled=True,
        body=body,
        streaming=streaming,
        cache_control=control,
    )

    assert plan.mode == "bypass"
    assert not plan.read
    assert not plan.write
    assert plan.reason == reason
    assert plan.event is not None
    assert plan.event.kind == "exact_cache_bypassed"


def test_refresh_skips_read_and_allows_write() -> None:
    plan = plan_exact_cache(
        enabled=True,
        body=b"{}",
        streaming=False,
        cache_control="refresh",
    )

    assert plan.mode == "refresh"
    assert not plan.read
    assert plan.write
    assert plan.event is not None
    assert plan.event.kind == "exact_cache_refresh"


def test_unknown_control_is_rejected() -> None:
    with pytest.raises(CacheControlError):
        plan_exact_cache(
            enabled=True,
            body=b"{}",
            streaming=False,
            cache_control="reload",
        )
