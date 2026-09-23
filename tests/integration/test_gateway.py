from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from optimizer.api import create_app
from optimizer.config import (
    AppConfig,
    ModelPrice,
    OutputPoliciesConfig,
    OutputPolicyConfig,
    PricingConfig,
    ProviderConfig,
    ProvidersConfig,
    RateLimitConfig,
    RemoteAuthenticationConfig,
    TelemetryConfig,
)
from optimizer.routing import FEATURE_NAMES, ROUTE_NAMES
from optimizer.telemetry import TelemetryStore


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class ReadFailingCache:
    enabled = True

    def initialize(self) -> None:
        pass

    def get(self, key: str, *, now: float | None = None):
        raise sqlite3.OperationalError("simulated cache read failure")

    def set(self, *args, **kwargs) -> None:
        raise AssertionError("write should be disabled after a read failure")

    def delete(self, key: str) -> None:
        pass

    def clear(self) -> int:
        return 0


def build_test_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        providers=ProvidersConfig(
            openai=ProviderConfig(
                base_url="https://openai.test/v1",
                api_key_env="UNUSED_OPENAI_KEY",
                allowed_hosts=["openai.test"],
            ),
            anthropic=ProviderConfig(
                base_url="https://anthropic.test/v1",
                api_key_env="UNUSED_ANTHROPIC_KEY",
                allowed_hosts=["anthropic.test"],
            ),
        ),
        telemetry=TelemetryConfig(database_path=tmp_path / "telemetry.db"),
        pricing=PricingConfig(
            models={
                "test-model": ModelPrice(input_per_million=2, output_per_million=8),
            }
        ),
    )


def write_decision_artifact(
    path: Path,
    *,
    preferred: str = "frontier",
    samples: int = 500,
    accuracy: float = 0.92,
    tied: bool = False,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "validation": {"samples": samples, "accuracy": accuracy},
                "routes": {
                    route: {
                        "bias": 0 if tied else (10 if route == preferred else -10),
                        "weights": {name: 0 for name in FEATURE_NAMES},
                    }
                    for route in ROUTE_NAMES
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_openai_non_streaming_is_transparent_and_observed(tmp_path: Path) -> None:
    incoming_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "private prompt"}],
    }
    upstream_body = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200, content=json.dumps(upstream_body).encode(), headers={"x-request-id": "upstream-id"}
        )

    config = build_test_config(tmp_path)
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=incoming_body,
                headers={"authorization": "Bearer caller-secret"},
            )

    assert response.status_code == 200
    assert response.json() == upstream_body
    assert response.headers["x-request-id"] == "upstream-id"
    assert response.headers["x-optimizer-request-id"]
    assert captured[0].url == "https://openai.test/v1/chat/completions"
    assert json.loads(captured[0].content) == incoming_body
    stats = store.stats()
    assert stats.requests == 1
    assert stats.input_tokens == 10
    assert stats.output_tokens == 4
    assert str(stats.actual_cost_usd) == "0.000052"


@pytest.mark.asyncio
async def test_anthropic_non_streaming_is_transparent_and_observed(tmp_path: Path) -> None:
    upstream_body = {
        "id": "msg_test",
        "type": "message",
        "usage": {"input_tokens": 7, "output_tokens": 2},
        "content": [{"type": "text", "text": "answer"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://anthropic.test/v1/messages"
        assert request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(200, json=upstream_body)

    config = build_test_config(tmp_path)
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={"model": "test-model", "max_tokens": 32, "messages": []},
                headers={"x-api-key": "caller-secret"},
            )

    assert response.status_code == 200
    assert response.json() == upstream_body
    assert store.stats().input_tokens == 7


@pytest.mark.asyncio
async def test_openai_stream_bytes_are_unchanged_and_usage_is_observed(tmp_path: Path) -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}],"usage":',
        b'{"prompt_tokens":6,"completion_tokens":2}}\n\ndata: [DONE]\n\n',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkStream(chunks),
        )

    config = build_test_config(tmp_path)
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [], "stream": True},
                headers={"authorization": "Bearer caller-secret"},
            )

    assert response.status_code == 200
    assert response.content == b"".join(chunks)
    stats = store.stats()
    assert stats.requests == 1
    assert stats.input_tokens == 6
    assert stats.output_tokens == 2


@pytest.mark.asyncio
async def test_missing_credential_returns_local_401_and_records_rejection(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    store = TelemetryStore(config.telemetry.database_path)
    app = create_app(config, telemetry=store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": []},
        )

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "missing_provider_credential"
    stats = store.stats()
    assert stats.requests == 1
    assert stats.failed_requests == 1


@pytest.mark.asyncio
async def test_unavailable_telemetry_does_not_block_provider_response(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    config = build_test_config(tmp_path)
    # A directory cannot be opened as a SQLite database.
    store = TelemetryStore(tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": []},
                headers={"authorization": "Bearer caller-secret"},
            )

    assert response.status_code == 200
    assert store.enabled is False


@pytest.mark.asyncio
async def test_enabled_input_optimization_is_forwarded_and_audited(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )

    config = build_test_config(tmp_path)
    config.optimization.input.deduplicate_messages = True
    store = TelemetryStore(config.telemetry.database_path)
    messages = [
        {"role": "user", "content": "old duplicate"},
        {"role": "user", "content": "old duplicate"},
        {"role": "assistant", "content": "acknowledged"},
        {"role": "user", "content": "current request"},
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": messages},
                headers={"authorization": "Bearer caller-secret"},
            )

    forwarded_messages = json.loads(captured[0].content)["messages"]
    request_id = response.headers["x-optimizer-request-id"]
    assert response.status_code == 200
    assert response.headers["x-optimizer-events"] == "1"
    assert int(response.headers["x-optimizer-estimated-input-tokens-removed"]) > 0
    assert len(forwarded_messages) == 3
    assert forwarded_messages[-1] == {"role": "user", "content": "current request"}
    assert store.events_for_request(request_id)[0].kind == "duplicate_message_removed"
    assert store.stats().estimated_input_tokens_removed > 0


@pytest.mark.asyncio
async def test_context_intelligence_prunes_before_forwarding_and_audits_scores(
    tmp_path: Path,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )

    config = build_test_config(tmp_path)
    config.optimization.input.max_estimated_tokens = 100
    config.optimization.input.context.enabled = True
    config.optimization.input.context.min_recent_turns = 0
    config.optimization.input.context.max_prunable_score = 1
    store = TelemetryStore(config.telemetry.database_path)
    current = {"role": "user", "content": "Current billing request."}
    messages = [
        {"role": "user", "content": "Old gardening context " + ("mulch " * 200)},
        {"role": "assistant", "content": "Old gardening answer."},
        current,
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": messages},
                headers={"authorization": "Bearer caller-secret"},
            )

    request_id = response.headers["x-optimizer-request-id"]
    forwarded = json.loads(captured[0].content)
    events = store.events_for_request(request_id)

    assert response.status_code == 200
    assert response.headers["x-optimizer-events"] == "1"
    assert int(response.headers["x-optimizer-estimated-input-tokens-removed"]) > 0
    assert forwarded["messages"] == [current]
    assert events[0].kind == "context_chunk_pruned"
    assert "relevance=" in events[0].detail
    assert "gardening" not in events[0].detail


@pytest.mark.asyncio
async def test_validation_escalates_cheap_to_mid_to_frontier_and_tracks_cost(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "cheap-model":
            content = "partial"
            finish_reason = "length"
            usage = {"prompt_tokens": 10, "completion_tokens": 2}
        elif model == "mid-model":
            content = ""
            finish_reason = "stop"
            usage = {"prompt_tokens": 20, "completion_tokens": 1}
        else:
            content = "complete"
            finish_reason = "stop"
            usage = {"prompt_tokens": 30, "completion_tokens": 3}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
                "usage": usage,
            },
        )

    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    config.optimization.routing.models.openai.cheap = "cheap-model"
    config.optimization.routing.models.openai.mid = "mid-model"
    config.optimization.routing.models.openai.frontier = "frontier-model"
    config.optimization.validation.enabled = True
    config.optimization.validation.escalation.enabled = True
    config.pricing.models.update(
        {
            name: ModelPrice(input_per_million=1, output_per_million=2)
            for name in ("cheap-model", "mid-model", "frontier-model")
        }
    )
    store = TelemetryStore(config.telemetry.database_path)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "short request"}],
                },
                headers={"authorization": "Bearer caller-secret"},
            )

    request_id = response.headers["x-optimizer-request-id"]
    attempts = store.validations_for_request(request_id)
    stats = store.stats()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "complete"
    assert calls == ["cheap-model", "mid-model", "frontier-model"]
    assert response.headers["x-optimizer-validation"] == "pass"
    assert response.headers["x-optimizer-escalations"] == "2"
    assert response.headers["x-optimizer-route"] == "frontier"
    assert [attempt.reason for attempt in attempts] == [
        "truncated",
        "empty_response",
        "passed",
    ]
    assert [attempt.model for attempt in attempts] == [
        "cheap-model",
        "mid-model",
        "frontier-model",
    ]
    assert stats.input_tokens == 60
    assert stats.output_tokens == 6
    assert stats.validated_requests == 1
    assert stats.escalated_requests == 1
    assert stats.escalation_attempts == 2
    assert stats.escalation_cost_usd > 0


@pytest.mark.asyncio
async def test_streaming_validation_is_bypassed_without_escalation(tmp_path: Path) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkStream([b"data: [DONE]\n\n"]),
        )

    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    config.optimization.routing.models.openai.cheap = "cheap-model"
    config.optimization.routing.models.openai.mid = "mid-model"
    config.optimization.validation.enabled = True
    config.optimization.validation.escalation.enabled = True

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "stream": True, "messages": []},
                headers={"authorization": "Bearer caller-secret"},
            )

    assert response.status_code == 200
    assert response.headers["x-optimizer-validation"] == "bypass"
    assert response.headers["x-optimizer-validation-reason"] == "streaming_bypass"
    assert calls == 1


@pytest.mark.asyncio
async def test_remote_authentication_and_rate_limiting_protect_provider_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [], "usage": {}})

    monkeypatch.setenv("TEST_OPTIMIZER_KEY", "gateway-secret")
    config = build_test_config(tmp_path)
    config.security.remote_auth.enabled = True
    config.security.remote_auth.api_key_env = "TEST_OPTIMIZER_KEY"
    config.security.rate_limit.enabled = True
    config.security.rate_limit.requests_per_minute = 1
    config.security.rate_limit.burst = 1

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            unauthorized = await client.post("/v1/chat/completions", json={})
            first = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": []},
                headers={
                    "x-optimizer-api-key": "gateway-secret",
                    "authorization": "Bearer provider-secret",
                },
            )
            limited = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": []},
                headers={
                    "x-optimizer-api-key": "gateway-secret",
                    "authorization": "Bearer provider-secret",
                },
            )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["type"] == "invalid_api_key"
    assert first.status_code == 200
    assert first.headers["x-content-type-options"] == "nosniff"
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"
    assert calls == 1


@pytest.mark.asyncio
async def test_ollama_route_uses_local_compatible_provider_without_key(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [], "usage": {}})

    config = build_test_config(tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/ollama/v1/chat/completions",
                json={"model": "llama-local", "messages": []},
            )

    assert response.status_code == 200
    assert captured[0].url == "http://127.0.0.1:11434/v1/chat/completions"
    assert "authorization" not in captured[0].headers


@pytest.mark.asyncio
async def test_request_size_limit_rejects_before_provider_call(tmp_path: Path) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    config = build_test_config(tmp_path)
    config.security.limits.max_request_bytes = 1024
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                content=b"x" * 1025,
                headers={"authorization": "Bearer provider-secret"},
            )

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "request_too_large"
    assert calls == 0


@pytest.mark.asyncio
async def test_chunked_request_stops_reading_at_size_limit(tmp_path: Path) -> None:
    calls = 0
    read_past_limit = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    async def chunks():
        nonlocal read_past_limit
        yield b"a" * 900
        yield b"b" * 200
        read_past_limit = True
        yield b"c" * 100

    config = build_test_config(tmp_path)
    config.security.limits.max_request_bytes = 1024
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                content=chunks(),
                headers={"authorization": "Bearer provider-secret"},
            )

    assert response.status_code == 413
    assert not read_past_limit
    assert calls == 0


@pytest.mark.asyncio
async def test_bad_gateway_keys_cannot_evade_quota_with_untrusted_forwarding_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_GATEWAY_KEY", "correct-secret")
    config = build_test_config(tmp_path)
    config.security.remote_auth = RemoteAuthenticationConfig(
        enabled=True, api_key_env="TEST_GATEWAY_KEY"
    )
    config.security.rate_limit = RateLimitConfig(
        enabled=False,
        authentication_requests_per_minute=1,
        authentication_burst=2,
    )
    app = create_app(config)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
    ) as client:
        responses = [
            await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": []},
                headers={
                    "x-optimizer-api-key": f"guess-{attempt}",
                    "x-optimizer-client-ip": f"192.0.2.{attempt}",
                },
            )
            for attempt in range(1, 4)
        ]

    assert [response.status_code for response in responses] == [401, 401, 429]
    assert responses[-1].headers["retry-after"]


@pytest.mark.asyncio
async def test_output_policy_caps_generation_before_forwarding(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
        )

    config = build_test_config(tmp_path)
    config.optimization.output.enabled = True
    config.optimization.output.policies = OutputPoliciesConfig(
        user_answer=OutputPolicyConfig(max_tokens=64)
    )
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "answer concisely"}],
                    "max_completion_tokens": 500,
                },
                headers={
                    "authorization": "Bearer caller-secret",
                    "x-optimizer-output-policy": "user_answer",
                },
            )

    forwarded = json.loads(captured[0].content)
    request_id = response.headers["x-optimizer-request-id"]
    assert response.status_code == 200
    assert forwarded["max_completion_tokens"] == 64
    assert "x-optimizer-output-policy" not in captured[0].headers
    assert response.headers["x-optimizer-output-policy"] == "user_answer"
    assert response.headers["x-optimizer-output-limit"] == "64"
    assert store.events_for_request(request_id)[0].kind == "output_token_limit_capped"


@pytest.mark.asyncio
async def test_invalid_output_policy_is_rejected_before_provider_call(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": []})

    config = build_test_config(tmp_path)
    config.optimization.output.enabled = True
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": []},
                headers={
                    "authorization": "Bearer caller-secret",
                    "x-optimizer-output-policy": "unknown",
                },
            )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_output_policy"
    assert calls == 0
    assert store.stats().failed_requests == 1


@pytest.mark.asyncio
async def test_anthropic_output_policy_caps_max_tokens(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "type": "message",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    config = build_test_config(tmp_path)
    config.optimization.output.enabled = True
    config.optimization.output.policies = OutputPoliciesConfig(
        tool_call=OutputPolicyConfig(max_tokens=48)
    )
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={"model": "test-model", "messages": [], "max_tokens": 500},
                headers={
                    "x-api-key": "caller-secret",
                    "x-optimizer-output-policy": "tool_call",
                },
            )

    forwarded = json.loads(captured[0].content)
    assert response.status_code == 200
    assert forwarded["max_tokens"] == 48
    assert response.headers["x-optimizer-output-policy"] == "tool_call"


@pytest.mark.asyncio
async def test_disabled_router_preserves_request_and_ignores_route_header(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [], "usage": {}})

    config = build_test_config(tmp_path)
    store = TelemetryStore(config.telemetry.database_path)
    body = {"model": "caller-model", "messages": [{"role": "user", "content": "hello"}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={
                    "authorization": "Bearer caller-secret",
                    "x-optimizer-route": "not-a-route",
                },
            )

    assert response.status_code == 200
    assert json.loads(captured[0].content) == body
    assert "x-optimizer-route" not in response.headers
    assert store.stats().routing_decisions == 0


@pytest.mark.asyncio
async def test_rule_router_rewrites_openai_model_and_records_training_data(
    tmp_path: Path,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    config.optimization.routing.models.openai.cheap = "cheap-model"
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "caller-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers={"authorization": "Bearer caller-secret"},
            )

    assert response.status_code == 200
    assert json.loads(captured[0].content)["model"] == "cheap-model"
    assert response.headers["x-optimizer-route"] == "cheap"
    assert response.headers["x-optimizer-route-source"] == "heuristic"
    assert response.headers["x-optimizer-route-applied"] == "true"
    assert response.headers["x-optimizer-routed-model"] == "cheap-model"
    assert "x-optimizer-route" not in captured[0].headers
    request_id = response.headers["x-optimizer-request-id"]
    routing = store.routing_for_request(request_id)
    assert routing is not None
    assert routing.selected_route == "cheap"
    assert routing.executed_route == "cheap"
    assert routing.original_model == "caller-model"
    assert routing.routed_model == "cheap-model"
    assert routing.reason == "small_simple_request"
    assert routing.message_count == 1
    assert store.stats().routing_decisions == 1


@pytest.mark.asyncio
async def test_invalid_route_is_rejected_and_unconfigured_route_falls_back(
    tmp_path: Path,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [], "usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    body = {"model": "caller-model", "messages": [{"role": "user", "content": "hello"}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            invalid = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={
                    "authorization": "Bearer caller-secret",
                    "x-optimizer-route": "fastest",
                },
            )
            fallback = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={"authorization": "Bearer caller-secret"},
            )

    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_route"
    assert len(captured) == 1
    assert json.loads(captured[0].content)["model"] == "caller-model"
    assert fallback.headers["x-optimizer-route"] == "cheap"
    assert fallback.headers["x-optimizer-route-applied"] == "false"


@pytest.mark.asyncio
async def test_tool_route_rewrites_anthropic_model(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"content": [], "usage": {"input_tokens": 1, "output_tokens": 1}},
        )

    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    config.optimization.routing.models.anthropic.tool = "tool-model"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": "caller-model",
                    "max_tokens": 64,
                    "tools": [{"name": "lookup", "input_schema": {"type": "object"}}],
                    "messages": [{"role": "user", "content": "look it up"}],
                },
                headers={"x-api-key": "caller-secret"},
            )

    assert response.status_code == 200
    assert json.loads(captured[0].content)["model"] == "tool-model"
    assert response.headers["x-optimizer-route"] == "tool"


@pytest.mark.asyncio
async def test_cache_only_route_returns_miss_without_provider_call(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [], "usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    config.optimization.cache.exact.enabled = True
    config.optimization.cache.exact.database_path = tmp_path / "cache.db"
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "caller-model",
                    "messages": [{"role": "user", "content": "uncached"}],
                },
                headers={
                    "authorization": "Bearer caller-secret",
                    "x-optimizer-route": "cache",
                },
            )

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "cache_miss"
    assert response.headers["x-optimizer-route"] == "cache"
    assert response.headers["x-optimizer-route-applied"] == "false"
    assert calls == 0
    routing = store.routing_for_request(response.headers["x-optimizer-request-id"])
    assert routing is not None
    assert routing.selected_route == "cache"
    assert routing.executed_route is None


@pytest.mark.asyncio
async def test_cache_hit_records_cache_as_executed_route(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"call": calls, "usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    config.optimization.routing.models.openai.cheap = "cheap-model"
    config.optimization.cache.exact.enabled = True
    config.optimization.cache.exact.database_path = tmp_path / "cache.db"
    store = TelemetryStore(config.telemetry.database_path)
    body = {"model": "caller-model", "messages": [{"role": "user", "content": "same"}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={"authorization": "Bearer caller-secret"},
            )
            second = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={"authorization": "Bearer caller-secret"},
            )
            cache_only = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={
                    "authorization": "Bearer caller-secret",
                    "x-optimizer-route": "cache",
                },
            )

    assert first.headers["x-optimizer-route"] == "cheap"
    assert second.headers["x-optimizer-cache"] == "HIT"
    assert second.headers["x-optimizer-route"] == "cache"
    assert second.headers["x-optimizer-route-decision"] == "cheap"
    assert cache_only.headers["x-optimizer-cache"] == "HIT"
    assert cache_only.headers["x-optimizer-route"] == "cache"
    assert cache_only.json()["call"] == 1
    assert calls == 1
    routing = store.routing_for_request(second.headers["x-optimizer-request-id"])
    assert routing is not None
    assert routing.selected_route == "cheap"
    assert routing.executed_route == "cache"


@pytest.mark.asyncio
async def test_shadow_decision_model_does_not_change_rule_execution(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [], "usage": {}})

    artifact = write_decision_artifact(tmp_path / "decision-model.json")
    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    config.optimization.routing.models.openai.cheap = "cheap-model"
    config.optimization.routing.models.openai.frontier = "frontier-model"
    config.optimization.routing.decision_model.enabled = True
    config.optimization.routing.decision_model.mode = "shadow"
    config.optimization.routing.decision_model.artifact_path = artifact
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "caller-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers={"authorization": "Bearer caller-secret"},
            )

    assert json.loads(captured[0].content)["model"] == "cheap-model"
    assert response.headers["x-optimizer-route"] == "cheap"
    assert response.headers["x-optimizer-decision-model-route"] == "frontier"
    assert response.headers["x-optimizer-decision-model-mode"] == "shadow"
    assert response.headers["x-optimizer-decision-model-applied"] == "false"
    routing = store.routing_for_request(response.headers["x-optimizer-request-id"])
    assert routing is not None
    assert routing.rule_route == "cheap"
    assert routing.model_route == "frontier"
    assert routing.model_agreed is False


@pytest.mark.asyncio
async def test_active_decision_model_routes_only_with_evidence_and_confidence(
    tmp_path: Path,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [], "usage": {}})

    artifact = write_decision_artifact(tmp_path / "decision-model.json")
    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    config.optimization.routing.models.openai.cheap = "cheap-model"
    config.optimization.routing.models.openai.frontier = "frontier-model"
    config.optimization.routing.decision_model.enabled = True
    config.optimization.routing.decision_model.mode = "active"
    config.optimization.routing.decision_model.artifact_path = artifact
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "caller-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers={"authorization": "Bearer caller-secret"},
            )

    assert json.loads(captured[0].content)["model"] == "frontier-model"
    assert response.headers["x-optimizer-route"] == "frontier"
    assert response.headers["x-optimizer-route-source"] == "decision_model"
    assert response.headers["x-optimizer-decision-model-applied"] == "true"
    routing = store.routing_for_request(response.headers["x-optimizer-request-id"])
    assert routing is not None
    assert routing.rule_route == "cheap"
    assert routing.model_route == "frontier"
    assert routing.model_applied
    assert store.stats().active_model_routes == 1


@pytest.mark.asyncio
async def test_active_decision_model_falls_back_without_validation_evidence(
    tmp_path: Path,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [], "usage": {}})

    artifact = write_decision_artifact(tmp_path / "decision-model.json", samples=10)
    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    config.optimization.routing.models.openai.cheap = "cheap-model"
    config.optimization.routing.models.openai.frontier = "frontier-model"
    config.optimization.routing.decision_model.enabled = True
    config.optimization.routing.decision_model.mode = "active"
    config.optimization.routing.decision_model.artifact_path = artifact
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "caller-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers={"authorization": "Bearer caller-secret"},
            )

    assert json.loads(captured[0].content)["model"] == "cheap-model"
    assert response.headers["x-optimizer-route"] == "cheap"
    assert (
        response.headers["x-optimizer-decision-model-fallback"]
        == "insufficient_validation_evidence"
    )


@pytest.mark.asyncio
async def test_invalid_decision_model_artifact_falls_back_to_rules(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [], "usage": {}})

    artifact = tmp_path / "invalid-model.json"
    artifact.write_text("not-json", encoding="utf-8")
    config = build_test_config(tmp_path)
    config.optimization.routing.enabled = True
    config.optimization.routing.models.openai.cheap = "cheap-model"
    config.optimization.routing.decision_model.enabled = True
    config.optimization.routing.decision_model.mode = "active"
    config.optimization.routing.decision_model.artifact_path = artifact
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "caller-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers={"authorization": "Bearer caller-secret"},
            )

    assert response.status_code == 200
    assert json.loads(captured[0].content)["model"] == "cheap-model"
    assert response.headers["x-optimizer-route"] == "cheap"
    assert "x-optimizer-decision-model-route" not in response.headers


@pytest.mark.asyncio
async def test_exact_cache_hit_skips_provider_and_records_savings(tmp_path: Path) -> None:
    calls = 0
    upstream_body = {
        "id": "chatcmpl-cached",
        "choices": [{"message": {"role": "assistant", "content": "cached answer"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=upstream_body)

    config = build_test_config(tmp_path)
    config.optimization.cache.exact.enabled = True
    config.optimization.cache.exact.database_path = tmp_path / "cache.db"
    store = TelemetryStore(config.telemetry.database_path)
    request_body = {"model": "test-model", "messages": [{"role": "user", "content": "same"}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                json=request_body,
                headers={"authorization": "Bearer caller-secret"},
            )
            second = await client.post(
                "/v1/chat/completions",
                json=request_body,
                headers={"authorization": "Bearer caller-secret"},
            )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == upstream_body
    assert second.content == first.content
    assert first.headers["x-optimizer-cache"] == "MISS"
    assert second.headers["x-optimizer-cache"] == "HIT"
    assert calls == 1
    stats = store.stats()
    assert stats.requests == 2
    assert stats.cache_hits == 1
    assert stats.input_tokens == 10
    assert stats.output_tokens == 4
    assert stats.baseline_cost_usd == Decimal("0.000104")
    assert stats.actual_cost_usd == Decimal("0.000052")
    assert stats.saved_usd == Decimal("0.000052")


@pytest.mark.asyncio
async def test_exact_cache_is_credential_scoped(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"call": calls, "usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.cache.exact.enabled = True
    config.optimization.cache.exact.database_path = tmp_path / "cache.db"
    body = {"model": "test-model", "messages": []}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={"authorization": "Bearer tenant-a"},
            )
            second = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={"authorization": "Bearer tenant-b"},
            )
            third = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={"authorization": "Bearer tenant-a"},
            )

    assert first.json()["call"] == 1
    assert second.json()["call"] == 2
    assert third.json()["call"] == 1
    assert third.headers["x-optimizer-cache"] == "HIT"
    assert calls == 2


@pytest.mark.asyncio
async def test_exact_cache_refresh_replaces_entry(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"version": calls, "usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.cache.exact.enabled = True
    config.optimization.cache.exact.database_path = tmp_path / "cache.db"
    body = {"model": "test-model", "messages": []}
    credential = {"authorization": "Bearer caller-secret"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            first = await client.post("/v1/chat/completions", json=body, headers=credential)
            refreshed = await client.post(
                "/v1/chat/completions",
                json=body,
                headers={**credential, "x-optimizer-cache-control": "refresh"},
            )
            cached = await client.post("/v1/chat/completions", json=body, headers=credential)

    assert first.json()["version"] == 1
    assert refreshed.json()["version"] == 2
    assert refreshed.headers["x-optimizer-cache"] == "REFRESH"
    assert cached.json()["version"] == 2
    assert cached.headers["x-optimizer-cache"] == "HIT"
    assert calls == 2


@pytest.mark.asyncio
async def test_exact_cache_no_store_and_tool_requests_bypass(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"call": calls, "usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.cache.exact.enabled = True
    config.optimization.cache.exact.database_path = tmp_path / "cache.db"
    credential = {"authorization": "Bearer caller-secret"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            no_store_first = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": []},
                headers={**credential, "x-optimizer-cache-control": "no-store"},
            )
            no_store_second = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": []},
                headers={**credential, "x-optimizer-cache-control": "no-store"},
            )
            tool_first = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [], "tools": []},
                headers=credential,
            )
            tool_second = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [], "tools": []},
                headers=credential,
            )

    assert no_store_first.headers["x-optimizer-cache"] == "BYPASS"
    assert no_store_second.json()["call"] == 2
    assert tool_first.headers["x-optimizer-cache"] == "BYPASS"
    assert tool_second.json()["call"] == 4
    assert calls == 4


@pytest.mark.asyncio
async def test_exact_cache_rejects_unknown_control_before_provider_call(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.cache.exact.enabled = True
    config.optimization.cache.exact.database_path = tmp_path / "cache.db"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": []},
                headers={
                    "authorization": "Bearer caller-secret",
                    "x-optimizer-cache-control": "reload",
                },
            )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_cache_control"
    assert calls == 0


@pytest.mark.asyncio
async def test_exact_cache_read_failure_fails_open_and_disables_cache(tmp_path: Path) -> None:
    cache = ReadFailingCache()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"ok": True, "usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.cache.exact.enabled = True
    store = TelemetryStore(config.telemetry.database_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(
            config,
            client=upstream_client,
            telemetry=store,
            exact_cache=cache,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": []},
                headers={"authorization": "Bearer caller-secret"},
            )

    assert response.status_code == 200
    assert response.headers["x-optimizer-cache"] == "ERROR"
    assert calls == 1
    assert cache.enabled is False
    request_id = response.headers["x-optimizer-request-id"]
    assert "exact_cache_error" in {event.kind for event in store.events_for_request(request_id)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "content", "content_type"),
    [
        (500, b'{"error":"temporary"}', "application/json"),
        (200, b"not-json", "text/plain"),
    ],
)
async def test_exact_cache_does_not_store_ineligible_responses(
    tmp_path: Path,
    status_code: int,
    content: bytes,
    content_type: str,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status_code,
            content=content,
            headers={"content-type": content_type},
        )

    config = build_test_config(tmp_path)
    config.optimization.cache.exact.enabled = True
    config.optimization.cache.exact.database_path = tmp_path / "cache.db"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            for _ in range(2):
                response = await client.post(
                    "/v1/chat/completions",
                    json={"model": "test-model", "messages": []},
                    headers={"authorization": "Bearer caller-secret"},
                )
                assert response.headers["x-optimizer-cache"] == "MISS"

    assert calls == 2


@pytest.mark.asyncio
async def test_semantic_cache_reuses_paraphrase_and_records_savings(tmp_path: Path) -> None:
    calls = 0
    upstream_body = {
        "id": "chatcmpl-semantic",
        "choices": [{"message": {"role": "assistant", "content": "Use the reset link."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=upstream_body)

    config = build_test_config(tmp_path)
    config.optimization.cache.semantic.enabled = True
    config.optimization.cache.semantic.database_path = tmp_path / "semantic-cache.db"
    store = TelemetryStore(config.telemetry.database_path)
    credential = {"authorization": "Bearer caller-secret"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "How do I reset my password?"}],
                },
                headers=credential,
            )
            second = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "How can I reset my password?"}],
                },
                headers=credential,
            )

    assert first.status_code == 200
    assert first.headers["x-optimizer-cache"] == "MISS"
    assert second.status_code == 200
    assert second.content == first.content
    assert second.headers["x-optimizer-cache"] == "SEMANTIC_HIT"
    assert float(second.headers["x-optimizer-semantic-similarity"]) >= 0.75
    assert calls == 1
    stats = store.stats()
    assert stats.cache_hits == 1
    assert stats.exact_cache_hits == 0
    assert stats.semantic_cache_hits == 1
    assert stats.baseline_cost_usd == Decimal("0.000104")
    assert stats.actual_cost_usd == Decimal("0.000052")


@pytest.mark.asyncio
async def test_anthropic_semantic_cache_reuses_text_block_paraphrase(tmp_path: Path) -> None:
    calls = 0
    upstream_body = {
        "id": "msg_semantic",
        "type": "message",
        "usage": {"input_tokens": 8, "output_tokens": 3},
        "content": [{"type": "text", "text": "Use the reset link."}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=upstream_body)

    config = build_test_config(tmp_path)
    config.optimization.cache.semantic.enabled = True
    config.optimization.cache.semantic.database_path = tmp_path / "semantic-cache.db"
    credential = {"x-api-key": "caller-secret"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            first = await client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 64,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "How do I reset my password?"}],
                        }
                    ],
                },
                headers=credential,
            )
            second = await client.post(
                "/v1/messages",
                json={
                    "model": "test-model",
                    "max_tokens": 64,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "How can I reset my password?"}],
                        }
                    ],
                },
                headers=credential,
            )

    assert first.status_code == 200
    assert second.content == first.content
    assert second.headers["x-optimizer-cache"] == "SEMANTIC_HIT"
    assert calls == 1


@pytest.mark.asyncio
async def test_semantic_cache_is_scoped_by_credential_and_task(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"call": calls, "usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.cache.semantic.enabled = True
    config.optimization.cache.semantic.database_path = tmp_path / "semantic-cache.db"
    first_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "How do I reset my password?"}],
    }
    paraphrase_body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "How can I reset my password?"}],
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                json=first_body,
                headers={
                    "authorization": "Bearer tenant-a",
                    "x-optimizer-semantic-task": "support",
                },
            )
            other_credential = await client.post(
                "/v1/chat/completions",
                json=paraphrase_body,
                headers={
                    "authorization": "Bearer tenant-b",
                    "x-optimizer-semantic-task": "support",
                },
            )
            other_task = await client.post(
                "/v1/chat/completions",
                json=paraphrase_body,
                headers={
                    "authorization": "Bearer tenant-a",
                    "x-optimizer-semantic-task": "billing",
                },
            )
            hit = await client.post(
                "/v1/chat/completions",
                json=paraphrase_body,
                headers={
                    "authorization": "Bearer tenant-a",
                    "x-optimizer-semantic-task": "support",
                },
            )

    assert first.json()["call"] == 1
    assert other_credential.json()["call"] == 2
    assert other_task.json()["call"] == 3
    assert hit.json()["call"] == 1
    assert hit.headers["x-optimizer-cache"] == "SEMANTIC_HIT"
    assert calls == 3


@pytest.mark.asyncio
async def test_freshness_sensitive_request_bypasses_exact_and_semantic_caches(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"call": calls, "usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.cache.exact.enabled = True
    config.optimization.cache.exact.database_path = tmp_path / "exact-cache.db"
    config.optimization.cache.semantic.enabled = True
    config.optimization.cache.semantic.database_path = tmp_path / "semantic-cache.db"
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is the weather today?"}],
    }
    credential = {"authorization": "Bearer caller-secret"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            first = await client.post("/v1/chat/completions", json=body, headers=credential)
            second = await client.post("/v1/chat/completions", json=body, headers=credential)

    assert first.headers["x-optimizer-cache"] == "BYPASS"
    assert second.headers["x-optimizer-cache"] == "BYPASS"
    assert second.json()["call"] == 2
    assert calls == 2


@pytest.mark.asyncio
async def test_invalid_semantic_control_is_rejected_before_provider_call(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.cache.semantic.enabled = True
    config.optimization.cache.semantic.database_path = tmp_path / "semantic-cache.db"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Explain caching."}],
                },
                headers={
                    "authorization": "Bearer caller-secret",
                    "x-optimizer-semantic-freshness": "unsafe",
                },
            )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_semantic_cache_control"
    assert calls == 0


@pytest.mark.asyncio
async def test_tool_output_compression_is_forwarded_observed_and_not_cached(
    tmp_path: Path,
) -> None:
    captured: list[httpx.Request] = []
    output_lines = [f"passing test line {index:03d} with routine output" for index in range(180)]
    output_lines[90] = "ERROR assertion failed at tests/test_auth.py:42:7"
    tool_output = "\n".join(output_lines)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [], "usage": {}})

    config = build_test_config(tmp_path)
    config.optimization.compression.tool_output.enabled = True
    config.optimization.compression.tool_output.min_characters = 1
    config.optimization.cache.exact.enabled = True
    config.optimization.cache.exact.database_path = tmp_path / "cache.db"
    store = TelemetryStore(config.telemetry.database_path)
    request_body = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Run tests."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": "python -m pytest"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": tool_output},
        ],
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client, telemetry=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                json=request_body,
                headers={"authorization": "Bearer caller-secret"},
            )
            second = await client.post(
                "/v1/chat/completions",
                json=request_body,
                headers={"authorization": "Bearer caller-secret"},
            )

    forwarded = json.loads(captured[0].content)
    compressed = forwarded["messages"][-1]["content"]
    assert first.status_code == 200
    assert first.headers["x-optimizer-tool-compression"] == "ultra"
    assert first.headers["x-optimizer-tool-outputs-compressed"] == "1"
    assert first.headers["x-optimizer-cache"] == "BYPASS"
    assert second.headers["x-optimizer-cache"] == "BYPASS"
    assert "x-optimizer-tool-compression" not in captured[0].headers
    assert "ERROR assertion failed at tests/test_auth.py:42:7" in compressed
    assert "mode=ultra; tool=pytest" in compressed
    assert len(compressed) < len(tool_output)
    assert len(captured) == 2
    request_id = first.headers["x-optimizer-request-id"]
    events = store.events_for_request(request_id)
    assert "tool_output_compressed" in {event.kind for event in events}
    assert store.stats().estimated_input_tokens_removed > 0


@pytest.mark.asyncio
async def test_invalid_tool_compression_mode_is_rejected_before_provider_call(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": []})

    config = build_test_config(tmp_path)
    config.optimization.compression.tool_output.enabled = True
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream_client:
        app = create_app(config, client=upstream_client)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://optimizer.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": []},
                headers={
                    "authorization": "Bearer caller-secret",
                    "x-optimizer-tool-compression": "maximum",
                },
            )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_tool_compression_mode"
    assert calls == 0
