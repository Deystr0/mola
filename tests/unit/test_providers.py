import json

import httpx
import pytest

from optimizer.config import ProviderConfig
from optimizer.providers import (
    AnthropicProvider,
    MissingCredentialError,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)


@pytest.mark.asyncio
async def test_openai_forwards_raw_body_and_allowed_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setenv("TEST_OPENAI_KEY", "environment-secret")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(
            ProviderConfig(
                base_url="https://provider.test/v1/",
                api_key_env="TEST_OPENAI_KEY",
            ),
            client,
        )
        body = json.dumps({"model": "test", "messages": []}, separators=(",", ":")).encode()

        upstream = await provider.send(
            body=body,
            incoming_headers=httpx.Headers(
                {"authorization": "Bearer caller-secret", "x-do-not-forward": "private"}
            ),
            streaming=False,
        )

    request = captured["request"]
    assert isinstance(request, httpx.Request)
    assert request.url == "https://provider.test/v1/chat/completions"
    assert request.content == body
    assert request.headers["authorization"] == "Bearer caller-secret"
    assert request.headers["accept-encoding"] == "identity"
    assert "x-do-not-forward" not in request.headers
    assert upstream.response.status_code == 200


@pytest.mark.asyncio
async def test_anthropic_adds_version_and_environment_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "environment-secret")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            ProviderConfig(
                base_url="https://provider.test/v1",
                api_key_env="TEST_ANTHROPIC_KEY",
            ),
            client,
        )
        await provider.send(body=b"{}", incoming_headers=httpx.Headers(), streaming=False)

    assert captured[0].headers["x-api-key"] == "environment-secret"
    assert captured[0].headers["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_missing_provider_credential_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    ) as client:
        provider = OpenAIProvider(
            ProviderConfig(base_url="https://provider.test/v1", api_key_env="MISSING_KEY"),
            client,
        )
        with pytest.raises(MissingCredentialError):
            await provider.send(body=b"{}", incoming_headers=httpx.Headers(), streaming=False)


@pytest.mark.asyncio
async def test_openai_cache_identity_is_scoped_to_credentials_and_account_headers() -> None:
    async with httpx.AsyncClient() as client:
        provider = OpenAIProvider(
            ProviderConfig(
                base_url="https://provider.test/v1",
                api_key_env="UNUSED_OPENAI_KEY",
            ),
            client,
        )

        first = provider.cache_identity(
            httpx.Headers(
                {
                    "authorization": "Bearer first-secret",
                    "openai-organization": "org-a",
                    "openai-project": "project-a",
                }
            )
        )
        second_credential = provider.cache_identity(
            httpx.Headers(
                {
                    "authorization": "Bearer second-secret",
                    "openai-organization": "org-a",
                    "openai-project": "project-a",
                }
            )
        )
        second_project = provider.cache_identity(
            httpx.Headers(
                {
                    "authorization": "Bearer first-secret",
                    "openai-organization": "org-a",
                    "openai-project": "project-b",
                }
            )
        )

    assert first != second_credential
    assert first != second_project
    assert "first-secret" not in first
    assert provider.cache_namespace == "openai:https://provider.test/v1/chat/completions"


@pytest.mark.asyncio
async def test_anthropic_cache_identity_includes_default_version_and_beta() -> None:
    async with httpx.AsyncClient() as client:
        provider = AnthropicProvider(
            ProviderConfig(
                base_url="https://provider.test/v1",
                api_key_env="UNUSED_ANTHROPIC_KEY",
            ),
            client,
        )

        default = provider.cache_identity(httpx.Headers({"x-api-key": "secret"}))
        explicit_default = provider.cache_identity(
            httpx.Headers({"x-api-key": "secret", "anthropic-version": "2023-06-01"})
        )
        beta = provider.cache_identity(
            httpx.Headers({"x-api-key": "secret", "anthropic-beta": "feature-a"})
        )

    assert default == explicit_default
    assert default != beta


@pytest.mark.asyncio
async def test_openrouter_uses_compatible_endpoint_and_metadata_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []
    monkeypatch.setenv("TEST_OPENROUTER_KEY", "router-secret")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: captured.append(request) or httpx.Response(200, json={})
        )
    ) as client:
        provider = OpenRouterProvider(
            ProviderConfig(
                base_url="https://openrouter.test/api/v1",
                api_key_env="TEST_OPENROUTER_KEY",
            ),
            client,
        )
        await provider.send(
            body=b"{}",
            incoming_headers=httpx.Headers(
                {"http-referer": "https://app.test", "x-title": "Optimizer"}
            ),
            streaming=False,
        )

    assert captured[0].url == "https://openrouter.test/api/v1/chat/completions"
    assert captured[0].headers["authorization"] == "Bearer router-secret"
    assert captured[0].headers["http-referer"] == "https://app.test"
    assert captured[0].headers["x-title"] == "Optimizer"


@pytest.mark.asyncio
async def test_ollama_allows_local_requests_without_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []
    monkeypatch.delenv("UNSET_OLLAMA_KEY", raising=False)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: captured.append(request) or httpx.Response(200, json={})
        )
    ) as client:
        provider = OllamaProvider(
            ProviderConfig(
                base_url="http://ollama.test/v1",
                api_key_env="UNSET_OLLAMA_KEY",
                allow_insecure_http=True,
            ),
            client,
        )
        await provider.send(body=b"{}", incoming_headers=httpx.Headers(), streaming=False)

    assert captured[0].url == "http://ollama.test/v1/chat/completions"
    assert "authorization" not in captured[0].headers
