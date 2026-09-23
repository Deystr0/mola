from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from optimizer.config import ProviderConfig


class MissingCredentialError(ValueError):
    pass


@dataclass(slots=True)
class UpstreamResponse:
    response: httpx.Response
    streaming: bool


class Provider(Protocol):
    name: str

    @property
    def cache_namespace(self) -> str: ...

    def cache_identity(self, incoming_headers: httpx.Headers) -> str: ...

    async def send(
        self,
        *,
        body: bytes,
        incoming_headers: httpx.Headers,
        streaming: bool,
    ) -> UpstreamResponse: ...


class HTTPProvider:
    name: str
    endpoint: str
    forwarded_request_headers: frozenset[str]
    cache_vary_headers: frozenset[str]

    def __init__(self, config: ProviderConfig, client: httpx.AsyncClient) -> None:
        self.config = config
        self.client = client

    @property
    def cache_namespace(self) -> str:
        return f"{self.name}:{self.config.base_url}/{self.endpoint}"

    def cache_identity(self, incoming_headers: httpx.Headers) -> str:
        headers = self._build_headers(incoming_headers)
        values = {name: headers.get(name, "") for name in sorted(self.cache_vary_headers)}
        serialized = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(serialized).hexdigest()

    async def send(
        self,
        *,
        body: bytes,
        incoming_headers: httpx.Headers,
        streaming: bool,
    ) -> UpstreamResponse:
        request = self.client.build_request(
            "POST",
            f"{self.config.base_url}/{self.endpoint}",
            content=body,
            headers=self._build_headers(incoming_headers),
            timeout=self.config.timeout_seconds,
        )
        response = await self.client.send(request, stream=streaming)
        return UpstreamResponse(response=response, streaming=streaming)

    def _build_headers(self, incoming: httpx.Headers) -> dict[str, str]:
        headers = {
            key: value
            for key, value in incoming.multi_items()
            if key.lower() in self.forwarded_request_headers
        }
        self._add_credential(headers)
        headers["content-type"] = "application/json"
        # Keep response bodies observable without relying on provider-specific decompression.
        headers["accept-encoding"] = "identity"
        return headers

    def _add_credential(self, headers: dict[str, str]) -> None:
        raise NotImplementedError

    def _environment_credential(self) -> str | None:
        return os.environ.get(self.config.api_key_env)


RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        "content-type",
        "cache-control",
        "retry-after",
        "request-id",
        "x-request-id",
        "openai-processing-ms",
        "openai-version",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "anthropic-ratelimit-requests-limit",
        "anthropic-ratelimit-requests-remaining",
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-tokens-limit",
        "anthropic-ratelimit-tokens-remaining",
        "anthropic-ratelimit-tokens-reset",
    }
)


def response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.multi_items()
        if key.lower() in RESPONSE_HEADER_ALLOWLIST
    }
