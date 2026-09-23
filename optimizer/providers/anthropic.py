from __future__ import annotations

import httpx

from optimizer.config import ProviderConfig
from optimizer.providers.base import HTTPProvider, MissingCredentialError


class AnthropicProvider(HTTPProvider):
    name = "anthropic"
    endpoint = "messages"
    forwarded_request_headers = frozenset(
        {"accept", "x-api-key", "anthropic-version", "anthropic-beta"}
    )
    cache_vary_headers = frozenset({"accept", "x-api-key", "anthropic-version", "anthropic-beta"})

    def __init__(self, config: ProviderConfig, client: httpx.AsyncClient) -> None:
        super().__init__(config, client)

    def _build_headers(self, incoming: httpx.Headers) -> dict[str, str]:
        headers = super()._build_headers(incoming)
        headers.setdefault("anthropic-version", "2023-06-01")
        return headers

    def _add_credential(self, headers: dict[str, str]) -> None:
        if "x-api-key" in headers:
            return
        credential = self._environment_credential()
        if credential:
            headers["x-api-key"] = credential
            return
        raise MissingCredentialError(
            f"Anthropic credential missing: send x-api-key or set {self.config.api_key_env}."
        )
