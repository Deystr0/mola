from __future__ import annotations

import httpx

from optimizer.config import ProviderConfig
from optimizer.providers.base import HTTPProvider, MissingCredentialError


class OpenAIProvider(HTTPProvider):
    name = "openai"
    endpoint = "chat/completions"
    forwarded_request_headers = frozenset(
        {"accept", "authorization", "openai-organization", "openai-project", "idempotency-key"}
    )
    cache_vary_headers = frozenset(
        {"accept", "authorization", "openai-organization", "openai-project"}
    )

    def __init__(self, config: ProviderConfig, client: httpx.AsyncClient) -> None:
        super().__init__(config, client)

    def _add_credential(self, headers: dict[str, str]) -> None:
        if "authorization" in headers:
            return
        credential = self._environment_credential()
        if credential:
            headers["authorization"] = f"Bearer {credential}"
            return
        raise MissingCredentialError(
            f"OpenAI credential missing: send Authorization or set {self.config.api_key_env}."
        )
