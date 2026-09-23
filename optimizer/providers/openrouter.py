from __future__ import annotations

from optimizer.providers.base import HTTPProvider, MissingCredentialError


class OpenRouterProvider(HTTPProvider):
    name = "openrouter"
    endpoint = "chat/completions"
    forwarded_request_headers = frozenset(
        {
            "accept",
            "authorization",
            "http-referer",
            "x-title",
            "idempotency-key",
        }
    )
    cache_vary_headers = frozenset({"accept", "authorization", "http-referer", "x-title"})

    def _add_credential(self, headers: dict[str, str]) -> None:
        if "authorization" in headers:
            return
        credential = self._environment_credential()
        if credential:
            headers["authorization"] = f"Bearer {credential}"
            return
        raise MissingCredentialError(
            f"OpenRouter credential missing: send Authorization or set {self.config.api_key_env}."
        )
