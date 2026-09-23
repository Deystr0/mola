from __future__ import annotations

from optimizer.providers.base import HTTPProvider


class OllamaProvider(HTTPProvider):
    name = "ollama"
    endpoint = "chat/completions"
    forwarded_request_headers = frozenset({"accept", "authorization"})
    cache_vary_headers = frozenset({"accept", "authorization"})

    def _add_credential(self, headers: dict[str, str]) -> None:
        if "authorization" in headers:
            return
        credential = self._environment_credential()
        if credential:
            headers["authorization"] = f"Bearer {credential}"
