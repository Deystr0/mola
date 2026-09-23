from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


def parse_json_usage(provider: str, payload: Any) -> Usage:
    if not isinstance(payload, dict):
        return Usage()
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    if provider in {"openai", "openrouter", "ollama"}:
        return Usage(
            input_tokens=_non_negative_int(usage.get("prompt_tokens")),
            output_tokens=_non_negative_int(usage.get("completion_tokens")),
        )
    return Usage(
        input_tokens=_non_negative_int(usage.get("input_tokens")),
        output_tokens=_non_negative_int(usage.get("output_tokens")),
    )


class StreamingUsageObserver:
    """Reads complete SSE data lines while leaving streamed bytes untouched."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.usage = Usage()
        self._buffer = b""

    def observe(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._observe_line(line.rstrip(b"\r"))

    def finish(self) -> Usage:
        if self._buffer:
            self._observe_line(self._buffer.rstrip(b"\r"))
            self._buffer = b""
        return self.usage

    def _observe_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line.removeprefix(b"data:").strip()
        if not data or data == b"[DONE]":
            return
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if self.provider in {"openai", "openrouter", "ollama"}:
            parsed = parse_json_usage("openai", payload)
            if parsed.input_tokens or parsed.output_tokens:
                self.usage = parsed
            return
        self._observe_anthropic(payload)

    def _observe_anthropic(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        event_type = payload.get("type")
        if event_type == "message_start":
            message = payload.get("message", {})
            usage = message.get("usage", {}) if isinstance(message, dict) else {}
            if isinstance(usage, dict):
                self.usage.input_tokens = _non_negative_int(usage.get("input_tokens"))
                self.usage.output_tokens = _non_negative_int(usage.get("output_tokens"))
        elif event_type == "message_delta":
            usage = payload.get("usage", {})
            if isinstance(usage, dict):
                self.usage.output_tokens = _non_negative_int(usage.get("output_tokens"))


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
