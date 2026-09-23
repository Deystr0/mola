from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from optimizer.config import ValidationConfig

_OPENAI_COMPATIBLE = frozenset({"openai", "openrouter", "ollama"})


@dataclass(frozen=True, slots=True)
class ValidationResult:
    enabled: bool
    passed: bool | None
    reason: str


class ResponseValidator:
    """Deterministic response validation without an LLM judge."""

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def validate(
        self,
        *,
        provider: str,
        request_body: bytes,
        status_code: int,
        content_type: str,
        response_body: bytes,
        streaming: bool,
    ) -> ValidationResult:
        if not self.config.enabled:
            return ValidationResult(False, None, "disabled")
        if streaming:
            return ValidationResult(True, None, "streaming_bypass")
        if not 200 <= status_code < 300:
            return ValidationResult(True, False, "http_status")
        if "application/json" not in content_type.lower():
            return ValidationResult(True, False, "non_json_response")
        try:
            request = json.loads(request_body)
            response = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ValidationResult(True, False, "invalid_json")
        if not isinstance(request, dict) or not isinstance(response, dict):
            return ValidationResult(True, False, "invalid_shape")
        if provider in _OPENAI_COMPATIBLE:
            return self._validate_openai(request, response)
        if provider == "anthropic":
            return self._validate_anthropic(response)
        return ValidationResult(True, None, "unsupported_provider")

    def _validate_openai(
        self,
        request: dict[str, Any],
        response: dict[str, Any],
    ) -> ValidationResult:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return ValidationResult(True, False, "missing_choice")
        choice = choices[0]
        if self.config.reject_truncated and choice.get("finish_reason") == "length":
            return ValidationResult(True, False, "truncated")
        message = choice.get("message")
        if not isinstance(message, dict):
            return ValidationResult(True, False, "missing_message")
        refusal = message.get("refusal")
        if self.config.reject_refusal and isinstance(refusal, str) and refusal.strip():
            return ValidationResult(True, False, "refusal")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        has_content = isinstance(content, str) and bool(content.strip())
        has_tool_call = isinstance(tool_calls, list) and bool(tool_calls)
        if self.config.require_non_empty and not has_content and not has_tool_call:
            return ValidationResult(True, False, "empty_response")
        if self.config.validate_json_mode and has_content and _requests_json(request):
            try:
                json.loads(content)
            except json.JSONDecodeError:
                return ValidationResult(True, False, "invalid_json_mode")
        return ValidationResult(True, True, "passed")

    def _validate_anthropic(self, response: dict[str, Any]) -> ValidationResult:
        if self.config.reject_truncated and response.get("stop_reason") == "max_tokens":
            return ValidationResult(True, False, "truncated")
        content = response.get("content")
        if not isinstance(content, list):
            return ValidationResult(True, False, "missing_content")
        has_output = any(
            isinstance(block, dict)
            and (
                (block.get("type") == "text" and bool(str(block.get("text", "")).strip()))
                or block.get("type") == "tool_use"
            )
            for block in content
        )
        if self.config.require_non_empty and not has_output:
            return ValidationResult(True, False, "empty_response")
        return ValidationResult(True, True, "passed")


def _requests_json(request: dict[str, Any]) -> bool:
    response_format = request.get("response_format")
    return isinstance(response_format, dict) and response_format.get("type") in {
        "json_object",
        "json_schema",
    }
