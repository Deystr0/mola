import json

from optimizer.config import ValidationConfig
from optimizer.validation import ResponseValidator


def encode(value: object) -> bytes:
    return json.dumps(value).encode()


def test_openai_valid_response_passes() -> None:
    result = ResponseValidator(ValidationConfig(enabled=True)).validate(
        provider="openai",
        request_body=encode({"messages": []}),
        status_code=200,
        content_type="application/json",
        response_body=encode(
            {"choices": [{"message": {"content": "complete"}, "finish_reason": "stop"}]}
        ),
        streaming=False,
    )

    assert result.passed is True
    assert result.reason == "passed"


def test_openai_truncation_refusal_empty_and_json_mode_fail() -> None:
    validator = ResponseValidator(ValidationConfig(enabled=True))
    cases = [
        (
            {"messages": []},
            {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
            "truncated",
        ),
        (
            {"messages": []},
            {
                "choices": [
                    {
                        "message": {"content": None, "refusal": "cannot comply"},
                        "finish_reason": "stop",
                    }
                ]
            },
            "refusal",
        ),
        (
            {"messages": []},
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            "empty_response",
        ),
        (
            {"messages": [], "response_format": {"type": "json_object"}},
            {"choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]},
            "invalid_json_mode",
        ),
    ]

    for request, response, reason in cases:
        result = validator.validate(
            provider="openai",
            request_body=encode(request),
            status_code=200,
            content_type="application/json",
            response_body=encode(response),
            streaming=False,
        )
        assert result.passed is False
        assert result.reason == reason


def test_tool_calls_are_valid_non_empty_output() -> None:
    result = ResponseValidator(ValidationConfig(enabled=True)).validate(
        provider="openrouter",
        request_body=encode({"messages": []}),
        status_code=200,
        content_type="application/json",
        response_body=encode(
            {
                "choices": [
                    {
                        "message": {"content": None, "tool_calls": [{"id": "call-1"}]},
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        streaming=False,
    )

    assert result.passed is True


def test_anthropic_truncated_response_fails_and_tool_use_passes() -> None:
    validator = ResponseValidator(ValidationConfig(enabled=True))

    truncated = validator.validate(
        provider="anthropic",
        request_body=b"{}",
        status_code=200,
        content_type="application/json",
        response_body=encode(
            {"content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens"}
        ),
        streaming=False,
    )
    tool_use = validator.validate(
        provider="anthropic",
        request_body=b"{}",
        status_code=200,
        content_type="application/json",
        response_body=encode(
            {"content": [{"type": "tool_use", "name": "lookup"}], "stop_reason": "tool_use"}
        ),
        streaming=False,
    )

    assert truncated.passed is False
    assert truncated.reason == "truncated"
    assert tool_use.passed is True


def test_streaming_is_explicitly_bypassed() -> None:
    result = ResponseValidator(ValidationConfig(enabled=True)).validate(
        provider="openai",
        request_body=b"{}",
        status_code=200,
        content_type="text/event-stream",
        response_body=b"",
        streaming=True,
    )

    assert result.enabled is True
    assert result.passed is None
    assert result.reason == "streaming_bypass"
