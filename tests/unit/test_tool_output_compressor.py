from __future__ import annotations

import json

import pytest

from optimizer.compression import ToolCompressionModeError, ToolOutputCompressor
from optimizer.config import ToolOutputCompressionConfig


def _openai_body(command: str, output: str) -> bytes:
    return json.dumps(
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Run the checks."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": json.dumps({"cmd": command}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": output},
            ],
        },
        indent=2,
    ).encode()


def _long_output(*, signal: str = "ERROR auth failed at src/auth.py:42:7") -> str:
    lines = [f"routine output line {index:03d} with low-value progress" for index in range(180)]
    lines[90] = signal
    return "\n".join(lines) + "\n"


def _compressor(**overrides) -> ToolOutputCompressor:
    return ToolOutputCompressor(
        ToolOutputCompressionConfig(enabled=True, min_characters=1, **overrides)
    )


def _tool_content(body: bytes) -> str:
    return json.loads(body)["messages"][-1]["content"]


def test_ultra_is_default_and_compresses_current_turn_tool_output() -> None:
    body = _openai_body("python -m pytest", _long_output())

    result = _compressor().apply(body=body, requested_mode=None)
    content = _tool_content(result.body)

    assert result.mode == "ultra"
    assert result.compressed_outputs == 1
    assert len(result.body) < len(body)
    assert "ERROR auth failed at src/auth.py:42:7" in content
    assert "[llm-optimizer:" in content
    assert "mode=ultra; tool=pytest" in content
    assert result.events[0].kind == "tool_output_compressed"
    assert "tool=pytest" in result.events[0].detail
    assert result.optimized_estimated_tokens < len(body) // 4


def test_modes_apply_in_increasing_compression_order() -> None:
    body = _openai_body("pytest", _long_output())

    lite = _compressor().apply(body=body, requested_mode="lite")
    full = _compressor().apply(body=body, requested_mode="full")
    ultra = _compressor().apply(body=body, requested_mode="ultra")

    assert len(_tool_content(lite.body)) > len(_tool_content(full.body))
    assert len(_tool_content(full.body)) > len(_tool_content(ultra.body))
    for result in (lite, full, ultra):
        assert "ERROR auth failed at src/auth.py:42:7" in _tool_content(result.body)


@pytest.mark.parametrize(
    ("command", "expected_kind"),
    [
        ("git status --short", "git"),
        ("rg TODO src", "ripgrep"),
        ("python -m pytest", "pytest"),
        ("vendor/bin/phpunit", "phpunit"),
        ("npx jest", "jest"),
        ("npm test", "npm"),
        ("pnpm test", "pnpm"),
        ("npx eslint .", "eslint"),
        ("pnpm exec tsc --noEmit", "typescript"),
        ("vendor/bin/phpstan analyse", "phpstan"),
        ("composer install", "composer"),
    ],
)
def test_supported_command_families_are_detected(command: str, expected_kind: str) -> None:
    result = _compressor().apply(
        body=_openai_body(command, _long_output()),
        requested_mode="ultra",
    )

    assert result.compressed_outputs == 1
    assert f"tool={expected_kind}" in result.events[0].detail


@pytest.mark.parametrize(
    ("command", "signal"),
    [
        ("git merge feature", "CONFLICT (content): Merge conflict in src/auth.py"),
        ("rg secret .", "rg: ./private: Permission denied (os error 13)"),
        ("pytest", "E       AssertionError: expected 2 at tests/test_app.py:42"),
        ("phpunit", "RuntimeException: broken at /app/tests/AuthTest.php:31"),
        ("jest", "    at Object.login (/app/auth.test.js:18:9)"),
        ("npm test", "npm ERR! code ELIFECYCLE"),
        ("pnpm test", "ERR_PNPM_RECURSIVE_RUN_FIRST_FAIL package failed"),
        ("eslint .", "  12:4  warning  Unexpected any  no-explicit-any"),
        ("tsc --noEmit", "src/app.ts(8,14): error TS2322: Type mismatch"),
        ("phpstan analyse", "  42  Call to an undefined method User::login()."),
        ("composer install", "[RuntimeException] Package resolution failed"),
    ],
)
def test_actionable_command_signal_is_never_elided(command: str, signal: str) -> None:
    result = _compressor().apply(
        body=_openai_body(command, _long_output(signal=signal)),
        requested_mode="ultra",
    )

    assert signal in _tool_content(result.body)


def test_anthropic_text_tool_result_is_compressed_without_changing_other_blocks() -> None:
    body = json.dumps(
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Run tests."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "shell",
                            "input": {"command": "pnpm test"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": [
                                {"type": "text", "text": _long_output()},
                                {"type": "image", "source": {"data": "unchanged"}},
                            ],
                        }
                    ],
                },
            ],
        }
    ).encode()

    result = _compressor().apply(body=body, requested_mode="full")
    blocks = json.loads(result.body)["messages"][-1]["content"][0]["content"]

    assert result.compressed_outputs == 1
    assert "mode=full; tool=pnpm" in blocks[0]["text"]
    assert blocks[1] == {"type": "image", "source": {"data": "unchanged"}}


def test_unknown_tool_short_output_and_invalid_json_pass_through() -> None:
    unknown = _openai_body("custom-command", _long_output())
    ambiguous = _openai_body("git diff | rg TODO", _long_output())
    short = _openai_body("pytest", "1 passed\n")

    unknown_result = _compressor().apply(body=unknown, requested_mode="ultra")
    short_result = ToolOutputCompressor(
        ToolOutputCompressionConfig(enabled=True, min_characters=2000)
    ).apply(body=short, requested_mode=None)
    invalid_result = _compressor().apply(body=b"not-json", requested_mode="full")

    assert unknown_result.body == unknown
    assert unknown_result.events == ()
    assert _compressor().apply(body=ambiguous, requested_mode="ultra").body == ambiguous
    assert short_result.body == short
    assert invalid_result.body == b"not-json"


def test_off_bypasses_and_unknown_mode_is_rejected() -> None:
    body = _openai_body("pytest", _long_output())
    compressor = _compressor()

    off = compressor.apply(body=body, requested_mode="off")

    assert off.body == body
    assert off.mode == "off"
    assert off.events[0].kind == "tool_output_compression_bypassed"
    with pytest.raises(ToolCompressionModeError):
        compressor.apply(body=body, requested_mode="maximum")


def test_compression_is_idempotent_and_cleans_terminal_display_noise() -> None:
    noisy = _long_output(signal="\x1b[31mERROR\x1b[0m first\rERROR final at src/a.py:9:2")
    compressor = _compressor()

    first = compressor.apply(body=_openai_body("pytest", noisy), requested_mode="ultra")
    second = compressor.apply(body=first.body, requested_mode="ultra")
    content = _tool_content(first.body)

    assert "\x1b" not in content
    assert "ERROR first" not in content
    assert "ERROR final at src/a.py:9:2" in content
    assert second.body == first.body
    assert second.events == ()
