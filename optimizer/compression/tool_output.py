from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Literal, cast, get_args

from optimizer.config import ToolCompressionMode, ToolOutputCompressionConfig
from optimizer.context import OptimizationEvent, estimate_tokens

ToolCompressionSelection = ToolCompressionMode | Literal["off"]

_MARKER_PREFIX = "[llm-optimizer:"
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_COMMON_SIGNAL_RE = re.compile(
    r"(?i)(\b(?:error|err|fatal|panic|exception|traceback|fail|failed|failure|warn|warning)\b"
    r"|\bERR_[A-Z0-9_]+\b|[A-Za-z]+Error\b|[A-Za-z]+Exception\b|caused by|\bnot ok\b"
    r"|\bexit(?:ed)?(?: with)?(?: status| code)?\s*[=:]?\s*[1-9]\d*)"
)
_LOCATION_RE = re.compile(
    r"(?:[A-Za-z]:)?[^\s:()]+\.(?:c|cc|cpp|css|go|h|hpp|html|js|jsx|json|php|py|rb|rs|"
    r"sh|sql|ts|tsx|vue|xml|yaml|yml)(?::\d+(?::\d+)?|\(\d+,\d+\))"
)
_TRACE_RE = re.compile(r'^\s*(?:at\s+|File\s+"|--->|E\s{2,}|\^\s*$)')
_TEST_SIGNAL_RE = re.compile(
    r"(?i)(^\s*(?:FAILED|FAIL|ERROR|ERRORS|FAILURES|●)\b|\b(?:tests?|specs?)\s+failed\b"
    r"|\b\d+\s+failed\b|\bassert(?:ion)?\b)"
)
_GIT_SIGNAL_RE = re.compile(
    r"^(?:diff --git|index\s|---\s|\+\+\+\s|@@\s|CONFLICT\b|On branch\b|Your branch\b|"
    r"Changes (?:to be committed|not staged)|Untracked files:|[ MADRCU?!]{1,2}\s+\S)"
)
_PHPSTAN_SIGNAL_RE = re.compile(r"^\s*(?:Line\s+\S+\.php\b|\d+\s+\S)")
_MARKER_RE = re.compile(r"^\[llm-optimizer: \d+ lines omitted; mode=\w+; tool=\w+\]$")


@dataclass(frozen=True, slots=True)
class _ModePolicy:
    head: int
    tail: int
    signal_context: int


_POLICIES: dict[ToolCompressionMode, _ModePolicy] = {
    "lite": _ModePolicy(head=64, tail=32, signal_context=3),
    "full": _ModePolicy(head=32, tail=16, signal_context=2),
    "ultra": _ModePolicy(head=8, tail=8, signal_context=1),
}


@dataclass(frozen=True, slots=True)
class ToolCompressionResult:
    body: bytes
    events: tuple[OptimizationEvent, ...]
    mode: ToolCompressionSelection | None
    compressed_outputs: int
    optimized_estimated_tokens: int


@dataclass(frozen=True, slots=True)
class _Invocation:
    name: str | None
    command: str | None


@dataclass(frozen=True, slots=True)
class _TextCompression:
    text: str
    omitted_lines: int


class ToolCompressionModeError(ValueError):
    pass


class ToolOutputCompressor:
    """Deterministically reduces recognized command output inside provider requests."""

    def __init__(self, config: ToolOutputCompressionConfig) -> None:
        self.config = config

    def apply(self, *, body: bytes, requested_mode: str | None) -> ToolCompressionResult:
        if not self.config.enabled:
            return _unchanged(body)

        mode = self._resolve_mode(requested_mode)
        if mode == "off":
            return ToolCompressionResult(
                body=body,
                events=(
                    OptimizationEvent(
                        kind="tool_output_compression_bypassed",
                        path="messages",
                        before_estimated_tokens=0,
                        after_estimated_tokens=0,
                        detail="reason=request_off",
                    ),
                ),
                mode="off",
                compressed_outputs=0,
                optimized_estimated_tokens=estimate_tokens(body),
            )

        payload = _json_object(body)
        if payload is None:
            return _unchanged(body, mode=mode)
        messages = payload.get("messages")
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            return _unchanged(body, mode=mode)

        invocations = _invocation_map(cast(list[dict[str, Any]], messages))
        events: list[OptimizationEvent] = []
        compressed_outputs = 0
        for message_index, message in enumerate(cast(list[dict[str, Any]], messages)):
            compressed_outputs += self._compress_openai_message(
                message=message,
                message_index=message_index,
                invocations=invocations,
                mode=mode,
                events=events,
            )
            compressed_outputs += self._compress_anthropic_blocks(
                message=message,
                message_index=message_index,
                invocations=invocations,
                mode=mode,
                events=events,
            )

        if compressed_outputs == 0:
            return _unchanged(body, mode=mode)
        optimized_body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return ToolCompressionResult(
            body=optimized_body,
            events=tuple(events),
            mode=mode,
            compressed_outputs=compressed_outputs,
            optimized_estimated_tokens=estimate_tokens(optimized_body),
        )

    def _resolve_mode(self, requested_mode: str | None) -> ToolCompressionSelection:
        if requested_mode is None:
            return self.config.mode
        allowed = (*get_args(ToolCompressionMode), "off")
        if requested_mode not in allowed:
            choices = ", ".join(allowed)
            raise ToolCompressionModeError(
                f"Unknown tool compression mode '{requested_mode}'. Expected one of: {choices}."
            )
        return cast(ToolCompressionSelection, requested_mode)

    def _compress_openai_message(
        self,
        *,
        message: dict[str, Any],
        message_index: int,
        invocations: dict[str, _Invocation],
        mode: ToolCompressionMode,
        events: list[OptimizationEvent],
    ) -> int:
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            return 0
        invocation = invocations.get(str(message.get("tool_call_id", "")))
        kind = _classify_tool(invocation, fallback_name=message.get("name"))
        if kind is None:
            return 0
        return self._compress_field(
            target=message,
            field="content",
            path=f"messages[{message_index}].content",
            kind=kind,
            mode=mode,
            events=events,
        )

    def _compress_anthropic_blocks(
        self,
        *,
        message: dict[str, Any],
        message_index: int,
        invocations: dict[str, _Invocation],
        mode: ToolCompressionMode,
        events: list[OptimizationEvent],
    ) -> int:
        content = message.get("content")
        if not isinstance(content, list):
            return 0
        compressed = 0
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            invocation = invocations.get(str(block.get("tool_use_id", "")))
            kind = _classify_tool(invocation, fallback_name=block.get("name"))
            if kind is None:
                continue
            path = f"messages[{message_index}].content[{block_index}].content"
            if isinstance(block.get("content"), str):
                compressed += self._compress_field(
                    target=block,
                    field="content",
                    path=path,
                    kind=kind,
                    mode=mode,
                    events=events,
                )
                continue
            children = block.get("content")
            if not isinstance(children, list):
                continue
            for child_index, child in enumerate(children):
                if not isinstance(child, dict) or child.get("type") != "text":
                    continue
                compressed += self._compress_field(
                    target=child,
                    field="text",
                    path=f"{path}[{child_index}].text",
                    kind=kind,
                    mode=mode,
                    events=events,
                )
        return compressed

    def _compress_field(
        self,
        *,
        target: dict[str, Any],
        field: str,
        path: str,
        kind: str,
        mode: ToolCompressionMode,
        events: list[OptimizationEvent],
    ) -> int:
        original = target.get(field)
        if not isinstance(original, str):
            return 0
        compressed = _compress_text(
            original,
            kind=kind,
            mode=mode,
            min_characters=self.config.min_characters,
        )
        if compressed is None:
            return 0
        target[field] = compressed.text
        events.append(
            OptimizationEvent(
                kind="tool_output_compressed",
                path=path,
                before_estimated_tokens=estimate_tokens(original),
                after_estimated_tokens=estimate_tokens(compressed.text),
                detail=(f"mode={mode};tool={kind};omitted_lines={compressed.omitted_lines}"),
            )
        )
        return 1


def _unchanged(
    body: bytes,
    *,
    mode: ToolCompressionSelection | None = None,
) -> ToolCompressionResult:
    return ToolCompressionResult(
        body=body,
        events=(),
        mode=mode,
        compressed_outputs=0,
        optimized_estimated_tokens=estimate_tokens(body),
    )


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _invocation_map(messages: list[dict[str, Any]]) -> dict[str, _Invocation]:
    invocations: dict[str, _Invocation] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                identifier = tool_call.get("id")
                function = tool_call.get("function")
                if not isinstance(identifier, str) or not isinstance(function, dict):
                    continue
                name = function.get("name") if isinstance(function.get("name"), str) else None
                command = _command_from_arguments(function.get("arguments"))
                invocations[identifier] = _Invocation(name=name, command=command)
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            identifier = block.get("id")
            if not isinstance(identifier, str):
                continue
            name = block.get("name") if isinstance(block.get("name"), str) else None
            invocations[identifier] = _Invocation(
                name=name,
                command=_command_from_mapping(block.get("input")),
            )
    return invocations


def _command_from_arguments(arguments: Any) -> str | None:
    if isinstance(arguments, dict):
        return _command_from_mapping(arguments)
    if not isinstance(arguments, str):
        return None
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return _command_from_mapping(decoded)


def _command_from_mapping(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    for key in ("cmd", "command", "script"):
        command = value.get(key)
        if isinstance(command, str):
            return command
        if isinstance(command, list) and all(isinstance(part, str) for part in command):
            return " ".join(command)
    return None


def _classify_tool(
    invocation: _Invocation | None,
    *,
    fallback_name: Any,
) -> str | None:
    candidates: list[str] = []
    if invocation is not None:
        if invocation.name:
            candidates.append(invocation.name)
        if invocation.command:
            candidates.append(invocation.command)
    if isinstance(fallback_name, str):
        candidates.append(fallback_name)

    tokens: set[str] = set()
    for candidate in candidates:
        try:
            parts = shlex.split(candidate)
        except ValueError:
            parts = candidate.split()
        for part in parts:
            normalized = PurePath(part.rstrip(";,|&")).name.lower()
            tokens.add(normalized.removesuffix(".cmd").removesuffix(".exe"))

    ordered_aliases = (
        ("pytest", {"pytest", "py.test"}),
        ("phpunit", {"phpunit"}),
        ("jest", {"jest"}),
        ("eslint", {"eslint"}),
        ("typescript", {"tsc", "typescript"}),
        ("phpstan", {"phpstan"}),
        ("ripgrep", {"rg", "ripgrep", "grep"}),
        ("git", {"git"}),
        ("composer", {"composer"}),
        ("pnpm", {"pnpm", "pnpx"}),
        ("npm", {"npm", "npx"}),
    )
    matches = {kind for kind, aliases in ordered_aliases if tokens & aliases}
    specific_matches = matches - {"npm", "pnpm"}
    if specific_matches:
        matches = specific_matches
    return next(iter(matches)) if len(matches) == 1 else None


def _compress_text(
    text: str,
    *,
    kind: str,
    mode: ToolCompressionMode,
    min_characters: int,
) -> _TextCompression | None:
    if len(text) < min_characters or _MARKER_PREFIX in text:
        return None
    lines = _visible_lines(text)
    policy = _POLICIES[mode]
    if len(lines) <= policy.head + policy.tail:
        return None

    keep = set(range(min(policy.head, len(lines))))
    keep.update(range(max(0, len(lines) - policy.tail), len(lines)))
    for index, line in enumerate(lines):
        if not _signal_line(line, kind=kind):
            continue
        start = max(0, index - policy.signal_context)
        end = min(len(lines), index + policy.signal_context + 1)
        keep.update(range(start, end))

    rendered: list[str] = []
    omitted_total = 0
    index = 0
    while index < len(lines):
        if index in keep:
            rendered.append(lines[index])
            index += 1
            continue
        start = index
        while index < len(lines) and index not in keep:
            index += 1
        omitted = index - start
        omitted_total += omitted
        rendered.append(f"{_MARKER_PREFIX} {omitted} lines omitted; mode={mode}; tool={kind}]")

    if omitted_total == 0:
        return None
    candidate = "\n".join(rendered)
    if text.endswith(("\n", "\r")):
        candidate += "\n"
    if len(candidate.encode("utf-8")) >= len(text.encode("utf-8")):
        return None
    return _TextCompression(text=candidate, omitted_lines=omitted_total)


def _visible_lines(text: str) -> list[str]:
    raw_lines = text.split("\n")
    if text.endswith("\n"):
        raw_lines.pop()
    lines: list[str] = []
    for line in raw_lines:
        if line.endswith("\r"):
            line = line[:-1]
        segments = [segment for segment in line.split("\r") if segment]
        visible = segments[-1] if segments else ""
        lines.append(_ANSI_RE.sub("", visible))
    return lines


def _signal_line(line: str, *, kind: str) -> bool:
    if _MARKER_RE.match(line) or _COMMON_SIGNAL_RE.search(line) or _TRACE_RE.search(line):
        return True
    if kind in {"pytest", "phpunit", "jest"}:
        return bool(_TEST_SIGNAL_RE.search(line) or _LOCATION_RE.search(line))
    if kind in {"eslint", "typescript", "phpstan"}:
        return bool(
            _LOCATION_RE.search(line) or (kind == "phpstan" and _PHPSTAN_SIGNAL_RE.search(line))
        )
    if kind == "git":
        return bool(_GIT_SIGNAL_RE.search(line))
    return False
