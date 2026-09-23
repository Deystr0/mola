from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from optimizer.config import ContextIntelligenceConfig, InputOptimizationConfig

_TOKEN_PATTERN = re.compile(r"[\w./:-]{2,}", re.UNICODE)
_IMPORTANT_PATTERN = re.compile(
    r"\b(?:must|never|required|constraint|decision|error|failed|failure|warning|security|"
    r"critical|regression|blocked|breaking|todo|fixme)\b",
    re.IGNORECASE,
)
_FILE_PATTERN = re.compile(
    r"(?:^|\s)(?:[\w.-]+/)+[\w.-]+|\b[\w.-]+\."
    r"(?:py|pyi|js|jsx|ts|tsx|go|rs|java|kt|rb|php|cs|cpp|c|h|sql|ya?ml|json|toml|md)\b",
    re.IGNORECASE,
)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class OptimizationEvent:
    kind: str
    path: str
    before_estimated_tokens: int
    after_estimated_tokens: int
    detail: str


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    body: bytes
    events: tuple[OptimizationEvent, ...]
    original_estimated_tokens: int
    optimized_estimated_tokens: int


@dataclass(slots=True)
class _Message:
    value: dict[str, Any]
    original_index: int


@dataclass(frozen=True, slots=True)
class _ContextChunk:
    messages: tuple[_Message, ...]
    relevance: float
    dependency: float
    recency: float
    uniqueness: float
    importance: float

    @property
    def score(self) -> float:
        return (
            (0.40 * self.relevance)
            + (0.15 * self.dependency)
            + (0.15 * self.recency)
            + (0.10 * self.uniqueness)
            + (0.20 * self.importance)
        )


class InputOptimizer:
    """Conservative JSON-message optimizer with immutable-context guarantees."""

    def __init__(self, config: InputOptimizationConfig) -> None:
        self.config = config

    def optimize(self, body: bytes) -> OptimizationResult:
        original_tokens = estimate_tokens(body)
        if not self._enabled:
            return OptimizationResult(body, (), original_tokens, original_tokens)

        payload = _json_object(body)
        if payload is None:
            return OptimizationResult(body, (), original_tokens, original_tokens)
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not all(
            isinstance(message, dict) for message in raw_messages
        ):
            return OptimizationResult(body, (), original_tokens, original_tokens)

        messages = [
            _Message(value=message, original_index=index)
            for index, message in enumerate(raw_messages)
        ]
        latest_user_index = _latest_user_index(messages)
        events: list[OptimizationEvent] = []
        changed = False

        if self.config.normalize_tool_output_line_endings:
            changed |= _normalize_tool_output_line_endings(messages, latest_user_index, events)
        if self.config.deduplicate_tool_outputs:
            changed |= _deduplicate_tool_outputs(messages, latest_user_index, events)
        if self.config.deduplicate_content_blocks:
            changed |= _deduplicate_content_blocks(messages, latest_user_index, events)
        if self.config.deduplicate_messages:
            changed |= _deduplicate_messages(messages, latest_user_index, events)

        optimized_body = _materialize(payload, messages) if changed else body

        optimized_tokens = estimate_tokens(optimized_body)
        budget = self.config.max_estimated_tokens
        context_supported = _supports_context_intelligence(messages, latest_user_index)
        context_stop_reason = "unsupported_shape_or_request"
        if (
            self.config.context.enabled
            and context_supported
            and budget is not None
            and optimized_tokens > budget
        ):
            compact_body = _materialize(payload, messages)
            compact_tokens = estimate_tokens(compact_body)
            if compact_tokens < optimized_tokens:
                events.append(
                    OptimizationEvent(
                        kind="request_json_compacted",
                        path="$",
                        before_estimated_tokens=optimized_tokens,
                        after_estimated_tokens=compact_tokens,
                        detail="whitespace=removed;semantic_content=unchanged",
                    )
                )
                optimized_body = compact_body
                optimized_tokens = compact_tokens
        if (
            self.config.context.enabled
            and context_supported
            and budget is not None
            and optimized_tokens > budget
        ):
            optimized_body, optimized_tokens, context_stop_reason = _prune_context(
                payload=payload,
                messages=messages,
                latest_user_index=latest_user_index,
                budget=budget,
                config=self.config.context,
                events=events,
                current_body=optimized_body,
            )
        if budget is not None and optimized_tokens > budget:
            kind = (
                "context_budget_unmet" if self.config.context.enabled else "token_budget_exceeded"
            )
            action = "conservative_stop" if self.config.context.enabled else "none"
            reason = f";reason={context_stop_reason}" if self.config.context.enabled else ""
            events.append(
                OptimizationEvent(
                    kind=kind,
                    path="messages",
                    before_estimated_tokens=optimized_tokens,
                    after_estimated_tokens=optimized_tokens,
                    detail=(
                        f"estimated_tokens={optimized_tokens};budget={budget};"
                        f"action={action}{reason}"
                    ),
                )
            )

        return OptimizationResult(
            body=optimized_body,
            events=tuple(events),
            original_estimated_tokens=original_tokens,
            optimized_estimated_tokens=optimized_tokens,
        )

    @property
    def _enabled(self) -> bool:
        return any(
            (
                self.config.deduplicate_messages,
                self.config.deduplicate_content_blocks,
                self.config.deduplicate_tool_outputs,
                self.config.normalize_tool_output_line_endings,
                self.config.max_estimated_tokens is not None,
                self.config.context.enabled,
            )
        )


def _materialize(payload: dict[str, Any], messages: list[_Message]) -> bytes:
    payload["messages"] = [message.value for message in messages]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def estimate_tokens(value: bytes | str | dict[str, Any] | list[Any]) -> int:
    """Return a deterministic approximation without model-specific tokenizers."""
    if isinstance(value, bytes):
        byte_length = len(value)
    elif isinstance(value, str):
        byte_length = len(value.encode("utf-8"))
    else:
        byte_length = len(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    return (byte_length + 3) // 4


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _latest_user_index(messages: list[_Message]) -> int | None:
    for message in reversed(messages):
        if _is_human_user_message(message.value):
            return message.original_index
    return None


def _is_human_user_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return True
    return any(
        not isinstance(block, dict) or block.get("type") != "tool_result" for block in content
    )


def _is_immutable(message: _Message, latest_user_index: int | None) -> bool:
    role = message.value.get("role")
    if role in {"system", "developer"}:
        return True
    if latest_user_index is None:
        return True
    # A trailing assistant/tool sequence may belong to the active turn. Protect it with the request.
    return message.original_index >= latest_user_index


def _prune_context(
    *,
    payload: dict[str, Any],
    messages: list[_Message],
    latest_user_index: int | None,
    budget: int,
    config: ContextIntelligenceConfig,
    events: list[OptimizationEvent],
    current_body: bytes,
) -> tuple[bytes, int, str]:
    chunks = _score_context_chunks(messages, latest_user_index)
    protected_count = min(config.min_recent_turns, len(chunks))
    candidate_count = len(chunks) - protected_count
    candidates = sorted(
        (chunk for chunk in chunks[:candidate_count] if chunk.score <= config.max_prunable_score),
        key=lambda chunk: (chunk.score, chunk.messages[0].original_index),
    )

    optimized_body = current_body
    optimized_tokens = estimate_tokens(current_body)
    for chunk in candidates:
        if optimized_tokens <= budget:
            break
        removed_indexes = {message.original_index for message in chunk.messages}
        messages[:] = [
            message for message in messages if message.original_index not in removed_indexes
        ]
        events.append(
            OptimizationEvent(
                kind="context_chunk_pruned",
                path=_chunk_path(chunk),
                before_estimated_tokens=sum(
                    estimate_tokens(message.value) for message in chunk.messages
                ),
                after_estimated_tokens=0,
                detail=(
                    f"score={chunk.score:.6f};relevance={chunk.relevance:.6f};"
                    f"dependency={chunk.dependency:.6f};recency={chunk.recency:.6f};"
                    f"uniqueness={chunk.uniqueness:.6f};importance={chunk.importance:.6f};"
                    f"messages={len(chunk.messages)}"
                ),
            )
        )
        optimized_body = _materialize(payload, messages)
        optimized_tokens = estimate_tokens(optimized_body)
    if optimized_tokens <= budget:
        stop_reason = "budget_met"
    elif not chunks:
        stop_reason = "no_historical_chunks"
    elif candidate_count == 0:
        stop_reason = "recent_turn_floor"
    elif not candidates:
        stop_reason = "score_threshold"
    else:
        stop_reason = "protected_context_remaining"
    return optimized_body, optimized_tokens, stop_reason


def _score_context_chunks(
    messages: list[_Message],
    latest_user_index: int | None,
) -> list[_ContextChunk]:
    if not _supports_context_intelligence(messages, latest_user_index):
        return []
    assert latest_user_index is not None
    latest_user = next(
        (message for message in messages if message.original_index == latest_user_index),
        None,
    )
    if latest_user is None:
        return []
    query_tokens = _tokens(_message_text(latest_user.value))
    if not query_tokens:
        return []

    raw_chunks = _historical_chunks(messages, latest_user_index)
    chunk_tokens = [_tokens(_chunk_text(chunk)) for chunk in raw_chunks]
    token_frequency = Counter(token for tokens in chunk_tokens for token in tokens)
    scored: list[_ContextChunk] = []
    chunk_count = len(raw_chunks)
    for position, (chunk, tokens) in enumerate(zip(raw_chunks, chunk_tokens, strict=True)):
        relevance = len(tokens & query_tokens) / len(query_tokens)
        uniqueness = (
            sum(token_frequency[token] == 1 for token in tokens) / len(tokens) if tokens else 0.0
        )
        scored.append(
            _ContextChunk(
                messages=chunk,
                relevance=relevance,
                dependency=_dependency_score(chunk),
                recency=(position + 1) / chunk_count,
                uniqueness=uniqueness,
                importance=_importance_score(chunk),
            )
        )
    return scored


def _supports_context_intelligence(
    messages: list[_Message],
    latest_user_index: int | None,
) -> bool:
    if latest_user_index is None:
        return False
    if any(
        message.value.get("role")
        not in {"system", "developer", "user", "assistant", "tool", "function"}
        for message in messages
    ):
        return False
    latest_user = next(
        (message for message in messages if message.original_index == latest_user_index),
        None,
    )
    return latest_user is not None and bool(_tokens(_message_text(latest_user.value)))


def _historical_chunks(
    messages: list[_Message],
    latest_user_index: int,
) -> list[tuple[_Message, ...]]:
    chunks: list[tuple[_Message, ...]] = []
    current: list[_Message] = []
    for message in messages:
        if _is_immutable(message, latest_user_index):
            if current:
                chunks.append(tuple(current))
                current = []
            continue
        if _is_human_user_message(message.value) and current:
            chunks.append(tuple(current))
            current = []
        current.append(message)
    if current:
        chunks.append(tuple(current))
    return chunks


def _dependency_score(chunk: tuple[_Message, ...]) -> float:
    if any(_has_tool_dependency(message.value) for message in chunk):
        return 1.0
    if len(chunk) > 1:
        return 0.5
    if _is_human_user_message(chunk[0].value):
        return 0.25
    return 0.0


def _has_tool_dependency(message: dict[str, Any]) -> bool:
    if (
        message.get("role") in {"tool", "function"}
        or "tool_calls" in message
        or "tool_call_id" in message
    ):
        return True
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in {"tool_use", "tool_result"}
        for block in content
    )


def _importance_score(chunk: tuple[_Message, ...]) -> float:
    text = _chunk_text(chunk)
    score = 0.2 if any(_is_human_user_message(message.value) for message in chunk) else 0.0
    if _IMPORTANT_PATTERN.search(text):
        return 1.0
    if "```" in text or _FILE_PATTERN.search(text):
        score = max(score, 0.8)
    if any(_has_tool_dependency(message.value) for message in chunk):
        score = max(score, 0.5)
    return score


def _chunk_text(chunk: tuple[_Message, ...]) -> str:
    return "\n".join(_message_text(message.value) for message in chunk)


def _message_text(message: dict[str, Any]) -> str:
    fragments: list[str] = []
    _collect_text(message.get("content"), fragments)
    _collect_text(message.get("tool_calls"), fragments)
    return "\n".join(fragments)


def _collect_text(value: Any, fragments: list[str]) -> None:
    if isinstance(value, str):
        fragments.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_text(item, fragments)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in {"id", "type", "role", "tool_call_id", "tool_use_id"}:
                _collect_text(child, fragments)


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_PATTERN.findall(text.casefold()) if token not in _STOP_WORDS}


def _chunk_path(chunk: _ContextChunk) -> str:
    indexes = ",".join(str(message.original_index) for message in chunk.messages)
    return f"messages[{indexes}]"


def _deduplicate_messages(
    messages: list[_Message],
    latest_user_index: int | None,
    events: list[OptimizationEvent],
) -> bool:
    changed = False
    index = 1
    while index < len(messages):
        previous = messages[index - 1]
        current = messages[index]
        if (
            not _is_immutable(current, latest_user_index)
            and _plain_message(current.value)
            and current.value == previous.value
        ):
            events.append(
                OptimizationEvent(
                    kind="duplicate_message_removed",
                    path=f"messages[{current.original_index}]",
                    before_estimated_tokens=estimate_tokens(current.value),
                    after_estimated_tokens=0,
                    detail=f"duplicate_of=messages[{previous.original_index}]",
                )
            )
            messages.pop(index)
            changed = True
            continue
        index += 1
    return changed


def _plain_message(message: dict[str, Any]) -> bool:
    return (
        message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
        and "tool_calls" not in message
        and "tool_call_id" not in message
    )


def _deduplicate_content_blocks(
    messages: list[_Message],
    latest_user_index: int | None,
    events: list[OptimizationEvent],
) -> bool:
    changed = False
    for message in messages:
        if _is_immutable(message, latest_user_index):
            continue
        content = message.value.get("content")
        if not isinstance(content, list):
            continue
        seen: dict[str, int] = {}
        optimized: list[Any] = []
        for block_index, block in enumerate(content):
            key = _text_block_key(block)
            if key is None or key not in seen:
                if key is not None:
                    seen[key] = block_index
                optimized.append(block)
                continue
            events.append(
                OptimizationEvent(
                    kind="repeated_context_block_removed",
                    path=f"messages[{message.original_index}].content[{block_index}]",
                    before_estimated_tokens=estimate_tokens(block),
                    after_estimated_tokens=0,
                    detail=(
                        f"duplicate_of=messages[{message.original_index}].content[{seen[key]}]"
                    ),
                )
            )
            changed = True
        if len(optimized) != len(content):
            message.value["content"] = optimized
    return changed


def _text_block_key(block: Any) -> str | None:
    if not isinstance(block, dict) or block.get("type") != "text":
        return None
    return json.dumps(block, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _deduplicate_tool_outputs(
    messages: list[_Message],
    latest_user_index: int | None,
    events: list[OptimizationEvent],
) -> bool:
    changed = _deduplicate_openai_tool_messages(messages, latest_user_index, events)
    changed |= _deduplicate_anthropic_tool_blocks(messages, latest_user_index, events)
    return changed


def _deduplicate_openai_tool_messages(
    messages: list[_Message],
    latest_user_index: int | None,
    events: list[OptimizationEvent],
) -> bool:
    changed = False
    index = 1
    while index < len(messages):
        previous = messages[index - 1]
        current = messages[index]
        if (
            not _is_immutable(current, latest_user_index)
            and current.value.get("role") == "tool"
            and current.value == previous.value
        ):
            events.append(
                OptimizationEvent(
                    kind="redundant_tool_output_removed",
                    path=f"messages[{current.original_index}]",
                    before_estimated_tokens=estimate_tokens(current.value),
                    after_estimated_tokens=0,
                    detail=f"duplicate_of=messages[{previous.original_index}]",
                )
            )
            messages.pop(index)
            changed = True
            continue
        index += 1
    return changed


def _deduplicate_anthropic_tool_blocks(
    messages: list[_Message],
    latest_user_index: int | None,
    events: list[OptimizationEvent],
) -> bool:
    changed = False
    for message in messages:
        if _is_immutable(message, latest_user_index):
            continue
        content = message.value.get("content")
        if not isinstance(content, list):
            continue
        optimized: list[Any] = []
        previous_key: str | None = None
        previous_index: int | None = None
        for block_index, block in enumerate(content):
            key = _tool_result_key(block)
            if key is not None and key == previous_key:
                events.append(
                    OptimizationEvent(
                        kind="redundant_tool_output_removed",
                        path=f"messages[{message.original_index}].content[{block_index}]",
                        before_estimated_tokens=estimate_tokens(block),
                        after_estimated_tokens=0,
                        detail=(
                            f"duplicate_of=messages[{message.original_index}]"
                            f".content[{previous_index}]"
                        ),
                    )
                )
                changed = True
                continue
            optimized.append(block)
            previous_key = key
            previous_index = block_index
        if len(optimized) != len(content):
            message.value["content"] = optimized
    return changed


def _tool_result_key(block: Any) -> str | None:
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return None
    return json.dumps(block, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_tool_output_line_endings(
    messages: list[_Message],
    latest_user_index: int | None,
    events: list[OptimizationEvent],
) -> bool:
    changed = False
    for message in messages:
        if _is_immutable(message, latest_user_index):
            continue
        if message.value.get("role") == "tool" and isinstance(message.value.get("content"), str):
            changed |= _normalize_string_field(
                message.value,
                "content",
                f"messages[{message.original_index}].content",
                events,
            )
        content = message.value.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            changed |= _normalize_tool_result_block(
                block,
                f"messages[{message.original_index}].content[{block_index}]",
                events,
            )
    return changed


def _normalize_tool_result_block(
    block: dict[str, Any],
    path: str,
    events: list[OptimizationEvent],
) -> bool:
    content = block.get("content")
    if isinstance(content, str):
        return _normalize_string_field(block, "content", f"{path}.content", events)
    if not isinstance(content, list):
        return False
    changed = False
    for index, child in enumerate(content):
        if isinstance(child, dict) and child.get("type") == "text":
            changed |= _normalize_string_field(
                child,
                "text",
                f"{path}.content[{index}].text",
                events,
            )
    return changed


def _normalize_string_field(
    target: dict[str, Any],
    field: str,
    path: str,
    events: list[OptimizationEvent],
) -> bool:
    original = target.get(field)
    if not isinstance(original, str):
        return False
    normalized = original.replace("\r\n", "\n")
    if normalized == original:
        return False
    target[field] = normalized
    events.append(
        OptimizationEvent(
            kind="tool_output_line_endings_normalized",
            path=path,
            before_estimated_tokens=estimate_tokens(original),
            after_estimated_tokens=estimate_tokens(normalized),
            detail="line_endings=crlf_to_lf",
        )
    )
    return True
