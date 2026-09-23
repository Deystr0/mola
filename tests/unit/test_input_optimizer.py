import json

from optimizer.config import ContextIntelligenceConfig, InputOptimizationConfig
from optimizer.context import InputOptimizer, estimate_tokens


def encode(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2).encode()


def test_disabled_optimizer_preserves_request_bytes() -> None:
    body = encode(
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "repeat"},
                {"role": "user", "content": "repeat"},
            ],
        }
    )

    result = InputOptimizer(InputOptimizationConfig()).optimize(body)

    assert result.body == body
    assert result.events == ()
    assert result.original_estimated_tokens == result.optimized_estimated_tokens


def test_enabled_optimizer_only_changes_older_optimizable_content() -> None:
    latest_user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "current request"},
            {"type": "text", "text": "current request"},
        ],
    }
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    body = encode(
        {
            "model": "test-model",
            "tools": tools,
            "messages": [
                {"role": "system", "content": "immutable"},
                {"role": "system", "content": "immutable"},
                {"role": "user", "content": "old duplicate"},
                {"role": "user", "content": "old duplicate"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "repeated context"},
                        {"type": "text", "text": "repeated context"},
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "a\r\nb"},
                {"role": "tool", "tool_call_id": "call-1", "content": "a\r\nb"},
                latest_user,
            ],
        }
    )
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            deduplicate_messages=True,
            deduplicate_content_blocks=True,
            deduplicate_tool_outputs=True,
            normalize_tool_output_line_endings=True,
        )
    )

    result = optimizer.optimize(body)
    payload = json.loads(result.body)
    messages = payload["messages"]

    assert messages[0] == {"role": "system", "content": "immutable"}
    assert messages[1] == {"role": "system", "content": "immutable"}
    assert sum(message.get("content") == "old duplicate" for message in messages) == 1
    assert messages[-1] == latest_user
    assert payload["tools"] == tools
    assert {event.kind for event in result.events} == {
        "duplicate_message_removed",
        "repeated_context_block_removed",
        "redundant_tool_output_removed",
        "tool_output_line_endings_normalized",
    }
    assert result.optimized_estimated_tokens < result.original_estimated_tokens


def test_anthropic_redundant_tool_results_are_removed_only_from_old_messages() -> None:
    duplicate = {"type": "tool_result", "tool_use_id": "tool-1", "content": "result"}
    current = {"role": "user", "content": "current request"}
    body = encode(
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "old request"},
                {"role": "user", "content": [duplicate, duplicate]},
                {"role": "assistant", "content": "continue"},
                current,
            ],
        }
    )

    result = InputOptimizer(InputOptimizationConfig(deduplicate_tool_outputs=True)).optimize(body)
    messages = json.loads(result.body)["messages"]

    assert len(messages[1]["content"]) == 1
    assert messages[-1] == current
    assert [event.kind for event in result.events] == ["redundant_tool_output_removed"]


def test_token_budget_is_observable_but_non_destructive() -> None:
    body = encode(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Never prune this request."}],
        }
    )

    result = InputOptimizer(InputOptimizationConfig(max_estimated_tokens=1)).optimize(body)

    assert result.body == body
    assert len(result.events) == 1
    assert result.events[0].kind == "token_budget_exceeded"
    assert result.events[0].before_estimated_tokens == result.events[0].after_estimated_tokens
    assert "action=none" in result.events[0].detail


def test_invalid_json_is_never_modified() -> None:
    body = b"not json"

    result = InputOptimizer(
        InputOptimizationConfig(deduplicate_messages=True, max_estimated_tokens=1)
    ).optimize(body)

    assert result.body == body
    assert result.events == ()


def test_messages_without_a_user_boundary_are_treated_as_immutable() -> None:
    body = encode(
        {
            "model": "test-model",
            "messages": [
                {"role": "assistant", "content": "same"},
                {"role": "assistant", "content": "same"},
            ],
        }
    )

    result = InputOptimizer(InputOptimizationConfig(deduplicate_messages=True)).optimize(body)

    assert result.body == body
    assert result.events == ()


def test_context_intelligence_prunes_irrelevant_history_and_keeps_relevant_context() -> None:
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    body = encode(
        {
            "model": "test-model",
            "tools": tools,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "immutable system"},
                {"role": "developer", "content": "immutable developer"},
                {"role": "user", "content": "gardening " + ("mulch " * 120)},
                {"role": "assistant", "content": "garden notes " + ("soil " * 120)},
                {
                    "role": "user",
                    "content": "Database migration plan with rollback checkpoints.",
                },
                {
                    "role": "assistant",
                    "content": "The database migration plan retains rollback checkpoints.",
                },
                {
                    "role": "user",
                    "content": "Continue the database migration plan and rollback checkpoints.",
                },
            ],
        }
    )
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            max_estimated_tokens=180,
            context=ContextIntelligenceConfig(enabled=True, min_recent_turns=0),
        )
    )

    result = optimizer.optimize(body)
    payload = json.loads(result.body)
    contents = [message.get("content") for message in payload["messages"]]

    assert not any(isinstance(content, str) and "gardening" in content for content in contents)
    assert any(
        isinstance(content, str) and "Database migration plan" in content for content in contents
    )
    assert payload["messages"][0] == {"role": "system", "content": "immutable system"}
    assert payload["messages"][1] == {
        "role": "developer",
        "content": "immutable developer",
    }
    assert payload["messages"][-1]["content"].startswith("Continue the database")
    assert payload["tools"] == tools
    assert payload["response_format"] == {"type": "json_object"}
    assert [event.kind for event in result.events] == [
        "request_json_compacted",
        "context_chunk_pruned",
    ]
    assert "relevance=" in result.events[1].detail
    assert result.optimized_estimated_tokens <= 180


def test_context_intelligence_keeps_recent_turn_floor_and_reports_unmet_budget() -> None:
    body = encode(
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "oldest alpha " + ("x " * 100)},
                {"role": "assistant", "content": "oldest answer"},
                {"role": "user", "content": "recent beta " + ("y " * 100)},
                {"role": "assistant", "content": "recent beta answer"},
                {"role": "user", "content": "recent gamma " + ("z " * 100)},
                {"role": "assistant", "content": "recent gamma answer"},
                {"role": "user", "content": "current unrelated request"},
            ],
        }
    )
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            max_estimated_tokens=1,
            context=ContextIntelligenceConfig(enabled=True, min_recent_turns=2),
        )
    )

    result = optimizer.optimize(body)
    contents = [message["content"] for message in json.loads(result.body)["messages"]]

    assert not any("oldest alpha" in content for content in contents)
    assert any("recent beta" in content for content in contents)
    assert any("recent gamma" in content for content in contents)
    assert result.events[-1].kind == "context_budget_unmet"
    assert "action=conservative_stop" in result.events[-1].detail
    assert "reason=protected_context_remaining" in result.events[-1].detail


def test_context_intelligence_keeps_important_history_above_score_threshold() -> None:
    body = encode(
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": "Security constraint: never expose the required secret.",
                },
                {"role": "assistant", "content": "Acknowledged."},
                {"role": "user", "content": "Write an unrelated greeting."},
            ],
        }
    )
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            max_estimated_tokens=1,
            context=ContextIntelligenceConfig(enabled=True, min_recent_turns=0),
        )
    )

    result = optimizer.optimize(body)

    assert json.loads(result.body) == json.loads(body)
    assert [event.kind for event in result.events] == [
        "request_json_compacted",
        "context_budget_unmet",
    ]
    assert "reason=score_threshold" in result.events[-1].detail


def test_context_intelligence_prefers_redundant_history_over_unique_history() -> None:
    duplicate_turn = [
        {"role": "user", "content": "Boilerplate repeated context."},
        {"role": "assistant", "content": "Boilerplate repeated response."},
    ]
    unique_turn = [
        {"role": "user", "content": "Zephyr cobalt archival details."},
        {"role": "assistant", "content": "Quartz-specific response."},
    ]
    current = {"role": "user", "content": "Current unrelated request."}
    payload = {
        "model": "test-model",
        "messages": [*duplicate_turn, *duplicate_turn, *unique_turn, current],
    }
    body = encode(payload)
    one_duplicate_removed = {
        "model": "test-model",
        "messages": [*duplicate_turn, *unique_turn, current],
    }
    budget = estimate_tokens(one_duplicate_removed)
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            max_estimated_tokens=budget,
            context=ContextIntelligenceConfig(enabled=True, min_recent_turns=0),
        )
    )

    result = optimizer.optimize(body)
    contents = [message["content"] for message in json.loads(result.body)["messages"]]

    assert contents.count("Boilerplate repeated context.") == 1
    assert "Zephyr cobalt archival details." in contents
    pruning_event = next(event for event in result.events if event.kind == "context_chunk_pruned")
    assert "uniqueness=0.000000" in pruning_event.detail


def test_context_intelligence_prunes_complete_openai_tool_dependency_group() -> None:
    tools = [{"type": "function", "function": {"name": "weather"}}]
    old_turn = [
        {"role": "user", "content": "Old weather request."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
        {"role": "assistant", "content": "Old answer."},
    ]
    body = encode(
        {
            "model": "test-model",
            "tools": tools,
            "messages": [*old_turn, {"role": "user", "content": "Current request."}],
        }
    )
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            max_estimated_tokens=1,
            context=ContextIntelligenceConfig(
                enabled=True,
                min_recent_turns=0,
                max_prunable_score=1,
            ),
        )
    )

    result = optimizer.optimize(body)
    payload = json.loads(result.body)

    assert payload["messages"] == [{"role": "user", "content": "Current request."}]
    assert payload["tools"] == tools
    pruning_event = next(event for event in result.events if event.kind == "context_chunk_pruned")
    assert "messages=4" in pruning_event.detail
    assert result.events[-1].kind == "context_budget_unmet"


def test_context_intelligence_keeps_anthropic_tool_use_and_result_in_one_chunk() -> None:
    old_turn = [
        {"role": "user", "content": "Old lookup request."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool-1", "content": "result"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Old answer."}]},
    ]
    body = encode(
        {
            "model": "test-model",
            "messages": [*old_turn, {"role": "user", "content": "Current request."}],
        }
    )
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            max_estimated_tokens=1,
            context=ContextIntelligenceConfig(
                enabled=True,
                min_recent_turns=0,
                max_prunable_score=1,
            ),
        )
    )

    result = optimizer.optimize(body)

    assert json.loads(result.body)["messages"] == [{"role": "user", "content": "Current request."}]
    pruning_event = next(event for event in result.events if event.kind == "context_chunk_pruned")
    assert "messages=4" in pruning_event.detail


def test_context_intelligence_preserves_active_tool_sequence_after_current_request() -> None:
    active = [
        {"role": "user", "content": "Current request."},
        {
            "role": "assistant",
            "tool_calls": [{"id": "active", "type": "function", "function": {}}],
        },
        {"role": "tool", "tool_call_id": "active", "content": "active result"},
    ]
    body = encode(
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Old unrelated history " + ("old " * 100)},
                {"role": "assistant", "content": "Old answer."},
                *active,
            ],
        }
    )
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            max_estimated_tokens=1,
            context=ContextIntelligenceConfig(
                enabled=True,
                min_recent_turns=0,
                max_prunable_score=1,
            ),
        )
    )

    result = optimizer.optimize(body)

    assert json.loads(result.body)["messages"] == active


def test_context_intelligence_is_byte_transparent_below_budget() -> None:
    body = encode(
        {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Old context."},
                {"role": "assistant", "content": "Old response."},
                {"role": "user", "content": "Current request."},
            ],
        }
    )
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            max_estimated_tokens=10_000,
            context=ContextIntelligenceConfig(enabled=True),
        )
    )

    result = optimizer.optimize(body)

    assert result.body == body
    assert result.events == ()


def test_context_intelligence_compacts_json_before_pruning_history() -> None:
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Historical context."},
            {"role": "assistant", "content": "Historical response."},
            {"role": "user", "content": "Current request."},
        ],
    }
    body = encode(payload)
    compact = json.dumps(payload, separators=(",", ":")).encode()
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            max_estimated_tokens=estimate_tokens(compact),
            context=ContextIntelligenceConfig(enabled=True, min_recent_turns=0),
        )
    )

    result = optimizer.optimize(body)

    assert result.body == compact
    assert json.loads(result.body)["messages"] == payload["messages"]
    assert [event.kind for event in result.events] == ["request_json_compacted"]


def test_anthropic_tool_result_without_human_request_is_immutable() -> None:
    body = encode(
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tool-1", "content": "same"},
                        {"type": "tool_result", "tool_use_id": "tool-1", "content": "same"},
                    ],
                }
            ],
        }
    )

    result = InputOptimizer(
        InputOptimizationConfig(
            deduplicate_tool_outputs=True,
            max_estimated_tokens=1,
            context=ContextIntelligenceConfig(enabled=True, min_recent_turns=0),
        )
    ).optimize(body)

    assert result.body == body
    assert [event.kind for event in result.events] == ["context_budget_unmet"]


def test_context_intelligence_fails_safe_for_unknown_message_roles() -> None:
    body = encode(
        {
            "model": "test-model",
            "messages": [
                {"role": "policy", "content": "Unknown structured constraint."},
                {"role": "user", "content": "Old context " + ("old " * 100)},
                {"role": "assistant", "content": "Old response."},
                {"role": "user", "content": "Current request."},
            ],
        }
    )
    optimizer = InputOptimizer(
        InputOptimizationConfig(
            max_estimated_tokens=1,
            context=ContextIntelligenceConfig(
                enabled=True,
                min_recent_turns=0,
                max_prunable_score=1,
            ),
        )
    )

    first = optimizer.optimize(body)
    second = optimizer.optimize(body)

    assert first.body == body
    assert first == second
    assert [event.kind for event in first.events] == ["context_budget_unmet"]
    assert "reason=unsupported_shape_or_request" in first.events[0].detail
