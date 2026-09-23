import json

import pytest

from optimizer.config import (
    OutputOptimizationConfig,
    OutputPoliciesConfig,
    OutputPolicyConfig,
)
from optimizer.output import OutputBudgetController, OutputPolicyError


def controller(
    *,
    policy: str = "user_answer",
    max_tokens: int | None = 100,
    openai_parameter: str = "max_completion_tokens",
) -> OutputBudgetController:
    policies = OutputPoliciesConfig()
    setattr(policies, policy, OutputPolicyConfig(max_tokens=max_tokens))
    return OutputBudgetController(
        OutputOptimizationConfig(
            enabled=True,
            openai_parameter=openai_parameter,
            policies=policies,
        )
    )


def encode(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, indent=2).encode()


def test_disabled_controller_preserves_bytes_and_ignores_policy_header() -> None:
    body = encode({"model": "test-model", "messages": []})

    result = OutputBudgetController(OutputOptimizationConfig()).apply(
        provider="openai",
        body=body,
        requested_policy="not-a-policy",
    )

    assert result.body == body
    assert result.events == ()
    assert result.policy is None


def test_openai_policy_sets_configured_parameter_when_limit_is_missing() -> None:
    body = encode({"model": "test-model", "messages": []})

    result = controller().apply(
        provider="openai",
        body=body,
        requested_policy="user_answer",
    )

    assert json.loads(result.body)["max_completion_tokens"] == 100
    assert result.events[0].kind == "output_token_limit_set"
    assert result.events[0].after_estimated_tokens == 100


def test_openai_policy_caps_the_parameter_already_used_by_the_request() -> None:
    body = encode({"model": "test-model", "messages": [], "max_tokens": 500})

    result = controller().apply(
        provider="openai",
        body=body,
        requested_policy="user_answer",
    )

    payload = json.loads(result.body)
    assert payload["max_tokens"] == 100
    assert "max_completion_tokens" not in payload
    assert result.events[0].kind == "output_token_limit_capped"
    assert result.events[0].before_estimated_tokens == 500


def test_existing_lower_limit_is_preserved_without_reserializing_body() -> None:
    body = encode({"model": "test-model", "messages": [], "max_tokens": 50})

    result = controller().apply(
        provider="openai",
        body=body,
        requested_policy="user_answer",
    )

    assert result.body == body
    assert result.events[0].kind == "output_token_limit_preserved"


def test_anthropic_always_uses_max_tokens() -> None:
    body = encode({"model": "test-model", "messages": []})

    result = controller(policy="tool_call", max_tokens=32).apply(
        provider="anthropic",
        body=body,
        requested_policy="tool_call",
    )

    payload = json.loads(result.body)
    assert payload["max_tokens"] == 32
    assert "max_completion_tokens" not in payload


def test_invalid_existing_limit_is_not_rewritten() -> None:
    body = encode({"model": "test-model", "messages": [], "max_tokens": "many"})

    result = controller().apply(
        provider="openai",
        body=body,
        requested_policy="user_answer",
    )

    assert result.body == body
    assert result.events[0].kind == "output_budget_skipped_invalid_limit"


@pytest.mark.parametrize("value", [None, True, 0, "many"])
def test_ambiguous_existing_value_is_not_rewritten(value: object) -> None:
    body = encode({"model": "test-model", "messages": [], "max_tokens": value})

    result = controller().apply(
        provider="openai",
        body=body,
        requested_policy="user_answer",
    )

    assert result.body == body
    assert result.events[0].kind == "output_budget_skipped_invalid_limit"


def test_multiple_openai_limit_parameters_are_not_rewritten() -> None:
    body = encode(
        {
            "model": "test-model",
            "messages": [],
            "max_tokens": 500,
            "max_completion_tokens": 500,
        }
    )

    result = controller().apply(
        provider="openai",
        body=body,
        requested_policy="user_answer",
    )

    assert result.body == body
    assert result.events[0].kind == "output_budget_skipped_ambiguous_limit"


def test_default_policy_applies_without_request_header() -> None:
    policies = OutputPoliciesConfig(user_answer=OutputPolicyConfig(max_tokens=75))
    budget_controller = OutputBudgetController(
        OutputOptimizationConfig(
            enabled=True,
            default_policy="user_answer",
            policies=policies,
        )
    )

    result = budget_controller.apply(provider="openai", body=b"{}", requested_policy=None)

    assert json.loads(result.body)["max_completion_tokens"] == 75
    assert result.policy == "user_answer"


def test_policy_without_a_configured_limit_is_a_no_op() -> None:
    body = encode({"model": "test-model", "messages": []})

    result = controller(max_tokens=None).apply(
        provider="openai",
        body=body,
        requested_policy="user_answer",
    )

    assert result.body == body
    assert result.events[0].kind == "output_budget_skipped_unconfigured"
    assert result.policy == "user_answer"


def test_unknown_policy_is_rejected_when_controller_is_enabled() -> None:
    with pytest.raises(OutputPolicyError, match="Unknown output policy"):
        controller().apply(
            provider="openai",
            body=b"{}",
            requested_policy="unknown",
        )


@pytest.mark.parametrize(
    "policy",
    [
        "machine_to_machine",
        "tool_call",
        "code",
        "analysis",
        "user_answer",
        "documentation",
    ],
)
def test_all_roadmap_policies_are_selectable(policy: str) -> None:
    result = controller(policy=policy, max_tokens=80).apply(
        provider="openai",
        body=b"{}",
        requested_policy=policy,
    )

    assert result.policy == policy
    assert json.loads(result.body)["max_completion_tokens"] == 80
