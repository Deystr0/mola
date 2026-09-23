from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast, get_args

from optimizer.config import OutputOptimizationConfig, OutputPolicyName
from optimizer.context import OptimizationEvent


class OutputPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OutputControlResult:
    body: bytes
    events: tuple[OptimizationEvent, ...]
    policy: OutputPolicyName | None
    max_tokens: int | None


class OutputBudgetController:
    """Applies explicit output ceilings before provider generation starts."""

    def __init__(self, config: OutputOptimizationConfig) -> None:
        self.config = config

    def apply(
        self,
        *,
        provider: str,
        body: bytes,
        requested_policy: str | None,
    ) -> OutputControlResult:
        if not self.config.enabled:
            return OutputControlResult(body, (), None, None)

        policy = self._resolve_policy(requested_policy)
        if policy is None:
            return OutputControlResult(body, (), None, None)
        limit = getattr(self.config.policies, policy).max_tokens
        if limit is None:
            event = _event(
                kind="output_budget_skipped_unconfigured",
                path="$",
                before=0,
                after=0,
                policy=policy,
                provider=provider,
                parameter="none",
                action="skip",
            )
            return OutputControlResult(body, (event,), policy, None)

        payload = _json_object(body)
        if payload is None:
            event = _event(
                kind="output_budget_skipped_invalid_json",
                path="$",
                before=0,
                after=0,
                policy=policy,
                provider=provider,
                parameter="none",
                action="skip",
            )
            return OutputControlResult(body, (event,), policy, limit)

        parameter = self._parameter(provider, payload)
        if parameter is None:
            event = _event(
                kind="output_budget_skipped_ambiguous_limit",
                path="$",
                before=0,
                after=0,
                policy=policy,
                provider=provider,
                parameter="multiple",
                action="skip",
            )
            return OutputControlResult(body, (event,), policy, limit)
        if parameter not in payload:
            payload[parameter] = limit
            event = _event(
                kind="output_token_limit_set",
                path=parameter,
                before=0,
                after=limit,
                policy=policy,
                provider=provider,
                parameter=parameter,
                action="set",
            )
            return OutputControlResult(_encode(payload), (event,), policy, limit)

        existing = payload[parameter]
        if not _valid_token_limit(existing):
            event = _event(
                kind="output_budget_skipped_invalid_limit",
                path=parameter,
                before=0,
                after=0,
                policy=policy,
                provider=provider,
                parameter=parameter,
                action="skip",
            )
            return OutputControlResult(body, (event,), policy, limit)

        current_limit = cast(int, existing)
        if current_limit <= limit:
            event = _event(
                kind="output_token_limit_preserved",
                path=parameter,
                before=current_limit,
                after=current_limit,
                policy=policy,
                provider=provider,
                parameter=parameter,
                action="preserve",
            )
            return OutputControlResult(body, (event,), policy, limit)

        payload[parameter] = limit
        event = _event(
            kind="output_token_limit_capped",
            path=parameter,
            before=current_limit,
            after=limit,
            policy=policy,
            provider=provider,
            parameter=parameter,
            action="cap",
        )
        return OutputControlResult(_encode(payload), (event,), policy, limit)

    def _resolve_policy(self, requested_policy: str | None) -> OutputPolicyName | None:
        if requested_policy is None:
            return self.config.default_policy
        allowed = get_args(OutputPolicyName)
        if requested_policy not in allowed:
            choices = ", ".join(allowed)
            raise OutputPolicyError(
                f"Unknown output policy '{requested_policy}'. Expected one of: {choices}."
            )
        return cast(OutputPolicyName, requested_policy)

    def _parameter(self, provider: str, payload: dict[str, Any]) -> str | None:
        if provider == "anthropic":
            return "max_tokens"
        if "max_completion_tokens" in payload and "max_tokens" in payload:
            return None
        if "max_completion_tokens" in payload:
            return "max_completion_tokens"
        if "max_tokens" in payload:
            return "max_tokens"
        return self.config.openai_parameter


def _json_object(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _valid_token_limit(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _event(
    *,
    kind: str,
    path: str,
    before: int,
    after: int,
    policy: OutputPolicyName,
    provider: str,
    parameter: str,
    action: str,
) -> OptimizationEvent:
    return OptimizationEvent(
        kind=kind,
        path=path,
        before_estimated_tokens=before,
        after_estimated_tokens=after,
        detail=(f"policy={policy};provider={provider};parameter={parameter};action={action}"),
    )
