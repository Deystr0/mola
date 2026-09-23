from __future__ import annotations

import json
from pathlib import Path

import pytest

from optimizer.config import (
    DecisionModelConfig,
    ProviderRouteModels,
    RoutingModelsConfig,
    RoutingOptimizationConfig,
)
from optimizer.routing import (
    FEATURE_NAMES,
    ROUTE_NAMES,
    DecisionModelArtifactError,
    DecisionRouter,
    LocalDecisionModel,
    RuleRouter,
)


def _artifact(
    path: Path,
    *,
    preferred: str = "frontier",
    samples: int = 500,
    accuracy: float = 0.92,
    tied: bool = False,
) -> Path:
    routes = {}
    for route in ROUTE_NAMES:
        routes[route] = {
            "bias": 0 if tied else (10 if route == preferred else -10),
            "weights": {name: 0 for name in FEATURE_NAMES},
        }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "validation": {"samples": samples, "accuracy": accuracy},
                "routes": routes,
            }
        ),
        encoding="utf-8",
    )
    return path


def _rule_router(*, frontier_model: str | None = "frontier-model") -> RuleRouter:
    return RuleRouter(
        RoutingOptimizationConfig(
            enabled=True,
            models=RoutingModelsConfig(
                openai=ProviderRouteModels(
                    cheap="cheap-model",
                    mid="mid-model",
                    frontier=frontier_model,
                )
            ),
        )
    )


def _body() -> bytes:
    return json.dumps(
        {
            "model": "caller-model",
            "messages": [{"role": "user", "content": "short request"}],
        }
    ).encode()


def test_shadow_model_records_prediction_without_changing_rule_route(tmp_path: Path) -> None:
    model = LocalDecisionModel.load(_artifact(tmp_path / "model.json"))
    router = DecisionRouter(
        _rule_router(),
        DecisionModelConfig(enabled=True, mode="shadow", artifact_path=tmp_path / "model.json"),
        model,
    )

    decision = router.route(provider="openai", body=_body(), requested_route=None)

    assert decision.route == "cheap"
    assert decision.routed_model == "cheap-model"
    assert decision.rule_route == "cheap"
    assert decision.model_route == "frontier"
    assert decision.model_confidence is not None and decision.model_confidence > 0.99
    assert decision.model_mode == "shadow"
    assert not decision.model_applied


def test_active_model_routes_when_evidence_confidence_and_target_are_available(
    tmp_path: Path,
) -> None:
    path = _artifact(tmp_path / "model.json")
    router = DecisionRouter(
        _rule_router(),
        DecisionModelConfig(enabled=True, mode="active", artifact_path=path),
        LocalDecisionModel.load(path),
    )

    decision = router.route(provider="openai", body=_body(), requested_route=None)

    assert decision.route == "frontier"
    assert decision.routed_model == "frontier-model"
    assert decision.rule_route == "cheap"
    assert decision.model_route == "frontier"
    assert decision.model_applied
    assert decision.source == "decision_model"


@pytest.mark.parametrize(
    ("artifact_options", "frontier_model", "expected_reason"),
    [
        ({"samples": 10}, "frontier-model", "insufficient_validation_evidence"),
        ({"tied": True}, "frontier-model", "low_confidence"),
        ({}, None, "model_unconfigured"),
        ({"preferred": "cache"}, "frontier-model", "cache_checked_implicitly"),
    ],
)
def test_active_model_falls_back_to_rule_router(
    tmp_path: Path,
    artifact_options: dict,
    frontier_model: str | None,
    expected_reason: str,
) -> None:
    path = _artifact(tmp_path / "model.json", **artifact_options)
    router = DecisionRouter(
        _rule_router(frontier_model=frontier_model),
        DecisionModelConfig(enabled=True, mode="active", artifact_path=path),
        LocalDecisionModel.load(path),
    )

    decision = router.route(provider="openai", body=_body(), requested_route=None)

    assert decision.route == "cheap"
    assert not decision.model_applied
    assert decision.model_fallback_reason == expected_reason


def test_explicit_route_wins_over_active_model(tmp_path: Path) -> None:
    path = _artifact(tmp_path / "model.json")
    router = DecisionRouter(
        _rule_router(),
        DecisionModelConfig(enabled=True, mode="active", artifact_path=path),
        LocalDecisionModel.load(path),
    )

    decision = router.route(provider="openai", body=_body(), requested_route="mid")

    assert decision.route == "mid"
    assert decision.rule_route == "mid"
    assert decision.model_route == "frontier"
    assert decision.model_fallback_reason == "explicit_route"


def test_artifact_validation_rejects_incomplete_or_nonfinite_weights(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(DecisionModelArtifactError):
        LocalDecisionModel.load(invalid)

    path = _artifact(tmp_path / "nonfinite.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["routes"]["cheap"]["bias"] = float("inf")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DecisionModelArtifactError):
        LocalDecisionModel.load(path)


def test_active_model_falls_back_when_request_cannot_be_rewritten(tmp_path: Path) -> None:
    path = _artifact(tmp_path / "model.json")
    router = DecisionRouter(
        _rule_router(),
        DecisionModelConfig(enabled=True, mode="active", artifact_path=path),
        LocalDecisionModel.load(path),
    )

    decision = router.route(provider="openai", body=b"not-json", requested_route=None)

    assert decision.route == "cheap"
    assert not decision.model_applied
    assert decision.routed_model is None
    assert decision.model_fallback_reason == "request_not_routable"


def test_nonfinite_inference_falls_back_to_rule_router(tmp_path: Path) -> None:
    path = _artifact(tmp_path / "model.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["routes"]["frontier"]["weights"]["log_input_tokens"] = 1e308
    path.write_text(json.dumps(payload), encoding="utf-8")
    router = DecisionRouter(
        _rule_router(),
        DecisionModelConfig(enabled=True, mode="active", artifact_path=path),
        LocalDecisionModel.load(path),
    )

    decision = router.route(provider="openai", body=_body(), requested_route=None)

    assert decision.route == "cheap"
    assert not decision.model_applied
    assert decision.model_fallback_reason == "prediction_error"
