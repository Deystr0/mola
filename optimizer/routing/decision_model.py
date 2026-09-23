from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from optimizer.config import DecisionModelConfig
from optimizer.context import OptimizationEvent
from optimizer.routing.rule_router import (
    ROUTE_NAMES,
    RouteDecision,
    RouteFeatures,
    RouteName,
    RuleRouter,
)

FEATURE_NAMES = (
    "log_input_tokens",
    "message_count",
    "tool_count",
    "has_media",
    "streaming",
)


class DecisionModelArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RoutePrediction:
    route: RouteName
    confidence: float
    probabilities: dict[RouteName, float]


@dataclass(frozen=True, slots=True)
class _RouteWeights:
    bias: float
    weights: dict[str, float]


class LocalDecisionModel:
    """Small local multiclass linear model loaded from a versioned JSON artifact."""

    def __init__(
        self,
        *,
        route_weights: dict[RouteName, _RouteWeights],
        validation_samples: int,
        validation_accuracy: float,
    ) -> None:
        self.route_weights = route_weights
        self.validation_samples = validation_samples
        self.validation_accuracy = validation_accuracy

    @classmethod
    def load(cls, path: Path) -> LocalDecisionModel:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DecisionModelArtifactError(
                f"Cannot load decision-model artifact: {type(error).__name__}."
            ) from error
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise DecisionModelArtifactError("Decision-model artifact must use schema_version 1.")
        validation = raw.get("validation")
        if not isinstance(validation, dict):
            raise DecisionModelArtifactError("Decision-model artifact is missing validation data.")
        samples = validation.get("samples")
        accuracy = validation.get("accuracy")
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 0:
            raise DecisionModelArtifactError("validation.samples must be a non-negative integer.")
        if (
            not isinstance(accuracy, (int, float))
            or isinstance(accuracy, bool)
            or not 0 <= float(accuracy) <= 1
        ):
            raise DecisionModelArtifactError("validation.accuracy must be between 0 and 1.")

        raw_routes = raw.get("routes")
        if not isinstance(raw_routes, dict) or set(raw_routes) != set(ROUTE_NAMES):
            raise DecisionModelArtifactError(
                "Decision-model artifact must define exactly: " + ", ".join(ROUTE_NAMES) + "."
            )
        route_weights: dict[RouteName, _RouteWeights] = {}
        for route_name in ROUTE_NAMES:
            route = cast(RouteName, route_name)
            value = raw_routes[route]
            if not isinstance(value, dict):
                raise DecisionModelArtifactError(f"Route '{route}' must be an object.")
            bias = _finite_number(value.get("bias"), f"routes.{route}.bias")
            weights = value.get("weights")
            if not isinstance(weights, dict) or set(weights) != set(FEATURE_NAMES):
                raise DecisionModelArtifactError(
                    f"routes.{route}.weights must define exactly: " + ", ".join(FEATURE_NAMES) + "."
                )
            route_weights[route] = _RouteWeights(
                bias=bias,
                weights={
                    name: _finite_number(weights[name], f"routes.{route}.weights.{name}")
                    for name in FEATURE_NAMES
                },
            )
        return cls(
            route_weights=route_weights,
            validation_samples=samples,
            validation_accuracy=float(accuracy),
        )

    def predict(self, features: RouteFeatures) -> RoutePrediction:
        values = _feature_values(features)
        scores = {
            route: weights.bias
            + sum(weights.weights[name] * values[name] for name in FEATURE_NAMES)
            for route, weights in self.route_weights.items()
        }
        if not all(math.isfinite(score) for score in scores.values()):
            raise ValueError("Decision-model inference produced a non-finite score.")
        maximum = max(scores.values())
        exponentials = {route: math.exp(score - maximum) for route, score in scores.items()}
        denominator = sum(exponentials.values())
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("Decision-model inference produced invalid probabilities.")
        probabilities = {route: value / denominator for route, value in exponentials.items()}
        selected = max(probabilities, key=probabilities.__getitem__)
        return RoutePrediction(selected, probabilities[selected], probabilities)

    def has_activation_evidence(self, config: DecisionModelConfig) -> bool:
        return (
            self.validation_samples >= config.min_validation_samples
            and self.validation_accuracy >= config.min_validation_accuracy
        )


class DecisionRouter:
    """Combine the production rule router with optional shadow or active predictions."""

    def __init__(
        self,
        rule_router: RuleRouter,
        config: DecisionModelConfig,
        model: LocalDecisionModel | None,
    ) -> None:
        self.rule_router = rule_router
        self.config = config
        self.model = model

    def route(
        self,
        *,
        provider: str,
        body: bytes,
        requested_route: str | None,
    ) -> RouteDecision:
        rule = self.rule_router.route(
            provider=provider,
            body=body,
            requested_route=requested_route,
        )
        if not rule.enabled or not self.config.enabled or self.model is None:
            return rule
        try:
            prediction = self.model.predict(rule.features)
        except Exception:
            return replace(
                rule,
                model_mode=self.config.mode,
                model_fallback_reason="prediction_error",
            )

        common = {
            "model_route": prediction.route,
            "model_confidence": prediction.confidence,
            "model_mode": self.config.mode,
        }
        if self.config.mode == "shadow":
            return replace(rule, **common)
        fallback = self._active_fallback(
            provider=provider,
            body=body,
            prediction=prediction,
            requested_route=requested_route,
        )
        if fallback is not None:
            return replace(rule, **common, model_fallback_reason=fallback)

        active = self.rule_router.route(
            provider=provider,
            body=body,
            requested_route=prediction.route,
        )
        event = OptimizationEvent(
            kind="route_selected",
            path="model",
            before_estimated_tokens=active.features.estimated_input_tokens,
            after_estimated_tokens=active.features.estimated_input_tokens,
            detail=(
                f"route={prediction.route};source=decision_model;reason=active_prediction;"
                f"model_configured={str(active.model_configured).lower()};"
                f"model_changed={str(active.model_changed).lower()}"
            ),
        )
        return replace(
            active,
            source="decision_model",
            reason="active_prediction",
            event=event,
            rule_route=rule.route,
            model_applied=True,
            **common,
        )

    def _active_fallback(
        self,
        *,
        provider: str,
        body: bytes,
        prediction: RoutePrediction,
        requested_route: str | None,
    ) -> str | None:
        if requested_route is not None:
            return "explicit_route"
        assert self.model is not None
        if not self.model.has_activation_evidence(self.config):
            return "insufficient_validation_evidence"
        if prediction.confidence < self.config.min_confidence:
            return "low_confidence"
        if prediction.route == "cache":
            return "cache_checked_implicitly"
        candidate = self.rule_router.route(
            provider=provider,
            body=body,
            requested_route=prediction.route,
        )
        if not candidate.model_configured:
            return "model_unconfigured"
        if not candidate.model_changed and candidate.original_model is None:
            return "request_not_routable"
        return None


def _feature_values(features: RouteFeatures) -> dict[str, float]:
    return {
        "log_input_tokens": math.log1p(features.estimated_input_tokens),
        "message_count": min(features.message_count, 100) / 10,
        "tool_count": min(features.tool_count, 20) / 5,
        "has_media": float(features.has_media),
        "streaming": float(features.streaming),
    }


def _finite_number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DecisionModelArtifactError(f"{path} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise DecisionModelArtifactError(f"{path} must be a finite number.")
    return result
