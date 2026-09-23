from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from optimizer.config import PricingConfig


@dataclass(frozen=True, slots=True)
class RequestCost:
    baseline_usd: Decimal | None
    actual_usd: Decimal | None


class CostCalculator:
    """Calculates configured costs without guessing prices for unknown models."""

    def __init__(self, pricing: PricingConfig) -> None:
        self._pricing = pricing

    def calculate(self, model: str | None, input_tokens: int, output_tokens: int) -> RequestCost:
        price = self._pricing.models.get(model or "")
        if price is None:
            return RequestCost(baseline_usd=None, actual_usd=None)

        divisor = Decimal(1_000_000)
        input_cost = Decimal(str(price.input_per_million)) * Decimal(input_tokens) / divisor
        output_cost = Decimal(str(price.output_per_million)) * Decimal(output_tokens) / divisor
        total = input_cost + output_cost
        # Transparent forwarding has no savings: baseline and actual are identical.
        return RequestCost(baseline_usd=total, actual_usd=total)
