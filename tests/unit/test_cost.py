from decimal import Decimal

from optimizer.config import ModelPrice, PricingConfig
from optimizer.cost import CostCalculator


def test_cost_calculator_uses_configured_prices() -> None:
    calculator = CostCalculator(
        PricingConfig(
            models={
                "test-model": ModelPrice(input_per_million=2, output_per_million=8),
            }
        )
    )

    cost = calculator.calculate("test-model", input_tokens=1_000, output_tokens=250)

    assert cost.baseline_usd == Decimal("0.004")
    assert cost.actual_usd == Decimal("0.004")


def test_cost_calculator_does_not_guess_unknown_prices() -> None:
    cost = CostCalculator(PricingConfig()).calculate("unknown", 100, 100)

    assert cost.baseline_usd is None
    assert cost.actual_usd is None
