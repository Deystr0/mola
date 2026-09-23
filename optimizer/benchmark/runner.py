from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkAggregate:
    runs: int
    successes: int
    total_cost_usd: Decimal
    input_tokens: int
    output_tokens: int
    median_latency_ms: float

    @property
    def success_rate(self) -> float:
        return self.successes / self.runs if self.runs else 0

    @property
    def cost_per_successful_task(self) -> Decimal | None:
        if not self.successes:
            return None
        return self.total_cost_usd / self.successes


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    direct: BenchmarkAggregate
    optimized: BenchmarkAggregate


def load_benchmark(path: Path) -> BenchmarkComparison:
    direct: list[dict[str, Any]] = []
    optimized: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid benchmark JSON on line {line_number}.") from error
        if not isinstance(row, dict):
            raise ValueError(f"Benchmark line {line_number} must be an object.")
        direct.append(_sample(row.get("direct"), line_number, "direct"))
        optimized.append(_sample(row.get("optimized"), line_number, "optimized"))
    if not direct:
        raise ValueError("Benchmark file contains no cases.")
    return BenchmarkComparison(_aggregate(direct), _aggregate(optimized))


def render_benchmark(comparison: BenchmarkComparison) -> str:
    direct = comparison.direct
    optimized = comparison.optimized
    return "\n".join(
        [
            "Metric                         Direct       Optimized",
            f"Tasks                    {direct.runs:>10,} {optimized.runs:>15,}",
            f"Successful               {direct.successes:>10,} {optimized.successes:>15,}",
            f"Success rate             {direct.success_rate:>9.2%} {optimized.success_rate:>14.2%}",
            f"Total cost              {_money(direct.total_cost_usd):>10} "
            f"{_money(optimized.total_cost_usd):>15}",
            f"Cost/success            {_optional_money(direct.cost_per_successful_task):>10} "
            f"{_optional_money(optimized.cost_per_successful_task):>15}",
            f"Input tokens             {direct.input_tokens:>10,} {optimized.input_tokens:>15,}",
            f"Output tokens            {direct.output_tokens:>10,} {optimized.output_tokens:>15,}",
            f"Median latency ms        {direct.median_latency_ms:>10.2f} "
            f"{optimized.median_latency_ms:>15.2f}",
        ]
    )


def _sample(value: Any, line_number: int, side: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark line {line_number} is missing '{side}'.")
    success = value.get("success")
    if not isinstance(success, bool):
        raise ValueError(f"Benchmark line {line_number} {side}.success must be boolean.")
    result: dict[str, Any] = {"success": success}
    for field in ("input_tokens", "output_tokens"):
        item = value.get(field)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(
                f"Benchmark line {line_number} {side}.{field} must be non-negative integer."
            )
        result[field] = item
    latency = value.get("latency_ms")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency < 0:
        raise ValueError(f"Benchmark line {line_number} {side}.latency_ms must be non-negative.")
    result["latency_ms"] = float(latency)
    try:
        cost = Decimal(str(value.get("cost_usd")))
    except Exception as error:
        raise ValueError(
            f"Benchmark line {line_number} {side}.cost_usd must be numeric."
        ) from error
    if not cost.is_finite() or cost < 0:
        raise ValueError(
            f"Benchmark line {line_number} {side}.cost_usd must be non-negative and finite."
        )
    result["cost_usd"] = cost
    return result


def _aggregate(samples: list[dict[str, Any]]) -> BenchmarkAggregate:
    return BenchmarkAggregate(
        runs=len(samples),
        successes=sum(bool(sample["success"]) for sample in samples),
        total_cost_usd=sum((sample["cost_usd"] for sample in samples), Decimal(0)),
        input_tokens=sum(int(sample["input_tokens"]) for sample in samples),
        output_tokens=sum(int(sample["output_tokens"]) for sample in samples),
        median_latency_ms=statistics.median(float(sample["latency_ms"]) for sample in samples),
    )


def _money(value: Decimal) -> str:
    return f"${value:.6f}"


def _optional_money(value: Decimal | None) -> str:
    return "n/a" if value is None else _money(value)
