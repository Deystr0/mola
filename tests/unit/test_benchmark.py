import json
from pathlib import Path

import pytest

from optimizer.benchmark import load_benchmark, render_benchmark


def test_benchmark_reports_cost_quality_tokens_and_latency(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "id": "one",
            "direct": {
                "success": True,
                "cost_usd": 0.10,
                "input_tokens": 100,
                "output_tokens": 20,
                "latency_ms": 1000,
            },
            "optimized": {
                "success": True,
                "cost_usd": 0.04,
                "input_tokens": 50,
                "output_tokens": 10,
                "latency_ms": 700,
            },
        },
        {
            "id": "two",
            "direct": {
                "success": True,
                "cost_usd": 0.20,
                "input_tokens": 200,
                "output_tokens": 40,
                "latency_ms": 2000,
            },
            "optimized": {
                "success": False,
                "cost_usd": 0.02,
                "input_tokens": 25,
                "output_tokens": 5,
                "latency_ms": 500,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = load_benchmark(path)
    output = render_benchmark(result)

    assert result.direct.successes == 2
    assert result.optimized.successes == 1
    assert result.optimized.cost_per_successful_task is not None
    assert "Success rate" in output
    assert "Cost/success" in output
    assert "300" in output
    assert "75" in output


def test_benchmark_rejects_missing_quality_signal(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text('{"direct": {}, "optimized": {}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="success must be boolean"):
        load_benchmark(path)
