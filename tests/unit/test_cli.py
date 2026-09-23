import json
from argparse import Namespace
from decimal import Decimal
from pathlib import Path

from optimizer.cache import SQLiteExactCache, SQLiteSemanticCache
from optimizer.cli.main import _dispatch, format_stats
from optimizer.embeddings import HashingEmbeddingBackend
from optimizer.telemetry import Stats


def test_format_stats_reports_usage_and_unpriced_requests() -> None:
    output = format_stats(
        Stats(
            requests=2,
            successful_requests=1,
            failed_requests=1,
            input_tokens=1200,
            output_tokens=300,
            cache_hits=0,
            baseline_cost_usd=Decimal("0.01"),
            actual_cost_usd=Decimal("0.01"),
            priced_requests=1,
            estimated_input_tokens_removed=25,
            optimization_events=2,
        )
    )

    assert "Requests                     2" in output
    assert "Input tokens             1,200" in output
    assert "Estimated removed           25" in output
    assert "Optimization events          2" in output
    assert "Exact                     0" in output
    assert "Semantic                  0" in output
    assert "Routing decisions           0" in output
    assert "ML predictions           0" in output
    assert "Saved                $0.000000" in output
    assert "1 request(s) used unpriced models" in output


def test_cache_clear_removes_entries_even_when_cache_is_disabled_in_config(
    tmp_path: Path,
    capsys,
) -> None:
    cache_path = tmp_path / "cache.db"
    cache = SQLiteExactCache(cache_path)
    cache.initialize()
    cache.set(
        "key",
        provider="openai",
        model="test-model",
        body=b"{}",
        status_code=200,
        content_type="application/json",
        input_tokens=0,
        output_tokens=0,
        ttl_seconds=60,
    )
    config_path = tmp_path / "optimizer.yaml"
    config_path.write_text(
        "optimization:\n"
        "  cache:\n"
        "    exact:\n"
        "      enabled: false\n"
        f"      database_path: {cache_path}\n"
        "      ttl_seconds: 3600\n",
        encoding="utf-8",
    )

    result = _dispatch(Namespace(command="cache", cache_command="clear", config=config_path))

    assert result == 0
    assert cache.get("key") is None
    assert "Cleared 1 exact-cache entry." in capsys.readouterr().out


def test_semantic_cache_clear_removes_entries_even_when_disabled_in_config(
    tmp_path: Path,
    capsys,
) -> None:
    cache_path = tmp_path / "semantic-cache.db"
    cache = SQLiteSemanticCache(cache_path)
    cache.initialize()
    embedding_backend = HashingEmbeddingBackend()
    cache.set(
        "key",
        compatibility_key="scope",
        embedding_version=embedding_backend.version,
        embedding=embedding_backend.embed("reset password"),
        provider="openai",
        model="test-model",
        body=b"{}",
        status_code=200,
        content_type="application/json",
        input_tokens=0,
        output_tokens=0,
        ttl_seconds=60,
    )
    config_path = tmp_path / "optimizer.yaml"
    config_path.write_text(
        "optimization:\n"
        "  cache:\n"
        "    semantic:\n"
        "      enabled: false\n"
        f"      database_path: {cache_path}\n"
        "      ttl_seconds: 900\n",
        encoding="utf-8",
    )

    result = _dispatch(
        Namespace(
            command="cache",
            cache_command="clear",
            kind="semantic",
            config=config_path,
        )
    )

    assert result == 0
    assert (
        cache.search(
            compatibility_key="scope",
            embedding_version=embedding_backend.version,
            embedding=embedding_backend.embed("reset password"),
            threshold=0,
            limit=10,
        )
        is None
    )
    assert "Cleared 1 semantic-cache entry." in capsys.readouterr().out


def test_doctor_reports_enabled_context_intelligence(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "optimizer.yaml"
    config_path.write_text(
        "telemetry:\n"
        "  enabled: false\n"
        "optimization:\n"
        "  input:\n"
        "    max_estimated_tokens: 4096\n"
        "    context:\n"
        "      enabled: true\n"
        "      min_recent_turns: 3\n"
        "      max_prunable_score: 0.25\n",
        encoding="utf-8",
    )

    result = _dispatch(Namespace(command="doctor", config=config_path))

    assert result == 0
    assert (
        "[ok] Context intelligence enabled: budget 4,096 estimated tokens; "
        "keep 3 recent turns; maximum prunable score 0.250" in capsys.readouterr().out
    )


def test_benchmark_command_renders_recorded_comparison(tmp_path: Path, capsys) -> None:
    path = tmp_path / "results.jsonl"
    side = {
        "success": True,
        "cost_usd": 0.01,
        "input_tokens": 10,
        "output_tokens": 2,
        "latency_ms": 25,
    }
    path.write_text(
        json.dumps({"direct": side, "optimized": side}) + "\n",
        encoding="utf-8",
    )

    result = _dispatch(Namespace(command="benchmark", path=path))

    assert result == 0
    output = capsys.readouterr().out
    assert "Cost/success" in output
    assert "Success rate" in output
