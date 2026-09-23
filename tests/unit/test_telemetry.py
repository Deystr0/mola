import sqlite3
import stat
from decimal import Decimal
from pathlib import Path

from optimizer.context import OptimizationEvent
from optimizer.telemetry import (
    RequestRecord,
    RoutingDecisionRecord,
    TelemetryStore,
    ValidationAttemptRecord,
)


def test_telemetry_aggregates_metadata_without_content(tmp_path: Path) -> None:
    store = TelemetryStore(tmp_path / "telemetry.db")
    store.initialize()
    store.record(
        RequestRecord(
            request_id="request-1",
            provider="openai",
            model="test-model",
            status="success",
            status_code=200,
            latency_ms=12.5,
            input_tokens=100,
            output_tokens=20,
            baseline_cost_usd=Decimal("0.004"),
            actual_cost_usd=Decimal("0.004"),
            cache_hit=True,
            cache_type="semantic",
            original_estimated_input_tokens=50,
            optimized_estimated_input_tokens=30,
            validated=True,
            escalated=True,
            escalation_attempts=1,
            escalation_cost_usd=Decimal("0.001"),
        ),
        (
            OptimizationEvent(
                kind="duplicate_message_removed",
                path="messages[2]",
                before_estimated_tokens=20,
                after_estimated_tokens=0,
                detail="duplicate_of=messages[1]",
            ),
        ),
        RoutingDecisionRecord(
            request_id="request-1",
            selected_route="cheap",
            executed_route="cheap",
            source="heuristic",
            reason="small_simple_request",
            original_model="caller-model",
            routed_model="cheap-model",
            model_configured=True,
            model_changed=True,
            estimated_input_tokens=50,
            message_count=1,
            tool_count=0,
            has_media=False,
            streaming=False,
            cache_only=False,
            rule_route="cheap",
            model_route="cheap",
            model_confidence=0.91,
            model_mode="shadow",
            model_agreed=True,
        ),
        (
            ValidationAttemptRecord(
                request_id="request-1",
                attempt=1,
                route="cheap",
                model="test-model",
                status_code=200,
                passed=False,
                reason="truncated",
                input_tokens=50,
                output_tokens=10,
                cost_usd=Decimal("0.001"),
                latency_ms=10,
            ),
            ValidationAttemptRecord(
                request_id="request-1",
                attempt=2,
                route="mid",
                model="test-model",
                status_code=200,
                passed=True,
                reason="passed",
                input_tokens=50,
                output_tokens=10,
                cost_usd=Decimal("0.003"),
                latency_ms=20,
            ),
        ),
    )

    stats = store.stats()

    assert stats.requests == 1
    assert stats.successful_requests == 1
    assert stats.input_tokens == 100
    assert stats.output_tokens == 20
    assert stats.actual_cost_usd == Decimal("0.004")
    assert stats.cache_hits == 1
    assert stats.semantic_cache_hits == 1
    assert stats.exact_cache_hits == 0
    assert stats.estimated_input_tokens_removed == 20
    assert stats.optimization_events == 1
    assert stats.routing_decisions == 1
    assert stats.routed_model_changes == 1
    assert stats.model_predictions == 1
    assert stats.model_agreements == 1
    assert stats.validated_requests == 1
    assert stats.escalated_requests == 1
    assert stats.escalation_attempts == 1
    assert stats.escalation_cost_usd == Decimal("0.001")
    assert store.events_for_request("request-1")[0].path == "messages[2]"
    routing = store.routing_for_request("request-1")
    assert routing is not None
    assert routing.selected_route == "cheap"
    assert routing.executed_route == "cheap"
    assert routing.model_confidence == 0.91
    validations = store.validations_for_request("request-1")
    assert [attempt.reason for attempt in validations] == ["truncated", "passed"]


def test_telemetry_database_is_owner_only_and_repairs_existing_permissions(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.db"
    path.touch(mode=0o644)
    path.chmod(0o644)

    TelemetryStore(path).initialize()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_telemetry_schema_expands_an_existing_v02_database(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE requests (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT,
                status TEXT NOT NULL,
                status_code INTEGER,
                latency_ms REAL NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                baseline_cost_usd TEXT,
                actual_cost_usd TEXT,
                cache_hit INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    store = TelemetryStore(path)
    # Reading stats is sufficient to apply the additive migration.
    store.stats()

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(requests)")}
        event_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'optimization_events'"
        ).fetchone()
        validation_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'validation_attempts'"
        ).fetchone()

    assert "original_estimated_input_tokens" in columns
    assert "optimized_estimated_input_tokens" in columns
    assert "cache_type" in columns
    assert "validated" in columns
    assert "escalated" in columns
    assert "escalation_attempts" in columns
    assert "escalation_cost_usd" in columns
    assert event_table == ("optimization_events",)
    assert validation_table == ("validation_attempts",)


def test_telemetry_expands_v08_routing_table_for_decision_model_fields(tmp_path: Path) -> None:
    path = tmp_path / "v08.db"
    store = TelemetryStore(path)
    store.initialize()
    with sqlite3.connect(path) as connection:
        for column in (
            "rule_route",
            "model_route",
            "model_confidence",
            "model_mode",
            "model_applied",
            "model_fallback_reason",
            "model_agreed",
        ):
            connection.execute(f"ALTER TABLE routing_decisions DROP COLUMN {column}")

    store.initialize()

    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(routing_decisions)").fetchall()
        }

    assert {
        "rule_route",
        "model_route",
        "model_confidence",
        "model_mode",
        "model_applied",
        "model_fallback_reason",
        "model_agreed",
    } <= columns
