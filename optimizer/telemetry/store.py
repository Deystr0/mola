from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from optimizer.context import OptimizationEvent


@dataclass(frozen=True, slots=True)
class RequestRecord:
    request_id: str
    provider: str
    model: str | None
    status: str
    status_code: int | None
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    baseline_cost_usd: Decimal | None = None
    actual_cost_usd: Decimal | None = None
    cache_hit: bool = False
    cache_type: str | None = None
    original_estimated_input_tokens: int = 0
    optimized_estimated_input_tokens: int = 0
    validated: bool | None = None
    escalated: bool = False
    escalation_attempts: int = 0
    escalation_cost_usd: Decimal | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RoutingDecisionRecord:
    request_id: str
    selected_route: str
    executed_route: str | None
    source: str
    reason: str
    original_model: str | None
    routed_model: str | None
    model_configured: bool
    model_changed: bool
    estimated_input_tokens: int
    message_count: int
    tool_count: int
    has_media: bool
    streaming: bool
    cache_only: bool
    rule_route: str | None = None
    model_route: str | None = None
    model_confidence: float | None = None
    model_mode: str | None = None
    model_applied: bool = False
    model_fallback_reason: str | None = None
    model_agreed: bool | None = None


@dataclass(frozen=True, slots=True)
class ValidationAttemptRecord:
    request_id: str
    attempt: int
    route: str | None
    model: str | None
    status_code: int
    passed: bool | None
    reason: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None
    latency_ms: float


@dataclass(frozen=True, slots=True)
class Stats:
    requests: int
    successful_requests: int
    failed_requests: int
    input_tokens: int
    output_tokens: int
    cache_hits: int
    baseline_cost_usd: Decimal
    actual_cost_usd: Decimal
    priced_requests: int
    estimated_input_tokens_removed: int = 0
    optimization_events: int = 0
    exact_cache_hits: int = 0
    semantic_cache_hits: int = 0
    routing_decisions: int = 0
    routed_model_changes: int = 0
    model_predictions: int = 0
    model_agreements: int = 0
    active_model_routes: int = 0
    validated_requests: int = 0
    validation_failures: int = 0
    escalated_requests: int = 0
    escalation_attempts: int = 0
    escalation_cost_usd: Decimal = Decimal(0)

    @property
    def saved_usd(self) -> Decimal:
        return self.baseline_cost_usd - self.actual_cost_usd


class TelemetryStore:
    """SQLite request ledger that intentionally excludes prompt and response bodies."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path.expanduser()
        self.enabled = enabled

    def initialize(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=True)
        os.chmod(self.path, 0o600)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT,
                    status TEXT NOT NULL,
                    status_code INTEGER,
                    latency_ms REAL NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
                    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
                    baseline_cost_usd TEXT,
                    actual_cost_usd TEXT,
                    cache_hit INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0, 1)),
                    cache_type TEXT,
                    original_estimated_input_tokens INTEGER NOT NULL DEFAULT 0,
                    optimized_estimated_input_tokens INTEGER NOT NULL DEFAULT 0,
                    validated INTEGER CHECK (validated IN (0, 1)),
                    escalated INTEGER NOT NULL DEFAULT 0 CHECK (escalated IN (0, 1)),
                    escalation_attempts INTEGER NOT NULL DEFAULT 0 CHECK (escalation_attempts >= 0),
                    escalation_cost_usd TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS routing_decisions (
                    request_id TEXT PRIMARY KEY,
                    selected_route TEXT NOT NULL,
                    executed_route TEXT,
                    source TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    original_model TEXT,
                    routed_model TEXT,
                    model_configured INTEGER NOT NULL CHECK (model_configured IN (0, 1)),
                    model_changed INTEGER NOT NULL CHECK (model_changed IN (0, 1)),
                    estimated_input_tokens INTEGER NOT NULL CHECK (estimated_input_tokens >= 0),
                    message_count INTEGER NOT NULL CHECK (message_count >= 0),
                    tool_count INTEGER NOT NULL CHECK (tool_count >= 0),
                    has_media INTEGER NOT NULL CHECK (has_media IN (0, 1)),
                    streaming INTEGER NOT NULL CHECK (streaming IN (0, 1)),
                    cache_only INTEGER NOT NULL CHECK (cache_only IN (0, 1)),
                    rule_route TEXT,
                    model_route TEXT,
                    model_confidence REAL,
                    model_mode TEXT,
                    model_applied INTEGER NOT NULL DEFAULT 0 CHECK (model_applied IN (0, 1)),
                    model_fallback_reason TEXT,
                    model_agreed INTEGER CHECK (model_agreed IN (0, 1)),
                    FOREIGN KEY (request_id) REFERENCES requests(request_id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_attempts (
                    request_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL CHECK (attempt >= 1),
                    route TEXT,
                    model TEXT,
                    status_code INTEGER NOT NULL,
                    passed INTEGER CHECK (passed IN (0, 1)),
                    reason TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
                    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
                    cost_usd TEXT,
                    latency_ms REAL NOT NULL CHECK (latency_ms >= 0),
                    PRIMARY KEY (request_id, attempt),
                    FOREIGN KEY (request_id) REFERENCES requests(request_id) ON DELETE CASCADE
                )
                """
            )
            self._expand_routing_schema(connection)
            self._expand_requests_schema(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS requests_created_at_idx ON requests(created_at)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_events (
                    request_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    before_estimated_tokens INTEGER NOT NULL,
                    after_estimated_tokens INTEGER NOT NULL,
                    detail TEXT NOT NULL,
                    PRIMARY KEY (request_id, sequence),
                    FOREIGN KEY (request_id) REFERENCES requests(request_id) ON DELETE CASCADE
                )
                """
            )

    def record(
        self,
        record: RequestRecord,
        events: tuple[OptimizationEvent, ...] = (),
        routing: RoutingDecisionRecord | None = None,
        validation_attempts: tuple[ValidationAttemptRecord, ...] = (),
    ) -> None:
        if not self.enabled:
            return
        created_at = record.created_at or datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO requests (
                    request_id, created_at, provider, model, status, status_code,
                    latency_ms, input_tokens, output_tokens, baseline_cost_usd,
                    actual_cost_usd, cache_hit, cache_type, original_estimated_input_tokens,
                    optimized_estimated_input_tokens, validated, escalated,
                    escalation_attempts, escalation_cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    created_at.isoformat(),
                    record.provider,
                    record.model,
                    record.status,
                    record.status_code,
                    record.latency_ms,
                    record.input_tokens,
                    record.output_tokens,
                    str(record.baseline_cost_usd) if record.baseline_cost_usd is not None else None,
                    str(record.actual_cost_usd) if record.actual_cost_usd is not None else None,
                    int(record.cache_hit),
                    record.cache_type,
                    record.original_estimated_input_tokens,
                    record.optimized_estimated_input_tokens,
                    int(record.validated) if record.validated is not None else None,
                    int(record.escalated),
                    record.escalation_attempts,
                    (
                        str(record.escalation_cost_usd)
                        if record.escalation_cost_usd is not None
                        else None
                    ),
                ),
            )
            connection.executemany(
                """
                INSERT INTO optimization_events (
                    request_id, sequence, kind, path, before_estimated_tokens,
                    after_estimated_tokens, detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.request_id,
                        sequence,
                        event.kind,
                        event.path,
                        event.before_estimated_tokens,
                        event.after_estimated_tokens,
                        event.detail,
                    )
                    for sequence, event in enumerate(events)
                ],
            )
            if routing is not None:
                connection.execute(
                    """
                    INSERT INTO routing_decisions (
                        request_id, selected_route, executed_route, source, reason,
                        original_model, routed_model, model_configured, model_changed,
                        estimated_input_tokens, message_count, tool_count, has_media,
                        streaming, cache_only, rule_route, model_route, model_confidence,
                        model_mode, model_applied, model_fallback_reason, model_agreed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        routing.request_id,
                        routing.selected_route,
                        routing.executed_route,
                        routing.source,
                        routing.reason,
                        routing.original_model,
                        routing.routed_model,
                        int(routing.model_configured),
                        int(routing.model_changed),
                        routing.estimated_input_tokens,
                        routing.message_count,
                        routing.tool_count,
                        int(routing.has_media),
                        int(routing.streaming),
                        int(routing.cache_only),
                        routing.rule_route,
                        routing.model_route,
                        routing.model_confidence,
                        routing.model_mode,
                        int(routing.model_applied),
                        routing.model_fallback_reason,
                        int(routing.model_agreed) if routing.model_agreed is not None else None,
                    ),
                )
            connection.executemany(
                """
                INSERT INTO validation_attempts (
                    request_id, attempt, route, model, status_code, passed, reason,
                    input_tokens, output_tokens, cost_usd, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        attempt.request_id,
                        attempt.attempt,
                        attempt.route,
                        attempt.model,
                        attempt.status_code,
                        int(attempt.passed) if attempt.passed is not None else None,
                        attempt.reason,
                        attempt.input_tokens,
                        attempt.output_tokens,
                        str(attempt.cost_usd) if attempt.cost_usd is not None else None,
                        attempt.latency_ms,
                    )
                    for attempt in validation_attempts
                ],
            )

    def stats(self) -> Stats:
        if not self.enabled or not self.path.exists():
            return Stats(0, 0, 0, 0, 0, 0, Decimal(0), Decimal(0), 0, 0, 0)
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(input_tokens), 0),
                    COALESCE(SUM(output_tokens), 0),
                    COALESCE(SUM(cache_hit), 0),
                    COALESCE(SUM(
                        CASE
                            WHEN cache_hit = 1 AND COALESCE(cache_type, 'exact') = 'exact' THEN 1
                            ELSE 0
                        END
                    ), 0),
                    COALESCE(SUM(CASE WHEN cache_type = 'semantic' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN actual_cost_usd IS NOT NULL THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(
                        CASE
                            WHEN status IN (
                                'success', 'provider_error', 'timeout', 'stream_error', 'cancelled'
                            ) AND original_estimated_input_tokens > optimized_estimated_input_tokens
                            THEN original_estimated_input_tokens - optimized_estimated_input_tokens
                            ELSE 0
                        END
                    ), 0)
                FROM requests
                """
            ).fetchone()
            event_count_row = connection.execute(
                "SELECT COUNT(*) FROM optimization_events"
            ).fetchone()
            routing_row = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(model_changed), 0),
                       COALESCE(SUM(CASE WHEN model_route IS NOT NULL THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(CASE WHEN model_agreed = 1 THEN 1 ELSE 0 END), 0),
                       COALESCE(SUM(model_applied), 0)
                FROM routing_decisions
                """
            ).fetchone()
            validation_row = connection.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN validated IS NOT NULL THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN validated = 0 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(escalated), 0),
                    COALESCE(SUM(escalation_attempts), 0)
                FROM requests
                """
            ).fetchone()
            escalation_cost_rows = connection.execute(
                "SELECT escalation_cost_usd FROM requests WHERE escalation_cost_usd IS NOT NULL"
            )
            cost_rows = connection.execute(
                """
                SELECT baseline_cost_usd, actual_cost_usd
                FROM requests
                WHERE baseline_cost_usd IS NOT NULL OR actual_cost_usd IS NOT NULL
                """
            )
            baseline_cost = Decimal(0)
            actual_cost = Decimal(0)
            for baseline_value, actual_value in cost_rows:
                if baseline_value is not None:
                    baseline_cost += Decimal(baseline_value)
                if actual_value is not None:
                    actual_cost += Decimal(actual_value)
            escalation_cost = sum(
                (Decimal(value) for (value,) in escalation_cost_rows),
                Decimal(0),
            )
        assert row is not None
        return Stats(
            requests=int(row[0]),
            successful_requests=int(row[1]),
            failed_requests=int(row[2]),
            input_tokens=int(row[3]),
            output_tokens=int(row[4]),
            cache_hits=int(row[5]),
            baseline_cost_usd=baseline_cost,
            actual_cost_usd=actual_cost,
            priced_requests=int(row[8]),
            estimated_input_tokens_removed=int(row[9]),
            optimization_events=int(event_count_row[0]) if event_count_row is not None else 0,
            exact_cache_hits=int(row[6]),
            semantic_cache_hits=int(row[7]),
            routing_decisions=int(routing_row[0]) if routing_row is not None else 0,
            routed_model_changes=int(routing_row[1]) if routing_row is not None else 0,
            model_predictions=int(routing_row[2]) if routing_row is not None else 0,
            model_agreements=int(routing_row[3]) if routing_row is not None else 0,
            active_model_routes=int(routing_row[4]) if routing_row is not None else 0,
            validated_requests=int(validation_row[0]) if validation_row is not None else 0,
            validation_failures=int(validation_row[1]) if validation_row is not None else 0,
            escalated_requests=int(validation_row[2]) if validation_row is not None else 0,
            escalation_attempts=int(validation_row[3]) if validation_row is not None else 0,
            escalation_cost_usd=escalation_cost,
        )

    def validations_for_request(self, request_id: str) -> list[ValidationAttemptRecord]:
        if not self.enabled or not self.path.exists():
            return []
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT request_id, attempt, route, model, status_code, passed, reason,
                       input_tokens, output_tokens, cost_usd, latency_ms
                FROM validation_attempts
                WHERE request_id = ?
                ORDER BY attempt
                """,
                (request_id,),
            ).fetchall()
        return [
            ValidationAttemptRecord(
                request_id=str(row[0]),
                attempt=int(row[1]),
                route=str(row[2]) if row[2] is not None else None,
                model=str(row[3]) if row[3] is not None else None,
                status_code=int(row[4]),
                passed=bool(row[5]) if row[5] is not None else None,
                reason=str(row[6]),
                input_tokens=int(row[7]),
                output_tokens=int(row[8]),
                cost_usd=Decimal(row[9]) if row[9] is not None else None,
                latency_ms=float(row[10]),
            )
            for row in rows
        ]

    def events_for_request(self, request_id: str) -> list[OptimizationEvent]:
        if not self.enabled or not self.path.exists():
            return []
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT kind, path, before_estimated_tokens, after_estimated_tokens, detail
                FROM optimization_events
                WHERE request_id = ?
                ORDER BY sequence
                """,
                (request_id,),
            ).fetchall()
        return [OptimizationEvent(*row) for row in rows]

    def routing_for_request(self, request_id: str) -> RoutingDecisionRecord | None:
        if not self.enabled or not self.path.exists():
            return None
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_id, selected_route, executed_route, source, reason,
                       original_model, routed_model, model_configured, model_changed,
                       estimated_input_tokens, message_count, tool_count, has_media,
                       streaming, cache_only, rule_route, model_route, model_confidence,
                       model_mode, model_applied, model_fallback_reason, model_agreed
                FROM routing_decisions
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return RoutingDecisionRecord(
            request_id=str(row[0]),
            selected_route=str(row[1]),
            executed_route=str(row[2]) if row[2] is not None else None,
            source=str(row[3]),
            reason=str(row[4]),
            original_model=str(row[5]) if row[5] is not None else None,
            routed_model=str(row[6]) if row[6] is not None else None,
            model_configured=bool(row[7]),
            model_changed=bool(row[8]),
            estimated_input_tokens=int(row[9]),
            message_count=int(row[10]),
            tool_count=int(row[11]),
            has_media=bool(row[12]),
            streaming=bool(row[13]),
            cache_only=bool(row[14]),
            rule_route=str(row[15]) if row[15] is not None else None,
            model_route=str(row[16]) if row[16] is not None else None,
            model_confidence=float(row[17]) if row[17] is not None else None,
            model_mode=str(row[18]) if row[18] is not None else None,
            model_applied=bool(row[19]),
            model_fallback_reason=str(row[20]) if row[20] is not None else None,
            model_agreed=bool(row[21]) if row[21] is not None else None,
        )

    def check(self) -> None:
        if not self.enabled:
            return
        self.initialize()
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _expand_requests_schema(connection: sqlite3.Connection) -> None:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(requests)").fetchall()}
        additions = {
            "original_estimated_input_tokens": (
                "INTEGER NOT NULL DEFAULT 0 CHECK (original_estimated_input_tokens >= 0)"
            ),
            "optimized_estimated_input_tokens": (
                "INTEGER NOT NULL DEFAULT 0 CHECK (optimized_estimated_input_tokens >= 0)"
            ),
            "cache_type": "TEXT",
            "validated": "INTEGER CHECK (validated IN (0, 1))",
            "escalated": "INTEGER NOT NULL DEFAULT 0 CHECK (escalated IN (0, 1))",
            "escalation_attempts": ("INTEGER NOT NULL DEFAULT 0 CHECK (escalation_attempts >= 0)"),
            "escalation_cost_usd": "TEXT",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE requests ADD COLUMN {name} {definition}")

    @staticmethod
    def _expand_routing_schema(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(routing_decisions)").fetchall()
        }
        additions = {
            "rule_route": "TEXT",
            "model_route": "TEXT",
            "model_confidence": "REAL",
            "model_mode": "TEXT",
            "model_applied": "INTEGER NOT NULL DEFAULT 0 CHECK (model_applied IN (0, 1))",
            "model_fallback_reason": "TEXT",
            "model_agreed": "INTEGER CHECK (model_agreed IN (0, 1))",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(f"ALTER TABLE routing_decisions ADD COLUMN {name} {definition}")
