from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

import uvicorn
from pydantic import ValidationError

from optimizer.api import create_app
from optimizer.benchmark import load_benchmark, render_benchmark
from optimizer.cache import SQLiteExactCache, SQLiteSemanticCache
from optimizer.config import AppConfig, initialize_config, load_config, resolve_config_path
from optimizer.routing import LocalDecisionModel
from optimizer.security import bind_host_is_loopback
from optimizer.telemetry import Stats, TelemetryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimizer",
        description="Private, local-first LLM optimization gateway.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create the local configuration and database.")
    _add_config_argument(init_parser)
    init_parser.add_argument(
        "--force", action="store_true", help="Replace an existing config file."
    )

    serve_parser = subparsers.add_parser("serve", help="Run the localhost gateway.")
    _add_config_argument(serve_parser)
    serve_parser.add_argument("--host", help="Override the configured bind host.")
    serve_parser.add_argument("--port", type=int, help="Override the configured bind port.")

    doctor_parser = subparsers.add_parser(
        "doctor", help="Validate local configuration and storage."
    )
    _add_config_argument(doctor_parser)

    stats_parser = subparsers.add_parser("stats", help="Show local request and usage totals.")
    _add_config_argument(stats_parser)

    cache_parser = subparsers.add_parser("cache", help="Manage the local response caches.")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", required=True)
    cache_clear_parser = cache_subparsers.add_parser("clear", help="Delete local cache entries.")
    _add_config_argument(cache_clear_parser)
    cache_clear_parser.add_argument(
        "--kind",
        choices=("exact", "semantic", "all"),
        default="exact",
        help="Cache to clear (default: exact).",
    )
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Compare recorded direct and optimized benchmark metadata.",
    )
    benchmark_parser.add_argument("path", type=Path, help="JSONL benchmark result file.")
    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="Path to optimizer.yaml.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        exit_code = _dispatch(args)
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        sqlite3.Error,
        ValueError,
        ValidationError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    raise SystemExit(exit_code)


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "benchmark":
        print(render_benchmark(load_benchmark(args.path)))
        return 0
    if args.command == "init":
        path = initialize_config(args.config, force=args.force)
        config = load_config(path)
        store = TelemetryStore(
            config.telemetry.database_path,
            enabled=config.telemetry.enabled,
        )
        store.initialize()
        print(f"Configuration: {path}")
        if store.enabled:
            print(f"Telemetry:     {store.path}")
        print("Raw prompts and responses are not stored.")
        return 0

    config = load_config(args.config)
    if args.command == "serve":
        host = args.host or config.server.host
        port = args.port or config.server.port
        if not bind_host_is_loopback(host) and not config.security.remote_auth.enabled:
            raise ValueError("Refusing non-loopback bind without security.remote_auth.enabled.")
        uvicorn.run(create_app(config), host=host, port=port, log_config=None, proxy_headers=False)
        return 0
    if args.command == "doctor":
        return _doctor(config, resolve_config_path(args.config))
    if args.command == "stats":
        store = TelemetryStore(
            config.telemetry.database_path,
            enabled=config.telemetry.enabled,
        )
        print(format_stats(store.stats()))
        return 0
    if args.command == "cache" and args.cache_command == "clear":
        kind = getattr(args, "kind", "exact")
        if kind in {"exact", "all"}:
            cache = SQLiteExactCache(
                config.optimization.cache.exact.database_path,
                enabled=True,
            )
            cleared = cache.clear()
            print(f"Cleared {cleared:,} exact-cache entr{'y' if cleared == 1 else 'ies'}.")
        if kind in {"semantic", "all"}:
            semantic_cache = SQLiteSemanticCache(
                config.optimization.cache.semantic.database_path,
                enabled=True,
            )
            cleared = semantic_cache.clear()
            print(f"Cleared {cleared:,} semantic-cache entr{'y' if cleared == 1 else 'ies'}.")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def _doctor(config: AppConfig, config_path: Path) -> int:
    store = TelemetryStore(
        config.telemetry.database_path,
        enabled=config.telemetry.enabled,
    )
    store.check()
    print(f"[ok] Configuration: {config_path}")
    if store.enabled:
        print(f"[ok] Telemetry database: {store.path}")
    else:
        print("[ok] Telemetry disabled")
    if config.optimization.cache.exact.enabled:
        cache = SQLiteExactCache(config.optimization.cache.exact.database_path)
        cache.initialize()
        print(f"[ok] Exact cache database: {cache.path}")
    else:
        print("[ok] Exact cache disabled")
    semantic_config = config.optimization.cache.semantic
    if semantic_config.enabled:
        semantic_cache = SQLiteSemanticCache(semantic_config.database_path)
        semantic_cache.initialize()
        print(
            "[ok] Semantic cache database: "
            f"{semantic_cache.path} (threshold {semantic_config.similarity_threshold:.3f}, "
            f"TTL {semantic_config.ttl_seconds:,}s)"
        )
    else:
        print("[ok] Semantic cache disabled")
    tool_compression = config.optimization.compression.tool_output
    if tool_compression.enabled:
        print(
            "[ok] Tool-output compression: "
            f"{tool_compression.mode} (minimum {tool_compression.min_characters:,} characters)"
        )
    else:
        print("[ok] Tool-output compression disabled")
    context = config.optimization.input.context
    if context.enabled:
        print(
            "[ok] Context intelligence enabled: "
            f"budget {config.optimization.input.max_estimated_tokens:,} estimated tokens; "
            f"keep {context.min_recent_turns:,} recent turns; "
            f"maximum prunable score {context.max_prunable_score:.3f}"
        )
    else:
        print("[ok] Context intelligence disabled")
    routing = config.optimization.routing
    if routing.enabled:
        print(
            "[ok] Rule routing enabled: "
            f"cheap <= {routing.cheap_max_estimated_tokens:,} estimated tokens; "
            f"frontier >= {routing.frontier_min_estimated_tokens:,} tokens or "
            f"{routing.frontier_min_messages:,} messages"
        )
    else:
        print("[ok] Rule routing disabled")
    decision = routing.decision_model
    if decision.enabled and decision.artifact_path is not None:
        model = LocalDecisionModel.load(decision.artifact_path)
        evidence = model.has_activation_evidence(decision)
        active_blocked = decision.mode == "active" and not evidence
        if decision.mode == "shadow":
            state = "shadow only"
        else:
            state = "eligible" if evidence else "insufficient evidence; rule fallback"
        prefix = "[warn]" if active_blocked else "[ok]"
        print(
            f"{prefix} Decision model: {decision.mode} ({decision.artifact_path}; "
            f"validation n={model.validation_samples:,}, accuracy={model.validation_accuracy:.3f}; "
            f"{state})"
        )
    else:
        print("[ok] Decision model disabled")
    validation = config.optimization.validation
    if validation.enabled:
        escalation = (
            f"enabled, max {validation.escalation.max_attempts} attempts"
            if validation.escalation.enabled
            else "disabled"
        )
        print(f"[ok] Response validation enabled; escalation {escalation}")
    else:
        print("[ok] Response validation disabled")
    remote_auth = config.security.remote_auth
    if remote_auth.enabled:
        state = "set" if os.environ.get(remote_auth.api_key_env) else "missing"
        prefix = "[ok]" if state == "set" else "[warn]"
        print(f"{prefix} Remote authentication: {remote_auth.api_key_env} {state}")
    else:
        print("[ok] Remote authentication disabled (loopback only)")
    rate_limit = config.security.rate_limit
    if rate_limit.enabled:
        print(
            "[ok] Rate limit: "
            f"{rate_limit.requests_per_minute:,}/minute; burst {rate_limit.burst:,}"
        )
    else:
        print("[ok] Rate limiting disabled")
    for provider_name, provider in (
        ("OpenAI", config.providers.openai),
        ("Anthropic", config.providers.anthropic),
        ("OpenRouter", config.providers.openrouter),
        ("Ollama", config.providers.ollama),
    ):
        state = (
            "set" if os.environ.get(provider.api_key_env) else "not set; request headers allowed"
        )
        print(
            f"[ok] {provider_name} endpoint: {provider.base_url} ({provider.api_key_env}: {state})"
        )
    return 0


def format_stats(stats: Stats) -> str:
    lines = [
        f"Requests          {stats.requests:>12,}",
        f"Successful        {stats.successful_requests:>12,}",
        f"Failed            {stats.failed_requests:>12,}",
        "",
        f"Input tokens      {stats.input_tokens:>12,}",
        f"Output tokens     {stats.output_tokens:>12,}",
        f"Estimated removed {stats.estimated_input_tokens_removed:>12,}",
        f"Optimization events{stats.optimization_events:>11,}",
        "",
        f"Cache hits        {stats.cache_hits:>12,}",
        f"  Exact          {stats.exact_cache_hits:>12,}",
        f"  Semantic       {stats.semantic_cache_hits:>12,}",
        "",
        f"Routing decisions{stats.routing_decisions:>12,}",
        f"  Model changes {stats.routed_model_changes:>12,}",
        f"  ML predictions{stats.model_predictions:>12,}",
        f"  ML agreements {stats.model_agreements:>12,}",
        f"  ML active     {stats.active_model_routes:>12,}",
        "",
        f"Validated        {stats.validated_requests:>12,}",
        f"Validation failed{stats.validation_failures:>11,}",
        f"Escalated        {stats.escalated_requests:>12,}",
        f"Escalation calls {stats.escalation_attempts:>11,}",
        f"Escalation cost  {_money(stats.escalation_cost_usd):>12}",
        "",
        f"Baseline cost     {_money(stats.baseline_cost_usd):>12}",
        f"Actual cost       {_money(stats.actual_cost_usd):>12}",
        f"Saved             {_money(stats.saved_usd):>12}",
    ]
    if stats.requests and stats.priced_requests < stats.requests:
        unpriced_requests = stats.requests - stats.priced_requests
        lines.extend(
            [
                "",
                f"Note: {unpriced_requests:,} request(s) used unpriced models.",
            ]
        )
    return "\n".join(lines)


def _money(value: Decimal) -> str:
    return f"${value:.6f}"


if __name__ == "__main__":
    main()
