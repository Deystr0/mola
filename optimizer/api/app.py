from __future__ import annotations

import json
import logging
import sqlite3
import time
from asyncio import CancelledError, Semaphore, wait_for
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from optimizer import __version__
from optimizer.cache import (
    CacheControlError,
    ExactCacheBackend,
    SemanticCacheBackend,
    SemanticCacheControlError,
    SQLiteExactCache,
    SQLiteSemanticCache,
    make_cache_key,
    make_semantic_compatibility_key,
    make_semantic_entry_key,
    plan_exact_cache,
    plan_semantic_cache,
)
from optimizer.compression import (
    ToolCompressionModeError,
    ToolCompressionResult,
    ToolOutputCompressor,
)
from optimizer.config import AppConfig, ProvidersConfig
from optimizer.context import (
    InputOptimizer,
    OptimizationEvent,
    OptimizationResult,
    estimate_tokens,
)
from optimizer.cost import CostCalculator
from optimizer.embeddings import Embedding, EmbeddingBackend, HashingEmbeddingBackend
from optimizer.logging import configure_logging, log_event
from optimizer.output import OutputBudgetController, OutputControlResult, OutputPolicyError
from optimizer.providers import (
    AnthropicProvider,
    MissingCredentialError,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    Provider,
)
from optimizer.providers.base import response_headers
from optimizer.providers.usage import StreamingUsageObserver, Usage, parse_json_usage
from optimizer.routing import (
    DecisionModelArtifactError,
    DecisionRouter,
    LocalDecisionModel,
    RouteDecision,
    RouterControlError,
    RuleRouter,
)
from optimizer.security import AccessController, client_ip_for_rate_limit
from optimizer.telemetry import (
    RequestRecord,
    RoutingDecisionRecord,
    TelemetryStore,
    ValidationAttemptRecord,
)
from optimizer.validation import ResponseValidator, ValidationResult

logger = logging.getLogger("optimizer.gateway")


@dataclass(frozen=True, slots=True)
class _ValidatedResponse:
    response: httpx.Response
    content: bytes
    final_usage: Usage
    total_usage: Usage
    final_model: str | None
    final_route: str | None
    validation: ValidationResult
    attempts: tuple[ValidationAttemptRecord, ...]
    escalation_cost_usd: Decimal | None
    actual_cost_usd: Decimal | None
    optimization: OptimizationResult


def create_app(
    config: AppConfig,
    *,
    client: httpx.AsyncClient | None = None,
    telemetry: TelemetryStore | None = None,
    exact_cache: ExactCacheBackend | None = None,
    semantic_cache: SemanticCacheBackend | None = None,
    embedding_backend: EmbeddingBackend | None = None,
    router: DecisionRouter | None = None,
    providers: Mapping[str, Provider] | None = None,
) -> FastAPI:
    ProvidersConfig.model_validate(config.providers.model_dump())
    configure_logging(config.logging.level)
    owns_client = client is None
    http_client = client or httpx.AsyncClient()
    store = telemetry or TelemetryStore(
        config.telemetry.database_path,
        enabled=config.telemetry.enabled,
    )
    try:
        store.initialize()
    except (OSError, sqlite3.Error) as error:
        store.enabled = False
        log_event(
            logger,
            "telemetry_disabled",
            reason=type(error).__name__,
        )
    cache = exact_cache or SQLiteExactCache(
        config.optimization.cache.exact.database_path,
        enabled=config.optimization.cache.exact.enabled,
    )
    try:
        cache.initialize()
    except (OSError, sqlite3.Error) as error:
        cache.enabled = False
        log_event(
            logger,
            "exact_cache_disabled",
            reason=type(error).__name__,
        )
    semantic = semantic_cache or SQLiteSemanticCache(
        config.optimization.cache.semantic.database_path,
        enabled=config.optimization.cache.semantic.enabled,
    )
    try:
        semantic.initialize()
    except (OSError, sqlite3.Error) as error:
        semantic.enabled = False
        log_event(
            logger,
            "semantic_cache_disabled",
            reason=type(error).__name__,
        )
    embeddings = embedding_backend or HashingEmbeddingBackend(
        config.optimization.cache.semantic.embedding_dimensions
    )
    calculator = CostCalculator(config.pricing)
    input_optimizer = InputOptimizer(config.optimization.input)
    output_controller = OutputBudgetController(config.optimization.output)
    tool_output_compressor = ToolOutputCompressor(config.optimization.compression.tool_output)
    response_validator = ResponseValidator(config.optimization.validation)
    access_controller = AccessController(config.security)
    concurrency = Semaphore(config.security.limits.max_concurrent_requests)
    decision_model = None
    decision_config = config.optimization.routing.decision_model
    if decision_config.enabled and decision_config.artifact_path is not None:
        try:
            decision_model = LocalDecisionModel.load(decision_config.artifact_path)
        except DecisionModelArtifactError as error:
            log_event(
                logger,
                "decision_model_disabled",
                reason=type(error).__name__,
            )
    routing_engine = router or DecisionRouter(
        RuleRouter(config.optimization.routing),
        decision_config,
        decision_model,
    )
    provider_map: Mapping[str, Provider] = providers or {
        "openai": OpenAIProvider(config.providers.openai, http_client),
        "anthropic": AnthropicProvider(config.providers.anthropic, http_client),
        "openrouter": OpenRouterProvider(config.providers.openrouter, http_client),
        "ollama": OllamaProvider(config.providers.ollama, http_client),
    }

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_client:
            await http_client.aclose()

    app = FastAPI(
        title="Local LLM Optimizer",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.telemetry = store
    app.state.exact_cache = cache
    app.state.semantic_cache = semantic
    app.state.router = routing_engine
    app.state.access_controller = access_controller

    @app.middleware("http")
    async def production_boundary(request: Request, call_next):
        health_path = request.url.path in {"/health", "/health/live", "/health/ready"}
        health_bypass = health_path and config.security.remote_auth.allow_health_unauthenticated
        if not health_bypass:
            decision = access_controller.check(
                supplied_key=request.headers.get("x-optimizer-api-key"),
                client_host=client_ip_for_rate_limit(
                    request.client.host if request.client is not None else None,
                    request.headers.get("x-optimizer-client-ip"),
                    config.security.remote_auth,
                ),
            )
            if not decision.allowed:
                headers = {"cache-control": "no-store"}
                if decision.status_code == 401:
                    headers["www-authenticate"] = 'ApiKey realm="optimizer"'
                if decision.retry_after_seconds is not None:
                    headers["retry-after"] = str(decision.retry_after_seconds)
                return JSONResponse(
                    status_code=decision.status_code,
                    content={"error": {"type": decision.reason, "message": "Request denied."}},
                    headers=headers,
                )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = 0
            if declared_length > config.security.limits.max_request_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "type": "request_too_large",
                            "message": "Request body exceeds the configured limit.",
                        }
                    },
                )
        try:
            await wait_for(
                concurrency.acquire(),
                timeout=config.security.limits.concurrency_wait_seconds,
            )
        except TimeoutError:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "type": "concurrency_limit",
                        "message": "Gateway concurrency limit reached.",
                    }
                },
                headers={"retry-after": "1"},
            )
        try:
            response = await call_next(request)
        except BaseException:
            concurrency.release()
            raise
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["referrer-policy"] = "no-referrer"
        response.headers["cache-control"] = response.headers.get("cache-control", "no-store")
        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is None:
            concurrency.release()
        else:

            async def release_after_body():
                try:
                    async for chunk in body_iterator:
                        yield chunk
                finally:
                    concurrency.release()

            response.body_iterator = release_after_body()
        return response

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/health/live")
    async def liveness() -> dict[str, str]:
        return {"status": "alive", "version": __version__}

    @app.get("/health/ready")
    async def readiness() -> JSONResponse:
        try:
            store.check()
        except (OSError, sqlite3.Error):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(content={"status": "ready", "version": __version__})

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        return await _proxy_request(
            request=request,
            provider=provider_map["openai"],
            store=store,
            calculator=calculator,
            input_optimizer=input_optimizer,
            output_controller=output_controller,
            tool_output_compressor=tool_output_compressor,
            router=routing_engine,
            exact_cache=cache,
            cache_ttl_seconds=config.optimization.cache.exact.ttl_seconds,
            semantic_cache=semantic,
            embedding_backend=embeddings,
            semantic_ttl_seconds=config.optimization.cache.semantic.ttl_seconds,
            semantic_similarity_threshold=(config.optimization.cache.semantic.similarity_threshold),
            semantic_max_candidates=config.optimization.cache.semantic.max_candidates,
            response_validator=response_validator,
            max_request_bytes=config.security.limits.max_request_bytes,
        )

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        return await _proxy_request(
            request=request,
            provider=provider_map["anthropic"],
            store=store,
            calculator=calculator,
            input_optimizer=input_optimizer,
            output_controller=output_controller,
            tool_output_compressor=tool_output_compressor,
            router=routing_engine,
            exact_cache=cache,
            cache_ttl_seconds=config.optimization.cache.exact.ttl_seconds,
            semantic_cache=semantic,
            embedding_backend=embeddings,
            semantic_ttl_seconds=config.optimization.cache.semantic.ttl_seconds,
            semantic_similarity_threshold=(config.optimization.cache.semantic.similarity_threshold),
            semantic_max_candidates=config.optimization.cache.semantic.max_candidates,
            response_validator=response_validator,
            max_request_bytes=config.security.limits.max_request_bytes,
        )

    @app.post("/openrouter/v1/chat/completions")
    async def openrouter_chat_completions(request: Request) -> Response:
        return await _proxy_request(
            request=request,
            provider=provider_map["openrouter"],
            store=store,
            calculator=calculator,
            input_optimizer=input_optimizer,
            output_controller=output_controller,
            tool_output_compressor=tool_output_compressor,
            router=routing_engine,
            exact_cache=cache,
            cache_ttl_seconds=config.optimization.cache.exact.ttl_seconds,
            semantic_cache=semantic,
            embedding_backend=embeddings,
            semantic_ttl_seconds=config.optimization.cache.semantic.ttl_seconds,
            semantic_similarity_threshold=config.optimization.cache.semantic.similarity_threshold,
            semantic_max_candidates=config.optimization.cache.semantic.max_candidates,
            response_validator=response_validator,
            max_request_bytes=config.security.limits.max_request_bytes,
        )

    @app.post("/ollama/v1/chat/completions")
    async def ollama_chat_completions(request: Request) -> Response:
        return await _proxy_request(
            request=request,
            provider=provider_map["ollama"],
            store=store,
            calculator=calculator,
            input_optimizer=input_optimizer,
            output_controller=output_controller,
            tool_output_compressor=tool_output_compressor,
            router=routing_engine,
            exact_cache=cache,
            cache_ttl_seconds=config.optimization.cache.exact.ttl_seconds,
            semantic_cache=semantic,
            embedding_backend=embeddings,
            semantic_ttl_seconds=config.optimization.cache.semantic.ttl_seconds,
            semantic_similarity_threshold=config.optimization.cache.semantic.similarity_threshold,
            semantic_max_candidates=config.optimization.cache.semantic.max_candidates,
            response_validator=response_validator,
            max_request_bytes=config.security.limits.max_request_bytes,
        )

    return app


async def _proxy_request(
    *,
    request: Request,
    provider: Provider,
    store: TelemetryStore,
    calculator: CostCalculator,
    input_optimizer: InputOptimizer,
    output_controller: OutputBudgetController,
    tool_output_compressor: ToolOutputCompressor,
    router: DecisionRouter,
    exact_cache: ExactCacheBackend,
    cache_ttl_seconds: int,
    semantic_cache: SemanticCacheBackend,
    embedding_backend: EmbeddingBackend,
    semantic_ttl_seconds: int,
    semantic_similarity_threshold: float,
    semantic_max_candidates: int,
    response_validator: ResponseValidator,
    max_request_bytes: int,
) -> Response:
    request_id = str(uuid4())
    started = time.perf_counter()
    body_buffer = bytearray()
    async for chunk in request.stream():
        if len(chunk) > max_request_bytes - len(body_buffer):
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "type": "request_too_large",
                        "message": "Request body exceeds the configured limit.",
                    }
                },
                headers={"x-optimizer-request-id": request_id},
            )
        body_buffer.extend(chunk)
    body = bytes(body_buffer)
    metadata = _request_metadata(body)
    model = metadata.get("model") if isinstance(metadata.get("model"), str) else None
    streaming = metadata.get("stream") is True
    input_optimization = _optimize_or_bypass(input_optimizer, body, request_id)
    try:
        compression = _compress_tool_output_or_bypass(
            compressor=tool_output_compressor,
            body=input_optimization.body,
            requested_mode=request.headers.get("x-optimizer-tool-compression"),
            request_id=request_id,
        )
    except ToolCompressionModeError as error:
        _record(
            store=store,
            calculator=calculator,
            request_id=request_id,
            provider=provider.name,
            model=model,
            status="rejected",
            status_code=400,
            started=started,
            usage=Usage(),
            optimization=input_optimization,
        )
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "invalid_tool_compression_mode", "message": str(error)}},
            headers={"x-optimizer-request-id": request_id},
        )
    compressed_optimization = _combine_input_and_compression(
        input_optimization,
        compression,
    )
    try:
        routing = router.route(
            provider=provider.name,
            body=compression.body,
            requested_route=request.headers.get("x-optimizer-route"),
        )
    except RouterControlError as error:
        _record(
            store=store,
            calculator=calculator,
            request_id=request_id,
            provider=provider.name,
            model=model,
            status="rejected",
            status_code=400,
            started=started,
            usage=Usage(),
            optimization=compressed_optimization,
        )
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "invalid_route", "message": str(error)}},
            headers={"x-optimizer-request-id": request_id},
        )
    routed_optimization = _combine_input_and_routing(compressed_optimization, routing)
    model = routing.routed_model
    try:
        output_control = _control_output_or_bypass(
            controller=output_controller,
            provider=provider.name,
            body=routing.body,
            requested_policy=request.headers.get("x-optimizer-output-policy"),
            request_id=request_id,
        )
    except OutputPolicyError as error:
        _record(
            store=store,
            calculator=calculator,
            request_id=request_id,
            provider=provider.name,
            model=model,
            status="rejected",
            status_code=400,
            started=started,
            usage=Usage(),
            optimization=routed_optimization,
            routing=routing,
        )
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "invalid_output_policy", "message": str(error)}},
            headers={"x-optimizer-request-id": request_id},
        )
    optimization = _combine_optimization(
        routed_optimization,
        output_control,
    )
    incoming_headers = httpx.Headers(request.headers)
    cache_control = request.headers.get("x-optimizer-cache-control")
    try:
        exact_plan = plan_exact_cache(
            enabled=exact_cache.enabled,
            body=optimization.body,
            streaming=streaming,
            cache_control=cache_control,
        )
        semantic_plan = plan_semantic_cache(
            enabled=semantic_cache.enabled,
            body=optimization.body,
            streaming=streaming,
            cache_control=cache_control,
            freshness=request.headers.get("x-optimizer-semantic-freshness"),
            task=request.headers.get("x-optimizer-semantic-task"),
        )
    except CacheControlError as error:
        _record(
            store=store,
            calculator=calculator,
            request_id=request_id,
            provider=provider.name,
            model=model,
            status="rejected",
            status_code=400,
            started=started,
            usage=Usage(),
            optimization=optimization,
            routing=routing,
        )
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "invalid_cache_control", "message": str(error)}},
            headers={"x-optimizer-request-id": request_id},
        )
    except SemanticCacheControlError as error:
        _record(
            store=store,
            calculator=calculator,
            request_id=request_id,
            provider=provider.name,
            model=model,
            status="rejected",
            status_code=400,
            started=started,
            usage=Usage(),
            optimization=optimization,
            routing=routing,
        )
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "invalid_semantic_cache_control", "message": str(error)}},
            headers={"x-optimizer-request-id": request_id},
        )

    if exact_plan.event is not None:
        optimization = _append_events(optimization, exact_plan.event)
    if semantic_plan.event is not None:
        optimization = _append_events(optimization, semantic_plan.event)

    exact_read = exact_plan.read
    exact_write = exact_plan.write
    semantic_read = semantic_plan.read
    semantic_write = semantic_plan.write
    freshness_bypass = semantic_plan.reason == "freshness_sensitive"
    if freshness_bypass:
        exact_read = False
        exact_write = False

    cache_key: str | None = None
    semantic_compatibility_key: str | None = None
    semantic_entry_key: str | None = None
    semantic_embedding: Embedding = {}
    cache_status = _initial_cache_status(
        exact_mode=exact_plan.mode,
        semantic_mode=semantic_plan.mode,
        freshness_bypass=freshness_bypass,
    )
    if exact_read or exact_write or semantic_read or semantic_write:
        try:
            identity = provider.cache_identity(incoming_headers)
        except MissingCredentialError as error:
            _record(
                store=store,
                calculator=calculator,
                request_id=request_id,
                provider=provider.name,
                model=model,
                status="rejected",
                status_code=401,
                started=started,
                usage=Usage(),
                optimization=optimization,
                routing=routing,
            )
            return JSONResponse(
                status_code=401,
                content={"error": {"type": "missing_provider_credential", "message": str(error)}},
                headers={"x-optimizer-request-id": request_id},
            )
        if exact_read or exact_write:
            cache_key = make_cache_key(
                namespace=provider.cache_namespace,
                identity=identity,
                body=optimization.body,
            )
        if (semantic_read or semantic_write) and semantic_plan.request is not None:
            try:
                semantic_embedding = embedding_backend.embed(semantic_plan.request.query)
                semantic_compatibility_key = make_semantic_compatibility_key(
                    namespace=provider.cache_namespace,
                    identity=identity,
                    task=semantic_plan.request.task,
                    compatibility_body=semantic_plan.request.compatibility_body,
                    embedding_version=embedding_backend.version,
                )
                semantic_entry_key = make_semantic_entry_key(
                    compatibility_key=semantic_compatibility_key,
                    query=semantic_plan.request.query,
                )
            except Exception as error:
                semantic_read = False
                semantic_write = False
                cache_status = "ERROR"
                optimization = _append_events(
                    optimization,
                    _cache_event(
                        "semantic_cache_error",
                        f"operation=embed;reason={type(error).__name__}",
                    ),
                )
                log_event(
                    logger,
                    "semantic_cache_error",
                    request_id=request_id,
                    operation="embed",
                    reason=type(error).__name__,
                )

    if exact_read and cache_key is not None:
        try:
            cached = exact_cache.get(cache_key)
        except (OSError, sqlite3.Error) as error:
            exact_cache.enabled = False
            exact_write = False
            cache_status = "ERROR"
            optimization = _append_events(
                optimization,
                _cache_event("exact_cache_error", f"operation=get;reason={type(error).__name__}"),
            )
            log_event(
                logger,
                "exact_cache_error",
                request_id=request_id,
                operation="get",
                reason=type(error).__name__,
            )
        else:
            if cached is not None:
                cache_validation = response_validator.validate(
                    provider=provider.name,
                    request_body=optimization.body,
                    status_code=cached.status_code,
                    content_type=cached.content_type,
                    response_body=cached.body,
                    streaming=False,
                )
                if cache_validation.passed is False:
                    exact_cache.delete(cache_key)
                    cached = None
                    cache_status = "MISS"
                    optimization = _append_events(
                        optimization,
                        _cache_event(
                            "exact_cache_invalidated",
                            f"reason={cache_validation.reason}",
                        ),
                    )
            if cached is not None:
                if cache_validation.enabled:
                    optimization = _append_events(
                        optimization,
                        _validation_event(cache_validation, 1, "cache"),
                    )
                optimization = _append_events(
                    optimization,
                    _cache_event("exact_cache_hit", "result=hit"),
                )
                headers = _optimizer_headers(
                    {"content-type": cached.content_type},
                    request_id=request_id,
                    optimization=optimization,
                    output_control=output_control,
                    compression=compression,
                    cache_status="HIT",
                    routing=routing,
                    executed_route="cache",
                    validation=cache_validation,
                )
                _record(
                    store=store,
                    calculator=calculator,
                    request_id=request_id,
                    provider=provider.name,
                    model=model,
                    status="success",
                    status_code=cached.status_code,
                    started=started,
                    usage=Usage(),
                    baseline_usage=Usage(cached.input_tokens, cached.output_tokens),
                    cache_hit=True,
                    cache_type="exact",
                    optimization=optimization,
                    routing=routing,
                    executed_route="cache",
                    validation=cache_validation,
                )
                return Response(
                    content=cached.body,
                    status_code=cached.status_code,
                    headers=headers,
                )
            cache_status = "MISS"
            optimization = _append_events(
                optimization,
                _cache_event("exact_cache_miss", "result=miss"),
            )

    if semantic_read and semantic_compatibility_key is not None:
        try:
            semantic_hit = semantic_cache.search(
                compatibility_key=semantic_compatibility_key,
                embedding_version=embedding_backend.version,
                embedding=semantic_embedding,
                threshold=semantic_similarity_threshold,
                limit=semantic_max_candidates,
            )
        except Exception as error:
            semantic_cache.enabled = False
            semantic_write = False
            cache_status = "ERROR"
            optimization = _append_events(
                optimization,
                _cache_event(
                    "semantic_cache_error",
                    f"operation=search;reason={type(error).__name__}",
                ),
            )
            log_event(
                logger,
                "semantic_cache_error",
                request_id=request_id,
                operation="search",
                reason=type(error).__name__,
            )
        else:
            if semantic_hit is not None:
                cached = semantic_hit.entry
                cache_validation = response_validator.validate(
                    provider=provider.name,
                    request_body=optimization.body,
                    status_code=cached.status_code,
                    content_type=cached.content_type,
                    response_body=cached.body,
                    streaming=False,
                )
                if cache_validation.passed is False:
                    semantic_hit = None
                    cache_status = "MISS"
                    optimization = _append_events(
                        optimization,
                        _cache_event(
                            "semantic_cache_rejected",
                            f"reason={cache_validation.reason}",
                        ),
                    )
            if semantic_hit is not None:
                cached = semantic_hit.entry
                if cache_validation.enabled:
                    optimization = _append_events(
                        optimization,
                        _validation_event(cache_validation, 1, "cache"),
                    )
                optimization = _append_events(
                    optimization,
                    _cache_event(
                        "semantic_cache_hit",
                        f"result=hit;similarity={semantic_hit.similarity:.6f}",
                    ),
                )
                headers = _optimizer_headers(
                    {"content-type": cached.content_type},
                    request_id=request_id,
                    optimization=optimization,
                    output_control=output_control,
                    compression=compression,
                    cache_status="SEMANTIC_HIT",
                    semantic_similarity=semantic_hit.similarity,
                    routing=routing,
                    executed_route="cache",
                    validation=cache_validation,
                )
                _record(
                    store=store,
                    calculator=calculator,
                    request_id=request_id,
                    provider=provider.name,
                    model=model,
                    status="success",
                    status_code=cached.status_code,
                    started=started,
                    usage=Usage(),
                    baseline_usage=Usage(cached.input_tokens, cached.output_tokens),
                    cache_hit=True,
                    cache_type="semantic",
                    optimization=optimization,
                    routing=routing,
                    executed_route="cache",
                    validation=cache_validation,
                )
                return Response(
                    content=cached.body,
                    status_code=cached.status_code,
                    headers=headers,
                )
            cache_status = "MISS"
            optimization = _append_events(
                optimization,
                _cache_event("semantic_cache_miss", "result=miss"),
            )

    if routing.cache_only:
        unavailable = cache_status == "ERROR"
        status_code = 503 if unavailable else 409
        error_type = "cache_unavailable" if unavailable else "cache_miss"
        message = (
            "Cache lookup failed; the cache-only route did not call a provider."
            if unavailable
            else "No reusable cache entry exists; the cache-only route did not call a provider."
        )
        optimization = _append_events(
            optimization,
            _cache_event("cache_only_not_served", f"reason={error_type}"),
        )
        headers = _optimizer_headers(
            {},
            request_id=request_id,
            optimization=optimization,
            output_control=output_control,
            compression=compression,
            cache_status=cache_status,
            routing=routing,
        )
        _record(
            store=store,
            calculator=calculator,
            request_id=request_id,
            provider=provider.name,
            model=model,
            status="cache_miss",
            status_code=status_code,
            started=started,
            usage=Usage(),
            optimization=optimization,
            routing=routing,
        )
        return JSONResponse(
            status_code=status_code,
            content={"error": {"type": error_type, "message": message}},
            headers=headers,
        )

    executed_provider_route = _provider_executed_route(routing)

    try:
        upstream = await provider.send(
            body=optimization.body,
            incoming_headers=incoming_headers,
            streaming=streaming,
        )
    except MissingCredentialError as error:
        _record(
            store=store,
            calculator=calculator,
            request_id=request_id,
            provider=provider.name,
            model=model,
            status="rejected",
            status_code=401,
            started=started,
            usage=Usage(),
            optimization=optimization,
            routing=routing,
        )
        return JSONResponse(
            status_code=401,
            content={"error": {"type": "missing_provider_credential", "message": str(error)}},
            headers={"x-optimizer-request-id": request_id},
        )
    except httpx.TimeoutException:
        _record(
            store=store,
            calculator=calculator,
            request_id=request_id,
            provider=provider.name,
            model=model,
            status="timeout",
            status_code=504,
            started=started,
            usage=Usage(),
            optimization=optimization,
            routing=routing,
            executed_route=executed_provider_route,
        )
        return JSONResponse(
            status_code=504,
            content={"error": {"type": "upstream_timeout", "message": "Provider timed out."}},
            headers={"x-optimizer-request-id": request_id},
        )
    except httpx.RequestError:
        _record(
            store=store,
            calculator=calculator,
            request_id=request_id,
            provider=provider.name,
            model=model,
            status="upstream_error",
            status_code=502,
            started=started,
            usage=Usage(),
            optimization=optimization,
            routing=routing,
            executed_route=executed_provider_route,
        )
        return JSONResponse(
            status_code=502,
            content={
                "error": {"type": "upstream_connection_error", "message": "Provider unavailable."}
            },
            headers={"x-optimizer-request-id": request_id},
        )

    if streaming:
        observer = StreamingUsageObserver(provider.name)
        stream_validation = response_validator.validate(
            provider=provider.name,
            request_body=optimization.body,
            status_code=upstream.response.status_code,
            content_type=upstream.response.headers.get("content-type", "text/event-stream"),
            response_body=b"",
            streaming=True,
        )
        if stream_validation.enabled:
            optimization = _append_events(
                optimization,
                _validation_event(stream_validation, 1, executed_provider_route),
            )
        headers = _optimizer_headers(
            response_headers(upstream.response),
            request_id=request_id,
            optimization=optimization,
            output_control=output_control,
            compression=compression,
            cache_status=cache_status,
            routing=routing,
            executed_route=executed_provider_route,
            validation=stream_validation,
        )
        return StreamingResponse(
            _stream_body(
                response=upstream.response,
                observer=observer,
                store=store,
                calculator=calculator,
                request_id=request_id,
                provider=provider.name,
                model=model,
                started=started,
                optimization=optimization,
                routing=routing,
                executed_route=executed_provider_route,
                validation=stream_validation,
            ),
            status_code=upstream.response.status_code,
            headers=headers,
        )

    validated = await _validate_and_escalate(
        initial=upstream.response,
        provider=provider,
        incoming_headers=incoming_headers,
        request_body=optimization.body,
        routing=routing,
        router=router,
        validator=response_validator,
        calculator=calculator,
        max_attempts=response_validator.config.escalation.max_attempts,
        escalation_enabled=response_validator.config.escalation.enabled,
        request_id=request_id,
        optimization=optimization,
    )
    content = validated.content
    usage = validated.total_usage
    model = validated.final_model
    executed_provider_route = validated.final_route
    optimization = validated.optimization
    status = (
        "validation_failed"
        if validated.validation.passed is False and 200 <= validated.response.status_code < 300
        else ("success" if 200 <= validated.response.status_code < 300 else "provider_error")
    )
    content_type = validated.response.headers.get("content-type", "application/json")
    cacheable = validated.validation.passed is not False and _cacheable_response(
        validated.response.status_code, content_type, content
    )
    if exact_write and cache_key is not None:
        if cacheable:
            try:
                exact_cache.set(
                    cache_key,
                    provider=provider.name,
                    model=model,
                    body=content,
                    status_code=validated.response.status_code,
                    content_type=content_type,
                    input_tokens=validated.final_usage.input_tokens,
                    output_tokens=validated.final_usage.output_tokens,
                    ttl_seconds=cache_ttl_seconds,
                )
            except (OSError, sqlite3.Error) as error:
                exact_cache.enabled = False
                cache_status = "ERROR"
                optimization = _append_events(
                    optimization,
                    _cache_event(
                        "exact_cache_error",
                        f"operation=set;reason={type(error).__name__}",
                    ),
                )
                log_event(
                    logger,
                    "exact_cache_error",
                    request_id=request_id,
                    operation="set",
                    reason=type(error).__name__,
                )
            else:
                optimization = _append_events(
                    optimization,
                    _cache_event("exact_cache_stored", f"ttl_seconds={cache_ttl_seconds}"),
                )
        else:
            optimization = _append_events(
                optimization,
                _cache_event("exact_cache_not_stored", "reason=response_not_cacheable"),
            )

    if semantic_write and semantic_compatibility_key is not None and semantic_entry_key is not None:
        if cacheable:
            try:
                semantic_cache.set(
                    semantic_entry_key,
                    compatibility_key=semantic_compatibility_key,
                    embedding_version=embedding_backend.version,
                    embedding=semantic_embedding,
                    provider=provider.name,
                    model=model,
                    body=content,
                    status_code=validated.response.status_code,
                    content_type=content_type,
                    input_tokens=validated.final_usage.input_tokens,
                    output_tokens=validated.final_usage.output_tokens,
                    ttl_seconds=semantic_ttl_seconds,
                )
            except Exception as error:
                semantic_cache.enabled = False
                cache_status = "ERROR"
                optimization = _append_events(
                    optimization,
                    _cache_event(
                        "semantic_cache_error",
                        f"operation=set;reason={type(error).__name__}",
                    ),
                )
                log_event(
                    logger,
                    "semantic_cache_error",
                    request_id=request_id,
                    operation="set",
                    reason=type(error).__name__,
                )
            else:
                optimization = _append_events(
                    optimization,
                    _cache_event(
                        "semantic_cache_stored",
                        f"ttl_seconds={semantic_ttl_seconds}",
                    ),
                )
        else:
            optimization = _append_events(
                optimization,
                _cache_event("semantic_cache_not_stored", "reason=response_not_cacheable"),
            )

    headers = _optimizer_headers(
        response_headers(validated.response),
        request_id=request_id,
        optimization=optimization,
        output_control=output_control,
        compression=compression,
        cache_status=cache_status,
        routing=routing,
        executed_route=executed_provider_route,
        validation=validated.validation,
        escalation_attempts=max(0, len(validated.attempts) - 1),
        escalation_cost_usd=validated.escalation_cost_usd,
    )
    _record(
        store=store,
        calculator=calculator,
        request_id=request_id,
        provider=provider.name,
        model=model,
        status=status,
        status_code=validated.response.status_code,
        started=started,
        usage=usage,
        baseline_usage=validated.final_usage,
        optimization=optimization,
        routing=routing,
        executed_route=executed_provider_route,
        validation=validated.validation,
        validation_attempts=validated.attempts,
        escalation_cost_usd=validated.escalation_cost_usd,
        actual_cost_usd=validated.actual_cost_usd,
    )
    return Response(content=content, status_code=validated.response.status_code, headers=headers)


async def _validate_and_escalate(
    *,
    initial: httpx.Response,
    provider: Provider,
    incoming_headers: httpx.Headers,
    request_body: bytes,
    routing: RouteDecision,
    router: DecisionRouter,
    validator: ResponseValidator,
    calculator: CostCalculator,
    max_attempts: int,
    escalation_enabled: bool,
    request_id: str,
    optimization: OptimizationResult,
) -> _ValidatedResponse:
    if not validator.config.enabled:
        content = await initial.aread()
        await initial.aclose()
        usage = _response_usage(provider.name, content)
        return _ValidatedResponse(
            response=initial,
            content=content,
            final_usage=usage,
            total_usage=usage,
            final_model=routing.routed_model,
            final_route=_provider_executed_route(routing),
            validation=ValidationResult(False, None, "disabled"),
            attempts=(),
            escalation_cost_usd=None,
            actual_cost_usd=None,
            optimization=optimization,
        )
    response = initial
    current_body = request_body
    current_model = routing.routed_model
    current_route = _provider_executed_route(routing)
    route_sequence = _escalation_routes(current_route) if escalation_enabled else ()
    attempts: list[ValidationAttemptRecord] = []
    costs: list[Decimal | None] = []
    total_usage = Usage()
    result = ValidationResult(validator.config.enabled, None, "not_evaluated")
    content = b""
    final_usage = Usage()

    for attempt_number in range(1, max_attempts + 1):
        attempt_started = time.perf_counter()
        content = await response.aread()
        await response.aclose()
        final_usage = _response_usage(provider.name, content)
        total_usage.input_tokens += final_usage.input_tokens
        total_usage.output_tokens += final_usage.output_tokens
        result = validator.validate(
            provider=provider.name,
            request_body=current_body,
            status_code=response.status_code,
            content_type=response.headers.get("content-type", "application/json"),
            response_body=content,
            streaming=False,
        )
        cost = calculator.calculate(
            current_model,
            final_usage.input_tokens,
            final_usage.output_tokens,
        ).actual_usd
        costs.append(cost)
        try:
            attempt_latency_ms = response.elapsed.total_seconds() * 1_000
        except RuntimeError:
            attempt_latency_ms = (time.perf_counter() - attempt_started) * 1_000
        attempts.append(
            ValidationAttemptRecord(
                request_id=request_id,
                attempt=attempt_number,
                route=current_route,
                model=current_model,
                status_code=response.status_code,
                passed=result.passed,
                reason=result.reason,
                input_tokens=final_usage.input_tokens,
                output_tokens=final_usage.output_tokens,
                cost_usd=cost,
                latency_ms=attempt_latency_ms,
            )
        )
        if result.enabled:
            optimization = _append_events(
                optimization,
                _validation_event(result, attempt_number, current_route),
            )
        if result.passed is not False or not 200 <= response.status_code < 300:
            break
        if len(attempts) >= max_attempts or not route_sequence:
            break

        next_response: httpx.Response | None = None
        while route_sequence and next_response is None:
            next_route = route_sequence[0]
            route_sequence = route_sequence[1:]
            next_decision = router.route(
                provider=provider.name,
                body=current_body,
                requested_route=next_route,
            )
            if not next_decision.model_configured or not next_decision.model_changed:
                continue
            try:
                upstream = await provider.send(
                    body=next_decision.body,
                    incoming_headers=incoming_headers,
                    streaming=False,
                )
            except (MissingCredentialError, httpx.RequestError):
                optimization = _append_events(
                    optimization,
                    _cache_event(
                        "escalation_stopped",
                        f"from={current_route};to={next_route};reason=transport_error",
                    ),
                )
                route_sequence = ()
                break
            optimization = _append_events(
                optimization,
                _cache_event(
                    "route_escalated",
                    f"from={current_route};to={next_route};reason={result.reason}",
                ),
            )
            current_body = next_decision.body
            current_model = next_decision.routed_model
            current_route = next_route
            next_response = upstream.response
        if next_response is None:
            break
        response = next_response

    total_cost = _sum_priced_costs(costs)
    escalation_cost = _sum_priced_costs(costs[:-1]) if len(costs) > 1 else Decimal(0)
    return _ValidatedResponse(
        response=response,
        content=content,
        final_usage=final_usage,
        total_usage=total_usage,
        final_model=current_model,
        final_route=current_route,
        validation=result,
        attempts=tuple(attempts),
        escalation_cost_usd=escalation_cost,
        actual_cost_usd=total_cost,
        optimization=optimization,
    )


def _escalation_routes(route: str | None) -> tuple[str, ...]:
    if route == "cheap":
        return ("mid", "frontier")
    if route == "mid":
        return ("frontier",)
    return ()


def _sum_priced_costs(costs: list[Decimal | None]) -> Decimal | None:
    if any(cost is None for cost in costs):
        return None
    return sum((cost for cost in costs if cost is not None), Decimal(0))


def _validation_event(
    result: ValidationResult,
    attempt: int,
    route: str | None,
) -> OptimizationEvent:
    kind = "response_validated" if result.passed else "response_validation_failed"
    if result.passed is None:
        kind = "response_validation_bypassed"
    return OptimizationEvent(
        kind=kind,
        path="$response",
        before_estimated_tokens=0,
        after_estimated_tokens=0,
        detail=f"attempt={attempt};route={route or 'unrouted'};reason={result.reason}",
    )


async def _stream_body(
    *,
    response: httpx.Response,
    observer: StreamingUsageObserver,
    store: TelemetryStore,
    calculator: CostCalculator,
    request_id: str,
    provider: str,
    model: str | None,
    started: float,
    optimization: OptimizationResult,
    routing: RouteDecision,
    executed_route: str | None,
    validation: ValidationResult | None = None,
) -> AsyncIterator[bytes]:
    status = "success" if 200 <= response.status_code < 300 else "provider_error"
    stream_status = status
    try:
        async for chunk in response.aiter_raw():
            observer.observe(chunk)
            yield chunk
    except (httpx.StreamError, httpx.TimeoutException):
        stream_status = "stream_error"
        raise
    except CancelledError:
        stream_status = "cancelled"
        raise
    finally:
        await response.aclose()
        _record(
            store=store,
            calculator=calculator,
            request_id=request_id,
            provider=provider,
            model=model,
            status=stream_status,
            status_code=response.status_code,
            started=started,
            usage=observer.finish(),
            optimization=optimization,
            routing=routing,
            executed_route=executed_route,
            validation=validation,
        )


def _request_metadata(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _response_usage(provider: str, content: bytes) -> Usage:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return Usage()
    return parse_json_usage(provider, payload)


def _optimize_or_bypass(
    optimizer: InputOptimizer,
    body: bytes,
    request_id: str,
) -> OptimizationResult:
    try:
        return optimizer.optimize(body)
    except Exception as error:
        log_event(
            logger,
            "input_optimization_bypassed",
            request_id=request_id,
            reason=type(error).__name__,
        )
        estimated_tokens = estimate_tokens(body)
        return OptimizationResult(
            body=body,
            events=(),
            original_estimated_tokens=estimated_tokens,
            optimized_estimated_tokens=estimated_tokens,
        )


def _control_output_or_bypass(
    *,
    controller: OutputBudgetController,
    provider: str,
    body: bytes,
    requested_policy: str | None,
    request_id: str,
) -> OutputControlResult:
    try:
        return controller.apply(
            provider=provider,
            body=body,
            requested_policy=requested_policy,
        )
    except OutputPolicyError:
        raise
    except Exception as error:
        log_event(
            logger,
            "output_control_bypassed",
            request_id=request_id,
            reason=type(error).__name__,
        )
        return OutputControlResult(body=body, events=(), policy=None, max_tokens=None)


def _compress_tool_output_or_bypass(
    *,
    compressor: ToolOutputCompressor,
    body: bytes,
    requested_mode: str | None,
    request_id: str,
) -> ToolCompressionResult:
    try:
        return compressor.apply(body=body, requested_mode=requested_mode)
    except ToolCompressionModeError:
        raise
    except Exception as error:
        log_event(
            logger,
            "tool_output_compression_bypassed",
            request_id=request_id,
            reason=type(error).__name__,
        )
        return ToolCompressionResult(
            body=body,
            events=(),
            mode=None,
            compressed_outputs=0,
            optimized_estimated_tokens=estimate_tokens(body),
        )


def _combine_input_and_compression(
    input_optimization: OptimizationResult,
    compression: ToolCompressionResult,
) -> OptimizationResult:
    return OptimizationResult(
        body=compression.body,
        events=input_optimization.events + compression.events,
        original_estimated_tokens=input_optimization.original_estimated_tokens,
        optimized_estimated_tokens=compression.optimized_estimated_tokens,
    )


def _combine_input_and_routing(
    optimization: OptimizationResult,
    routing: RouteDecision,
) -> OptimizationResult:
    events = optimization.events
    if routing.event is not None:
        events += (routing.event,)
    return OptimizationResult(
        body=routing.body,
        events=events,
        original_estimated_tokens=optimization.original_estimated_tokens,
        optimized_estimated_tokens=estimate_tokens(routing.body),
    )


def _combine_optimization(
    compressed_optimization: OptimizationResult,
    output_control: OutputControlResult,
) -> OptimizationResult:
    return OptimizationResult(
        body=output_control.body,
        events=compressed_optimization.events + output_control.events,
        original_estimated_tokens=compressed_optimization.original_estimated_tokens,
        optimized_estimated_tokens=compressed_optimization.optimized_estimated_tokens,
    )


def _append_events(
    optimization: OptimizationResult,
    *events: OptimizationEvent,
) -> OptimizationResult:
    return OptimizationResult(
        body=optimization.body,
        events=optimization.events + events,
        original_estimated_tokens=optimization.original_estimated_tokens,
        optimized_estimated_tokens=optimization.optimized_estimated_tokens,
    )


def _cache_event(kind: str, detail: str) -> OptimizationEvent:
    return OptimizationEvent(
        kind=kind,
        path="$",
        before_estimated_tokens=0,
        after_estimated_tokens=0,
        detail=detail,
    )


def _cacheable_response(status_code: int, content_type: str, content: bytes) -> bool:
    if status_code != 200 or "application/json" not in content_type.lower():
        return False
    try:
        json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return True


def _initial_cache_status(
    *,
    exact_mode: str,
    semantic_mode: str,
    freshness_bypass: bool,
) -> str:
    if freshness_bypass:
        return "BYPASS"
    modes = {exact_mode, semantic_mode}
    if "refresh" in modes:
        return "REFRESH"
    if "lookup" in modes:
        return "MISS"
    if "bypass" in modes:
        return "BYPASS"
    return "DISABLED"


def _provider_executed_route(routing: RouteDecision) -> str | None:
    if not routing.enabled or routing.cache_only or not routing.model_configured:
        return None
    if not routing.model_changed and (
        routing.original_model is None or routing.routed_model != routing.original_model
    ):
        return None
    return routing.route


def _optimizer_headers(
    headers: dict[str, str],
    *,
    request_id: str,
    optimization: OptimizationResult,
    output_control: OutputControlResult,
    compression: ToolCompressionResult,
    cache_status: str,
    semantic_similarity: float | None = None,
    routing: RouteDecision | None = None,
    executed_route: str | None = None,
    validation: ValidationResult | None = None,
    escalation_attempts: int = 0,
    escalation_cost_usd: Decimal | None = None,
) -> dict[str, str]:
    result = dict(headers)
    result["x-optimizer-request-id"] = request_id
    result["x-optimizer-events"] = str(len(optimization.events))
    removed_tokens = max(
        0,
        optimization.original_estimated_tokens - optimization.optimized_estimated_tokens,
    )
    result["x-optimizer-estimated-input-tokens-removed"] = str(removed_tokens)
    result["x-optimizer-cache"] = cache_status
    if semantic_similarity is not None:
        result["x-optimizer-semantic-similarity"] = f"{semantic_similarity:.6f}"
    if routing is not None and routing.enabled and routing.route is not None:
        result["x-optimizer-route"] = executed_route or routing.route
        result["x-optimizer-route-decision"] = routing.route
        result["x-optimizer-route-source"] = routing.source
        result["x-optimizer-route-applied"] = str(executed_route is not None).lower()
        if routing.routed_model is not None:
            result["x-optimizer-routed-model"] = routing.routed_model
        if routing.model_route is not None:
            result["x-optimizer-decision-model-route"] = routing.model_route
        if routing.model_confidence is not None:
            result["x-optimizer-decision-model-confidence"] = f"{routing.model_confidence:.6f}"
        if routing.model_mode is not None:
            result["x-optimizer-decision-model-mode"] = routing.model_mode
            result["x-optimizer-decision-model-applied"] = str(routing.model_applied).lower()
            if routing.model_fallback_reason is not None:
                result["x-optimizer-decision-model-fallback"] = routing.model_fallback_reason
    if compression.mode is not None:
        result["x-optimizer-tool-compression"] = compression.mode
        result["x-optimizer-tool-outputs-compressed"] = str(compression.compressed_outputs)
    if output_control.policy is not None:
        result["x-optimizer-output-policy"] = output_control.policy
    if output_control.max_tokens is not None:
        result["x-optimizer-output-limit"] = str(output_control.max_tokens)
    if validation is not None and validation.enabled:
        state = "bypass" if validation.passed is None else ("pass" if validation.passed else "fail")
        result["x-optimizer-validation"] = state
        result["x-optimizer-validation-reason"] = validation.reason
        result["x-optimizer-escalations"] = str(escalation_attempts)
        if escalation_cost_usd is not None:
            result["x-optimizer-escalation-cost-usd"] = f"{escalation_cost_usd:.8f}"
    return result


def _record(
    *,
    store: TelemetryStore,
    calculator: CostCalculator,
    request_id: str,
    provider: str,
    model: str | None,
    status: str,
    status_code: int | None,
    started: float,
    usage: Usage,
    optimization: OptimizationResult,
    baseline_usage: Usage | None = None,
    cache_hit: bool = False,
    cache_type: str | None = None,
    routing: RouteDecision | None = None,
    executed_route: str | None = None,
    validation: ValidationResult | None = None,
    validation_attempts: tuple[ValidationAttemptRecord, ...] = (),
    escalation_cost_usd: Decimal | None = None,
    actual_cost_usd: Decimal | None = None,
) -> None:
    latency_ms = (time.perf_counter() - started) * 1_000
    actual_cost = calculator.calculate(model, usage.input_tokens, usage.output_tokens)
    baseline = baseline_usage or usage
    baseline_cost = calculator.calculate(model, baseline.input_tokens, baseline.output_tokens)
    try:
        routing_record = None
        if routing is not None and routing.enabled and routing.route is not None:
            model_agreed = None
            if routing.model_route is not None and routing.rule_route is not None:
                model_agreed = routing.model_route == routing.rule_route
            routing_record = RoutingDecisionRecord(
                request_id=request_id,
                selected_route=routing.route,
                executed_route=executed_route,
                source=routing.source,
                reason=routing.reason,
                original_model=routing.original_model,
                routed_model=routing.routed_model,
                model_configured=routing.model_configured,
                model_changed=routing.model_changed,
                estimated_input_tokens=routing.features.estimated_input_tokens,
                message_count=routing.features.message_count,
                tool_count=routing.features.tool_count,
                has_media=routing.features.has_media,
                streaming=routing.features.streaming,
                cache_only=routing.cache_only,
                rule_route=routing.rule_route,
                model_route=routing.model_route,
                model_confidence=routing.model_confidence,
                model_mode=routing.model_mode,
                model_applied=routing.model_applied,
                model_fallback_reason=routing.model_fallback_reason,
                model_agreed=model_agreed,
            )
        store.record(
            RequestRecord(
                request_id=request_id,
                provider=provider,
                model=model,
                status=status,
                status_code=status_code,
                latency_ms=latency_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                baseline_cost_usd=baseline_cost.baseline_usd,
                actual_cost_usd=(
                    actual_cost_usd if validation_attempts else actual_cost.actual_usd
                ),
                cache_hit=cache_hit,
                cache_type=cache_type,
                original_estimated_input_tokens=optimization.original_estimated_tokens,
                optimized_estimated_input_tokens=optimization.optimized_estimated_tokens,
                validated=(
                    validation.passed if validation is not None and validation.enabled else None
                ),
                escalated=len(validation_attempts) > 1,
                escalation_attempts=max(0, len(validation_attempts) - 1),
                escalation_cost_usd=escalation_cost_usd,
            ),
            optimization.events,
            routing_record,
            validation_attempts,
        )
    except (OSError, sqlite3.Error) as error:
        log_event(
            logger,
            "telemetry_write_failed",
            request_id=request_id,
            reason=type(error).__name__,
        )
    log_event(
        logger,
        "request_complete",
        request_id=request_id,
        provider=provider,
        model=model,
        status=status,
        status_code=status_code,
        latency_ms=round(latency_ms, 2),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_hit=cache_hit,
        cache_type=cache_type,
        route=routing.route if routing is not None else None,
        executed_route=executed_route,
        optimization_events=len(optimization.events),
        estimated_input_tokens_removed=max(
            0,
            optimization.original_estimated_tokens - optimization.optimized_estimated_tokens,
        ),
    )
