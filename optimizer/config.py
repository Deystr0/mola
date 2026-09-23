from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

APP_NAME = "llm-optimizer"
DEFAULT_CONFIG_PATH = Path("~/.config/llm-optimizer/optimizer.yaml").expanduser()
DEFAULT_DATABASE_PATH = Path("~/.local/share/llm-optimizer/telemetry.db").expanduser()
DEFAULT_CACHE_DATABASE_PATH = Path("~/.local/share/llm-optimizer/cache.db").expanduser()
DEFAULT_SEMANTIC_CACHE_DATABASE_PATH = Path(
    "~/.local/share/llm-optimizer/semantic-cache.db"
).expanduser()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=4000, ge=1, le=65535)


class ProviderConfig(StrictModel):
    base_url: str
    api_key_env: str
    timeout_seconds: float = Field(default=120.0, gt=0)
    allowed_hosts: list[str] = Field(default_factory=list)
    allow_private_network: bool = False
    allow_insecure_http: bool = False

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        if value != value.strip() or "\\" in value or any(ord(char) < 32 for char in value):
            raise ValueError("provider base_url contains invalid characters")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("provider base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("provider base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("provider base_url must not contain query or fragment")
        try:
            _ = parsed.port
        except ValueError as error:
            raise ValueError("provider base_url has an invalid port") from error
        if any(part == ".." for part in unquote(parsed.path).split("/")):
            raise ValueError("provider base_url must not contain parent path segments")
        _validate_hostname(parsed.hostname)
        return value.rstrip("/")

    @field_validator("allowed_hosts")
    @classmethod
    def validate_allowed_hosts(cls, values: list[str]) -> list[str]:
        for value in values:
            _validate_hostname(value)
            if value != value.lower() or value.endswith("."):
                raise ValueError("allowed_hosts must use lowercase exact hostnames")
        return values


class ProvidersConfig(StrictModel):
    openai: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        )
    )
    anthropic: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
        )
    )
    openrouter: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        )
    )
    ollama: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            base_url="http://127.0.0.1:11434/v1",
            api_key_env="OLLAMA_API_KEY",
            allow_private_network=True,
            allow_insecure_http=True,
        )
    )

    @model_validator(mode="after")
    def validate_destinations(self) -> ProvidersConfig:
        for name in ("openai", "anthropic", "openrouter", "ollama"):
            provider = getattr(self, name)
            host = urlsplit(provider.base_url).hostname
            assert host is not None
            defaults = {
                "openai": {"api.openai.com"},
                "anthropic": {"api.anthropic.com"},
                "openrouter": {"openrouter.ai"},
                "ollama": {"127.0.0.1", "::1", "localhost", "host.docker.internal"},
            }
            if host not in defaults[name] | set(provider.allowed_hosts):
                raise ValueError(f"{name} provider host is not in its allowlist")
            address = _ip_address(host)
            if address is not None:
                if address.is_link_local or address.is_multicast or address.is_unspecified:
                    raise ValueError(f"{name} provider address is not permitted")
                if not address.is_global and not provider.allow_private_network:
                    raise ValueError(f"{name} provider private address requires opt-in")
            elif _is_private_name(host) and not provider.allow_private_network:
                raise ValueError(f"{name} provider private hostname requires opt-in")
            if urlsplit(provider.base_url).scheme == "http" and not provider.allow_insecure_http:
                raise ValueError(f"{name} provider HTTP requires explicit opt-in")
        return self


def _ip_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_private_name(host: str) -> bool:
    return host == "localhost" or host.endswith((".localhost", ".local", ".internal"))


def _validate_hostname(host: str) -> None:
    if _ip_address(host) is not None:
        return
    if not host.isascii() or len(host) > 253:
        raise ValueError("provider hostname must be an ASCII DNS name or IP address")
    if host.replace(".", "").isdigit() or host.lower().startswith("0x"):
        raise ValueError("provider hostname must not use an ambiguous numeric address")
    if not all(
        re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", part)
        for part in host.split(".")
    ):
        raise ValueError("provider hostname must be an exact DNS name")


class TelemetryConfig(StrictModel):
    enabled: bool = True
    database_path: Path = DEFAULT_DATABASE_PATH

    @field_validator("database_path")
    @classmethod
    def expand_database_path(cls, value: Path) -> Path:
        return value.expanduser()


class ModelPrice(StrictModel):
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)


class PricingConfig(StrictModel):
    models: dict[str, ModelPrice] = Field(default_factory=dict)


class ContextIntelligenceConfig(StrictModel):
    enabled: bool = False
    min_recent_turns: int = Field(default=2, ge=0)
    max_prunable_score: float = Field(default=0.35, ge=0, le=1)


class InputOptimizationConfig(StrictModel):
    deduplicate_messages: bool = False
    deduplicate_content_blocks: bool = False
    deduplicate_tool_outputs: bool = False
    normalize_tool_output_line_endings: bool = False
    max_estimated_tokens: int | None = Field(default=None, ge=1)
    context: ContextIntelligenceConfig = Field(default_factory=ContextIntelligenceConfig)

    @model_validator(mode="after")
    def require_budget_for_context_intelligence(self) -> InputOptimizationConfig:
        if self.context.enabled and self.max_estimated_tokens is None:
            raise ValueError(
                "max_estimated_tokens is required when context intelligence is enabled"
            )
        return self


OutputPolicyName: TypeAlias = Literal[
    "machine_to_machine",
    "tool_call",
    "code",
    "analysis",
    "user_answer",
    "documentation",
]

ToolCompressionMode: TypeAlias = Literal["lite", "full", "ultra"]
ModelRouteName: TypeAlias = Literal["local", "tool", "cheap", "mid", "frontier"]


class OutputPolicyConfig(StrictModel):
    max_tokens: int | None = Field(default=None, ge=1)


class OutputPoliciesConfig(StrictModel):
    machine_to_machine: OutputPolicyConfig = Field(default_factory=OutputPolicyConfig)
    tool_call: OutputPolicyConfig = Field(default_factory=OutputPolicyConfig)
    code: OutputPolicyConfig = Field(default_factory=OutputPolicyConfig)
    analysis: OutputPolicyConfig = Field(default_factory=OutputPolicyConfig)
    user_answer: OutputPolicyConfig = Field(default_factory=OutputPolicyConfig)
    documentation: OutputPolicyConfig = Field(default_factory=OutputPolicyConfig)


class OutputOptimizationConfig(StrictModel):
    enabled: bool = False
    default_policy: OutputPolicyName | None = None
    openai_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    policies: OutputPoliciesConfig = Field(default_factory=OutputPoliciesConfig)


class ExactCacheConfig(StrictModel):
    enabled: bool = False
    database_path: Path = DEFAULT_CACHE_DATABASE_PATH
    ttl_seconds: int = Field(default=3600, ge=1)

    @field_validator("database_path")
    @classmethod
    def expand_database_path(cls, value: Path) -> Path:
        return value.expanduser()


class SemanticCacheConfig(StrictModel):
    enabled: bool = False
    database_path: Path = DEFAULT_SEMANTIC_CACHE_DATABASE_PATH
    ttl_seconds: int = Field(default=900, ge=1)
    similarity_threshold: float = Field(default=0.75, ge=0, le=1)
    max_candidates: int = Field(default=200, ge=1)
    embedding_dimensions: int = Field(default=1024, ge=64, le=16384)

    @field_validator("database_path")
    @classmethod
    def expand_database_path(cls, value: Path) -> Path:
        return value.expanduser()


class CacheOptimizationConfig(StrictModel):
    exact: ExactCacheConfig = Field(default_factory=ExactCacheConfig)
    semantic: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)


class ToolOutputCompressionConfig(StrictModel):
    enabled: bool = False
    mode: ToolCompressionMode = "ultra"
    min_characters: int = Field(default=2000, ge=1)


class CompressionConfig(StrictModel):
    tool_output: ToolOutputCompressionConfig = Field(default_factory=ToolOutputCompressionConfig)


class ProviderRouteModels(StrictModel):
    local: str | None = None
    tool: str | None = None
    cheap: str | None = None
    mid: str | None = None
    frontier: str | None = None

    @field_validator("local", "tool", "cheap", "mid", "frontier")
    @classmethod
    def validate_model_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("model names must not be empty")
        return normalized


class RoutingModelsConfig(StrictModel):
    openai: ProviderRouteModels = Field(default_factory=ProviderRouteModels)
    anthropic: ProviderRouteModels = Field(default_factory=ProviderRouteModels)
    openrouter: ProviderRouteModels = Field(default_factory=ProviderRouteModels)
    ollama: ProviderRouteModels = Field(default_factory=ProviderRouteModels)


class DecisionModelConfig(StrictModel):
    enabled: bool = False
    mode: Literal["shadow", "active"] = "shadow"
    artifact_path: Path | None = None
    min_confidence: float = Field(default=0.80, ge=0, le=1)
    min_validation_samples: int = Field(default=100, ge=1)
    min_validation_accuracy: float = Field(default=0.80, ge=0, le=1)

    @field_validator("artifact_path")
    @classmethod
    def expand_artifact_path(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    @model_validator(mode="after")
    def require_artifact_when_enabled(self) -> DecisionModelConfig:
        if self.enabled and self.artifact_path is None:
            raise ValueError("artifact_path is required when the decision model is enabled")
        return self


class RoutingOptimizationConfig(StrictModel):
    enabled: bool = False
    default_route: Literal["cheap", "mid", "frontier"] = "mid"
    cheap_max_estimated_tokens: int = Field(default=600, ge=1)
    frontier_min_estimated_tokens: int = Field(default=8000, ge=2)
    frontier_min_messages: int = Field(default=20, ge=1)
    models: RoutingModelsConfig = Field(default_factory=RoutingModelsConfig)
    decision_model: DecisionModelConfig = Field(default_factory=DecisionModelConfig)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> RoutingOptimizationConfig:
        if self.cheap_max_estimated_tokens >= self.frontier_min_estimated_tokens:
            raise ValueError(
                "cheap_max_estimated_tokens must be lower than frontier_min_estimated_tokens"
            )
        if self.decision_model.enabled and not self.enabled:
            raise ValueError("routing must be enabled when the decision model is enabled")
        return self


class EscalationConfig(StrictModel):
    enabled: bool = False
    max_attempts: int = Field(default=3, ge=1, le=3)


class ValidationConfig(StrictModel):
    enabled: bool = False
    require_non_empty: bool = True
    reject_truncated: bool = True
    reject_refusal: bool = True
    validate_json_mode: bool = True
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)

    @model_validator(mode="after")
    def require_validation_for_escalation(self) -> ValidationConfig:
        if self.escalation.enabled and not self.enabled:
            raise ValueError("validation must be enabled when escalation is enabled")
        return self


class OptimizationConfig(StrictModel):
    input: InputOptimizationConfig = Field(default_factory=InputOptimizationConfig)
    output: OutputOptimizationConfig = Field(default_factory=OutputOptimizationConfig)
    cache: CacheOptimizationConfig = Field(default_factory=CacheOptimizationConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    routing: RoutingOptimizationConfig = Field(default_factory=RoutingOptimizationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @model_validator(mode="after")
    def require_routing_for_escalation(self) -> OptimizationConfig:
        if self.validation.escalation.enabled and not self.routing.enabled:
            raise ValueError("routing must be enabled when escalation is enabled")
        return self


class RemoteAuthenticationConfig(StrictModel):
    enabled: bool = False
    api_key_env: str = "OPTIMIZER_API_KEY"
    allow_health_unauthenticated: bool = True
    trusted_proxy_ips: list[str] = Field(default_factory=list)

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_env(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("api_key_env must not be empty")
        return normalized

    @field_validator("trusted_proxy_ips")
    @classmethod
    def validate_trusted_proxy_ips(cls, values: list[str]) -> list[str]:
        for value in values:
            try:
                ipaddress.ip_address(value)
            except ValueError as error:
                raise ValueError("trusted_proxy_ips requires exact IP addresses") from error
        return values


class RateLimitConfig(StrictModel):
    enabled: bool = False
    requests_per_minute: int = Field(default=60, ge=1)
    burst: int = Field(default=10, ge=1)
    authentication_requests_per_minute: int = Field(default=30, ge=1)
    authentication_burst: int = Field(default=5, ge=1)


class RuntimeLimitsConfig(StrictModel):
    max_request_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_concurrent_requests: int = Field(default=100, ge=1)
    concurrency_wait_seconds: float = Field(default=0.25, ge=0, le=30)


class SecurityConfig(StrictModel):
    remote_auth: RemoteAuthenticationConfig = Field(default_factory=RemoteAuthenticationConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    limits: RuntimeLimitsConfig = Field(default_factory=RuntimeLimitsConfig)


class LoggingConfig(StrictModel):
    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("must be a standard Python logging level")
        return normalized


class AppConfig(StrictModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def require_auth_for_remote_bind(self) -> AppConfig:
        if (
            self.server.host not in {"127.0.0.1", "::1", "localhost"}
            and not self.security.remote_auth.enabled
        ):
            raise ValueError("remote authentication must be enabled for a non-loopback bind host")
        return self


def resolve_config_path(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser()
    configured = os.environ.get("OPTIMIZER_CONFIG")
    return Path(configured).expanduser() if configured else DEFAULT_CONFIG_PATH


def load_config(path: Path | None = None, *, require_file: bool = True) -> AppConfig:
    config_path = resolve_config_path(path)
    if not config_path.exists():
        if require_file:
            raise FileNotFoundError(
                f"Configuration not found at {config_path}. Run 'optimizer init' first."
            )
        return AppConfig()

    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping.")
    return AppConfig.model_validate(raw)


def default_config_yaml() -> str:
    return """server:
  host: 127.0.0.1
  port: 4000

providers:
  openai:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    timeout_seconds: 120
  anthropic:
    base_url: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    timeout_seconds: 120
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    timeout_seconds: 120
  ollama:
    base_url: http://127.0.0.1:11434/v1
    api_key_env: OLLAMA_API_KEY
    timeout_seconds: 120
    allow_private_network: true
    allow_insecure_http: true

telemetry:
  enabled: true
  database_path: ~/.local/share/llm-optimizer/telemetry.db

pricing:
  models: {}

optimization:
  input:
    deduplicate_messages: false
    deduplicate_content_blocks: false
    deduplicate_tool_outputs: false
    normalize_tool_output_line_endings: false
    max_estimated_tokens: null
    context:
      enabled: false
      min_recent_turns: 2
      max_prunable_score: 0.35
  output:
    enabled: false
    default_policy: null
    openai_parameter: max_completion_tokens
    policies:
      machine_to_machine: {max_tokens: null}
      tool_call: {max_tokens: null}
      code: {max_tokens: null}
      analysis: {max_tokens: null}
      user_answer: {max_tokens: null}
      documentation: {max_tokens: null}
  cache:
    exact:
      enabled: false
      database_path: ~/.local/share/llm-optimizer/cache.db
      ttl_seconds: 3600
    semantic:
      enabled: false
      database_path: ~/.local/share/llm-optimizer/semantic-cache.db
      ttl_seconds: 900
      similarity_threshold: 0.75
      max_candidates: 200
      embedding_dimensions: 1024
  compression:
    tool_output:
      enabled: false
      mode: ultra
      min_characters: 2000
  routing:
    enabled: false
    default_route: mid
    cheap_max_estimated_tokens: 600
    frontier_min_estimated_tokens: 8000
    frontier_min_messages: 20
    models:
      openai:
        local: null
        tool: null
        cheap: null
        mid: null
        frontier: null
      anthropic:
        local: null
        tool: null
        cheap: null
        mid: null
        frontier: null
      openrouter:
        local: null
        tool: null
        cheap: null
        mid: null
        frontier: null
      ollama:
        local: null
        tool: null
        cheap: null
        mid: null
        frontier: null
    decision_model:
      enabled: false
      mode: shadow
      artifact_path: null
      min_confidence: 0.80
      min_validation_samples: 100
      min_validation_accuracy: 0.80
  validation:
    enabled: false
    require_non_empty: true
    reject_truncated: true
    reject_refusal: true
    validate_json_mode: true
    escalation:
      enabled: false
      max_attempts: 3

security:
  remote_auth:
    enabled: false
    api_key_env: OPTIMIZER_API_KEY
    allow_health_unauthenticated: true
    trusted_proxy_ips: []
  rate_limit:
    enabled: false
    requests_per_minute: 60
    burst: 10
    authentication_requests_per_minute: 30
    authentication_burst: 5
  limits:
    max_request_bytes: 10485760
    max_concurrent_requests: 100
    concurrency_wait_seconds: 0.25

logging:
  level: INFO
"""


def initialize_config(path: Path | None = None, *, force: bool = False) -> Path:
    config_path = resolve_config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and not force:
        raise FileExistsError(f"Configuration already exists at {config_path}.")
    config_path.write_text(default_config_yaml(), encoding="utf-8")
    return config_path
