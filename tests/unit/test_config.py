from pathlib import Path

import pytest
from pydantic import ValidationError

from optimizer.config import ProviderConfig, ProvidersConfig, initialize_config, load_config


def test_initialize_and_load_config(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.yaml"

    created = initialize_config(path)
    config = load_config(path)

    assert created == path
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 4000
    assert config.providers.openai.base_url == "https://api.openai.com/v1"
    assert config.providers.openrouter.base_url == "https://openrouter.ai/api/v1"
    assert config.providers.ollama.base_url == "http://127.0.0.1:11434/v1"
    assert config.optimization.input.deduplicate_messages is False
    assert config.optimization.input.max_estimated_tokens is None
    assert config.optimization.input.context.enabled is False
    assert config.optimization.input.context.min_recent_turns == 2
    assert config.optimization.input.context.max_prunable_score == 0.35
    assert config.optimization.output.enabled is False
    assert config.optimization.output.default_policy is None
    assert config.optimization.cache.exact.enabled is False
    assert config.optimization.cache.exact.ttl_seconds == 3600
    assert config.optimization.cache.semantic.enabled is False
    assert config.optimization.cache.semantic.ttl_seconds == 900
    assert config.optimization.cache.semantic.similarity_threshold == 0.75
    assert config.optimization.compression.tool_output.enabled is False
    assert config.optimization.compression.tool_output.mode == "ultra"
    assert config.optimization.compression.tool_output.min_characters == 2000
    assert config.optimization.routing.enabled is False
    assert config.optimization.routing.default_route == "mid"
    assert config.optimization.routing.cheap_max_estimated_tokens == 600
    assert config.optimization.routing.frontier_min_estimated_tokens == 8000
    assert config.optimization.routing.decision_model.enabled is False
    assert config.optimization.validation.enabled is False
    assert config.optimization.validation.escalation.enabled is False
    assert config.security.remote_auth.enabled is False
    assert config.security.rate_limit.enabled is False


def test_initialize_refuses_to_replace_config(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.yaml"
    initialize_config(path)

    with pytest.raises(FileExistsError):
        initialize_config(path)


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.yaml"
    path.write_text("unexpected: true\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_v02_config_without_optimization_section_remains_valid(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.yaml"
    path.write_text("server:\n  host: 127.0.0.1\n  port: 4000\n", encoding="utf-8")

    config = load_config(path)

    assert config.optimization.input.deduplicate_messages is False
    assert config.optimization.input.max_estimated_tokens is None
    assert config.optimization.output.enabled is False
    assert config.optimization.cache.exact.enabled is False
    assert config.optimization.cache.semantic.enabled is False
    assert config.optimization.compression.tool_output.mode == "ultra"
    assert config.optimization.routing.enabled is False


@pytest.mark.parametrize(
    "routing_yaml",
    [
        "enabled: true\n    cheap_max_estimated_tokens: 9000\n"
        "    frontier_min_estimated_tokens: 8000",
        "enabled: true\n    decision_model:\n      enabled: true\n      artifact_path: null",
        "enabled: false\n    decision_model:\n"
        "      enabled: true\n      artifact_path: /tmp/model.json",
    ],
)
def test_invalid_routing_configuration_is_rejected(
    tmp_path: Path,
    routing_yaml: str,
) -> None:
    path = tmp_path / "optimizer.yaml"
    path.write_text(
        "optimization:\n  routing:\n    " + routing_yaml + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_context_intelligence_requires_an_input_budget(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.yaml"
    path.write_text(
        "optimization:\n  input:\n    context:\n      enabled: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="max_estimated_tokens is required"):
        load_config(path)


def test_escalation_requires_validation_and_routing(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.yaml"
    path.write_text(
        "optimization:\n  validation:\n    enabled: true\n    escalation:\n      enabled: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="routing must be enabled"):
        load_config(path)


def test_non_loopback_bind_requires_remote_authentication(tmp_path: Path) -> None:
    path = tmp_path / "optimizer.yaml"
    path.write_text("server:\n  host: 0.0.0.0\n  port: 4000\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="remote authentication must be enabled"):
        load_config(path)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://169.254.169.254/latest",
        "http://127.0.0.1/v1",
        "https://127.1/v1",
        "https://metadata.google.internal/v1",
        "https://evil.example/v1",
        "https://api.openai.com@evil.example/v1",
        "https://api.openai.com/v1?redirect=evil",
        "file:///etc/passwd",
    ],
)
def test_provider_destination_rejects_unapproved_or_unsafe_urls(base_url: str) -> None:
    with pytest.raises(ValidationError):
        ProvidersConfig(openai=ProviderConfig(base_url=base_url, api_key_env="TEST_KEY"))


def test_custom_provider_host_requires_exact_allowlist_and_http_opt_in() -> None:
    with pytest.raises(ValidationError, match="allowlist"):
        ProvidersConfig(
            openai=ProviderConfig(base_url="https://custom.example/v1", api_key_env="TEST_KEY")
        )
    configured = ProvidersConfig(
        openai=ProviderConfig(
            base_url="https://custom.example/v1",
            api_key_env="TEST_KEY",
            allowed_hosts=["custom.example"],
        )
    )
    assert configured.openai.base_url == "https://custom.example/v1"


def test_local_compatible_provider_requires_explicit_private_network_and_http_opt_in() -> None:
    with pytest.raises(ValidationError, match="private address"):
        ProvidersConfig(
            openai=ProviderConfig(
                base_url="http://127.0.0.1:8000/v1",
                api_key_env="TEST_KEY",
                allowed_hosts=["127.0.0.1"],
                allow_insecure_http=True,
            )
        )
    configured = ProvidersConfig(
        openai=ProviderConfig(
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="TEST_KEY",
            allowed_hosts=["127.0.0.1"],
            allow_private_network=True,
            allow_insecure_http=True,
        )
    )
    assert configured.openai.base_url == "http://127.0.0.1:8000/v1"
