import pytest

from optimizer.config import (
    RateLimitConfig,
    RemoteAuthenticationConfig,
    SecurityConfig,
)
from optimizer.security import AccessController, bind_host_is_loopback, client_ip_for_rate_limit


def test_remote_authentication_requires_configured_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_GATEWAY_KEY", raising=False)
    config = SecurityConfig(
        remote_auth=RemoteAuthenticationConfig(
            enabled=True,
            api_key_env="TEST_GATEWAY_KEY",
        )
    )

    with pytest.raises(ValueError, match="TEST_GATEWAY_KEY is unset"):
        AccessController(config)


def test_authentication_uses_dedicated_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_GATEWAY_KEY", "correct-secret")
    controller = AccessController(
        SecurityConfig(
            remote_auth=RemoteAuthenticationConfig(
                enabled=True,
                api_key_env="TEST_GATEWAY_KEY",
            )
        )
    )

    assert controller.check(supplied_key="correct-secret", client_host="client").allowed
    denied = controller.check(supplied_key="wrong", client_host="client")
    assert denied.allowed is False
    assert denied.status_code == 401


def test_rate_limit_uses_token_bucket() -> None:
    controller = AccessController(
        SecurityConfig(rate_limit=RateLimitConfig(enabled=True, requests_per_minute=60, burst=2))
    )

    assert controller.check(supplied_key=None, client_host="client", now=0).allowed
    assert controller.check(supplied_key=None, client_host="client", now=0).allowed
    denied = controller.check(supplied_key=None, client_host="client", now=0)
    replenished = controller.check(supplied_key=None, client_host="client", now=1)

    assert denied.status_code == 429
    assert denied.retry_after_seconds == 1
    assert replenished.allowed


def test_loopback_detection_is_explicit() -> None:
    assert bind_host_is_loopback("127.0.0.1")
    assert bind_host_is_loopback("::1")
    assert bind_host_is_loopback("localhost")
    assert not bind_host_is_loopback("0.0.0.0")


def test_bad_keys_share_a_peer_quota_even_when_normal_rate_limit_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_GATEWAY_KEY", "correct-secret")
    controller = AccessController(
        SecurityConfig(
            remote_auth=RemoteAuthenticationConfig(enabled=True, api_key_env="TEST_GATEWAY_KEY"),
            rate_limit=RateLimitConfig(
                enabled=False,
                authentication_requests_per_minute=60,
                authentication_burst=2,
            ),
        )
    )

    assert (
        controller.check(supplied_key="guess-one", client_host="peer-a", now=0).status_code == 401
    )
    assert (
        controller.check(supplied_key="guess-two", client_host="peer-a", now=0).status_code == 401
    )
    denied = controller.check(supplied_key="guess-three", client_host="peer-a", now=0)
    other_peer = controller.check(supplied_key="wrong", client_host="peer-b", now=0)

    assert denied.status_code == 429
    assert denied.retry_after_seconds == 1
    assert other_peer.status_code == 401


def test_proxy_address_is_used_only_for_explicitly_trusted_peer() -> None:
    config = RemoteAuthenticationConfig(trusted_proxy_ips=["127.0.0.1"])
    assert client_ip_for_rate_limit("127.0.0.1", "192.0.2.1", config) == "192.0.2.1"
    assert client_ip_for_rate_limit("192.0.2.2", "192.0.2.1", config) == "192.0.2.2"
    assert client_ip_for_rate_limit("127.0.0.1", "1.2.3.4, 5.6.7.8", config) == "127.0.0.1"
