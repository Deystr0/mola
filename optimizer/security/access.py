from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import time
from dataclasses import dataclass

from optimizer.config import RemoteAuthenticationConfig, SecurityConfig


@dataclass(frozen=True, slots=True)
class AccessDecision:
    allowed: bool
    status_code: int
    reason: str
    principal: str
    retry_after_seconds: int | None = None


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class AccessController:
    """Rate-limit authentication by peer address before checking the API key."""

    def __init__(self, config: SecurityConfig) -> None:
        self.config = config
        self._expected_key = (
            os.environ.get(config.remote_auth.api_key_env) if config.remote_auth.enabled else None
        )
        if config.remote_auth.enabled and not self._expected_key:
            raise ValueError(
                f"Remote authentication is enabled but {config.remote_auth.api_key_env} is unset."
            )
        self._buckets: dict[str, _Bucket] = {}

    def check(
        self,
        *,
        supplied_key: str | None,
        client_host: str | None,
        now: float | None = None,
    ) -> AccessDecision:
        peer = _principal(None, client_host)
        principal = _principal(supplied_key, client_host)
        current = time.monotonic() if now is None else now
        if self.config.remote_auth.enabled:
            limited = self._consume(
                f"auth:{peer}",
                current,
                self.config.rate_limit.authentication_requests_per_minute,
                self.config.rate_limit.authentication_burst,
            )
            if limited is not None:
                return AccessDecision(False, 429, "rate_limited", peer, limited)
        if self.config.remote_auth.enabled and (
            supplied_key is None
            or not hmac.compare_digest(
                supplied_key,
                self._expected_key or "",
            )
        ):
            return AccessDecision(False, 401, "invalid_api_key", principal)
        if not self.config.rate_limit.enabled:
            return AccessDecision(True, 200, "allowed", principal)
        limited = self._consume(
            f"request:{principal}",
            current,
            self.config.rate_limit.requests_per_minute,
            self.config.rate_limit.burst,
        )
        if limited is not None:
            return AccessDecision(False, 429, "rate_limited", principal, limited)
        return AccessDecision(True, 200, "allowed", principal)

    def _consume(self, key: str, now: float, per_minute: int, burst: int) -> int | None:
        rate = per_minute / 60
        capacity = float(burst)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=capacity, updated_at=now)
            self._buckets[key] = bucket
        elapsed = max(0.0, now - bucket.updated_at)
        bucket.tokens = min(capacity, bucket.tokens + (elapsed * rate))
        bucket.updated_at = now
        if bucket.tokens < 1:
            return max(1, int((1 - bucket.tokens) / rate + 0.999))
        bucket.tokens -= 1
        return None


def bind_host_is_loopback(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "::1", "localhost"}


def client_ip_for_rate_limit(
    peer: str | None,
    forwarded: str | None,
    config: RemoteAuthenticationConfig,
) -> str | None:
    if peer not in config.trusted_proxy_ips or forwarded is None:
        return peer
    try:
        return str(ipaddress.ip_address(forwarded))
    except ValueError:
        return peer


def _principal(supplied_key: str | None, client_host: str | None) -> str:
    value = supplied_key if supplied_key is not None else (client_host or "unknown")
    return hashlib.sha256(value.encode()).hexdigest()
