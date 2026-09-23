from __future__ import annotations

import stat
from pathlib import Path

from optimizer.cache import SQLiteExactCache, make_cache_key


def test_exact_cache_round_trip_expiry_and_clear(tmp_path: Path) -> None:
    cache = SQLiteExactCache(tmp_path / "cache.db")
    cache.initialize()

    cache.set(
        "first",
        provider="openai",
        model="test-model",
        body=b'{"answer":"one"}',
        status_code=200,
        content_type="application/json",
        input_tokens=7,
        output_tokens=3,
        ttl_seconds=60,
        now=100,
    )
    entry = cache.get("first", now=159)

    assert entry is not None
    assert entry.body == b'{"answer":"one"}'
    assert entry.input_tokens == 7
    assert entry.output_tokens == 3
    assert cache.get("first", now=160) is None

    cache.set(
        "second",
        provider="anthropic",
        model=None,
        body=b"{}",
        status_code=200,
        content_type="application/json",
        input_tokens=0,
        output_tokens=0,
        ttl_seconds=60,
        now=200,
    )
    assert cache.clear() == 1
    assert cache.get("second", now=201) is None


def test_exact_cache_upsert_replaces_response(tmp_path: Path) -> None:
    cache = SQLiteExactCache(tmp_path / "cache.db")
    cache.initialize()
    common = {
        "provider": "openai",
        "model": "test-model",
        "status_code": 200,
        "content_type": "application/json",
        "input_tokens": 1,
        "output_tokens": 1,
        "ttl_seconds": 60,
    }

    cache.set("same-key", body=b'{"version":1}', now=10, **common)
    cache.set("same-key", body=b'{"version":2}', now=20, **common)

    entry = cache.get("same-key", now=21)
    assert entry is not None
    assert entry.body == b'{"version":2}'
    assert entry.created_at == 20


def test_exact_cache_database_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "cache.db"
    SQLiteExactCache(path).initialize()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_cache_key_is_exact_and_scoped_without_exposing_identity() -> None:
    secret_identity = "credential-secret"
    first = make_cache_key(namespace="openai:a", identity=secret_identity, body=b'{"x":1}')

    assert first == make_cache_key(namespace="openai:a", identity=secret_identity, body=b'{"x":1}')
    assert first != make_cache_key(namespace="openai:a", identity=secret_identity, body=b'{"x":1 }')
    assert first != make_cache_key(namespace="openai:b", identity=secret_identity, body=b'{"x":1}')
    assert first != make_cache_key(namespace="openai:a", identity="another-secret", body=b'{"x":1}')
    assert secret_identity not in first
