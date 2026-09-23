from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CacheEntry:
    body: bytes
    status_code: int
    content_type: str
    input_tokens: int
    output_tokens: int
    created_at: float
    expires_at: float


class ExactCacheBackend(Protocol):
    enabled: bool

    def initialize(self) -> None: ...

    def get(self, key: str, *, now: float | None = None) -> CacheEntry | None: ...

    def set(
        self,
        key: str,
        *,
        provider: str,
        model: str | None,
        body: bytes,
        status_code: int,
        content_type: str,
        input_tokens: int,
        output_tokens: int,
        ttl_seconds: int,
        now: float | None = None,
    ) -> None: ...

    def delete(self, key: str) -> None: ...

    def clear(self) -> int: ...


class SQLiteExactCache:
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
                CREATE TABLE IF NOT EXISTS exact_cache (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT,
                    status_code INTEGER NOT NULL,
                    content_type TEXT NOT NULL,
                    response_body BLOB NOT NULL,
                    input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
                    output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS exact_cache_expires_at_idx ON exact_cache(expires_at)"
            )
        os.chmod(self.path, 0o600)

    def get(self, key: str, *, now: float | None = None) -> CacheEntry | None:
        if not self.enabled:
            return None
        current_time = time.time() if now is None else now
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM exact_cache WHERE cache_key = ? AND expires_at <= ?",
                (key, current_time),
            )
            row = connection.execute(
                """
                SELECT response_body, status_code, content_type, input_tokens, output_tokens,
                       created_at, expires_at
                FROM exact_cache
                WHERE cache_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return CacheEntry(
            body=bytes(row[0]),
            status_code=int(row[1]),
            content_type=str(row[2]),
            input_tokens=int(row[3]),
            output_tokens=int(row[4]),
            created_at=float(row[5]),
            expires_at=float(row[6]),
        )

    def set(
        self,
        key: str,
        *,
        provider: str,
        model: str | None,
        body: bytes,
        status_code: int,
        content_type: str,
        input_tokens: int,
        output_tokens: int,
        ttl_seconds: int,
        now: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        created_at = time.time() if now is None else now
        expires_at = created_at + ttl_seconds
        with self._connect() as connection:
            connection.execute("DELETE FROM exact_cache WHERE expires_at <= ?", (created_at,))
            connection.execute(
                """
                INSERT INTO exact_cache (
                    cache_key, provider, model, status_code, content_type, response_body,
                    input_tokens, output_tokens, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    provider = excluded.provider,
                    model = excluded.model,
                    status_code = excluded.status_code,
                    content_type = excluded.content_type,
                    response_body = excluded.response_body,
                    input_tokens = excluded.input_tokens,
                    output_tokens = excluded.output_tokens,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    key,
                    provider,
                    model,
                    status_code,
                    content_type,
                    body,
                    input_tokens,
                    output_tokens,
                    created_at,
                    expires_at,
                ),
            )

    def delete(self, key: str) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            connection.execute("DELETE FROM exact_cache WHERE cache_key = ?", (key,))

    def clear(self) -> int:
        if not self.enabled or not self.path.exists():
            return 0
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM exact_cache").fetchone()
            connection.execute("DELETE FROM exact_cache")
        return int(row[0]) if row is not None else 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection


def make_cache_key(*, namespace: str, identity: str, body: bytes) -> str:
    digest = hashlib.sha256()
    for part in (b"exact-cache-v1", namespace.encode(), identity.encode(), body):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()
