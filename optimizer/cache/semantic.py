from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from optimizer.embeddings import Embedding, cosine_similarity


@dataclass(frozen=True, slots=True)
class SemanticCacheEntry:
    body: bytes
    status_code: int
    content_type: str
    input_tokens: int
    output_tokens: int
    created_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class SemanticCacheHit:
    entry: SemanticCacheEntry
    similarity: float


class SemanticCacheBackend(Protocol):
    enabled: bool

    def initialize(self) -> None: ...

    def search(
        self,
        *,
        compatibility_key: str,
        embedding_version: str,
        embedding: Embedding,
        threshold: float,
        limit: int,
        now: float | None = None,
    ) -> SemanticCacheHit | None: ...

    def set(
        self,
        key: str,
        *,
        compatibility_key: str,
        embedding_version: str,
        embedding: Embedding,
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

    def clear(self) -> int: ...


class SQLiteSemanticCache:
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
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    cache_key TEXT PRIMARY KEY,
                    compatibility_key TEXT NOT NULL,
                    embedding_version TEXT NOT NULL,
                    embedding TEXT NOT NULL,
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
                """
                CREATE INDEX IF NOT EXISTS semantic_cache_lookup_idx
                ON semantic_cache(compatibility_key, embedding_version, created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS semantic_cache_expires_at_idx
                ON semantic_cache(expires_at)
                """
            )
        os.chmod(self.path, 0o600)

    def search(
        self,
        *,
        compatibility_key: str,
        embedding_version: str,
        embedding: Embedding,
        threshold: float,
        limit: int,
        now: float | None = None,
    ) -> SemanticCacheHit | None:
        if not self.enabled or not embedding:
            return None
        current_time = time.time() if now is None else now
        with self._connect() as connection:
            connection.execute("DELETE FROM semantic_cache WHERE expires_at <= ?", (current_time,))
            rows = connection.execute(
                """
                SELECT embedding, response_body, status_code, content_type, input_tokens,
                       output_tokens, created_at, expires_at
                FROM semantic_cache
                WHERE compatibility_key = ? AND embedding_version = ? AND expires_at > ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (compatibility_key, embedding_version, current_time, limit),
            ).fetchall()

        best: SemanticCacheHit | None = None
        for row in rows:
            candidate = _decode_embedding(str(row[0]))
            if candidate is None:
                continue
            similarity = cosine_similarity(embedding, candidate)
            if similarity < threshold or (best is not None and similarity <= best.similarity):
                continue
            best = SemanticCacheHit(
                entry=SemanticCacheEntry(
                    body=bytes(row[1]),
                    status_code=int(row[2]),
                    content_type=str(row[3]),
                    input_tokens=int(row[4]),
                    output_tokens=int(row[5]),
                    created_at=float(row[6]),
                    expires_at=float(row[7]),
                ),
                similarity=similarity,
            )
        return best

    def set(
        self,
        key: str,
        *,
        compatibility_key: str,
        embedding_version: str,
        embedding: Embedding,
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
        if not self.enabled or not embedding:
            return
        created_at = time.time() if now is None else now
        expires_at = created_at + ttl_seconds
        with self._connect() as connection:
            connection.execute("DELETE FROM semantic_cache WHERE expires_at <= ?", (created_at,))
            connection.execute(
                """
                INSERT INTO semantic_cache (
                    cache_key, compatibility_key, embedding_version, embedding, provider, model,
                    status_code, content_type, response_body, input_tokens, output_tokens,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    compatibility_key = excluded.compatibility_key,
                    embedding_version = excluded.embedding_version,
                    embedding = excluded.embedding,
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
                    compatibility_key,
                    embedding_version,
                    _encode_embedding(embedding),
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

    def clear(self) -> int:
        if not self.enabled or not self.path.exists():
            return 0
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()
            connection.execute("DELETE FROM semantic_cache")
        return int(row[0]) if row is not None else 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection


def make_semantic_compatibility_key(
    *,
    namespace: str,
    identity: str,
    task: str,
    compatibility_body: bytes,
    embedding_version: str,
) -> str:
    return _digest_parts(
        b"semantic-compatibility-v1",
        namespace.encode(),
        identity.encode(),
        task.encode(),
        compatibility_body,
        embedding_version.encode(),
    )


def make_semantic_entry_key(*, compatibility_key: str, query: str) -> str:
    normalized_query = " ".join(query.casefold().split()).encode("utf-8")
    return _digest_parts(
        b"semantic-entry-v1",
        compatibility_key.encode(),
        normalized_query,
    )


def _digest_parts(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _encode_embedding(embedding: Embedding) -> str:
    values = [[index, round(value, 12)] for index, value in sorted(embedding.items())]
    return json.dumps(values, separators=(",", ":"))


def _decode_embedding(value: str) -> Embedding | None:
    try:
        raw = json.loads(value)
        if not isinstance(raw, list):
            return None
        result: Embedding = {}
        for item in raw:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], int)
                or not isinstance(item[1], (int, float))
            ):
                return None
            result[item[0]] = float(item[1])
        return result
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
