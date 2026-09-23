from __future__ import annotations

import stat
from pathlib import Path

from optimizer.cache import (
    SQLiteSemanticCache,
    make_semantic_compatibility_key,
    make_semantic_entry_key,
)
from optimizer.embeddings import HashingEmbeddingBackend


def test_semantic_cache_search_respects_similarity_scope_ttl_and_limit(tmp_path: Path) -> None:
    cache = SQLiteSemanticCache(tmp_path / "semantic.db")
    cache.initialize()
    embeddings = HashingEmbeddingBackend()
    compatibility = "scope-a"
    stored_query = "How do I reset my password?"
    vector = embeddings.embed(stored_query)
    cache.set(
        make_semantic_entry_key(
            compatibility_key=compatibility,
            query=stored_query,
        ),
        compatibility_key=compatibility,
        embedding_version=embeddings.version,
        embedding=vector,
        provider="openai",
        model="test-model",
        body=b'{"answer":"reset link"}',
        status_code=200,
        content_type="application/json",
        input_tokens=12,
        output_tokens=4,
        ttl_seconds=60,
        now=100,
    )

    hit = cache.search(
        compatibility_key=compatibility,
        embedding_version=embeddings.version,
        embedding=embeddings.embed("How can I reset my password?"),
        threshold=0.75,
        limit=10,
        now=120,
    )

    assert hit is not None
    assert hit.similarity > 0.75
    assert hit.entry.body == b'{"answer":"reset link"}'
    assert hit.entry.input_tokens == 12
    assert (
        cache.search(
            compatibility_key="scope-b",
            embedding_version=embeddings.version,
            embedding=vector,
            threshold=0,
            limit=10,
            now=120,
        )
        is None
    )
    assert (
        cache.search(
            compatibility_key=compatibility,
            embedding_version=embeddings.version,
            embedding=vector,
            threshold=0,
            limit=10,
            now=160,
        )
        is None
    )


def test_semantic_cache_clear_and_permissions(tmp_path: Path) -> None:
    path = tmp_path / "semantic.db"
    cache = SQLiteSemanticCache(path)
    cache.initialize()
    embeddings = HashingEmbeddingBackend()
    cache.set(
        "key",
        compatibility_key="scope",
        embedding_version=embeddings.version,
        embedding=embeddings.embed("query"),
        provider="anthropic",
        model="test-model",
        body=b"{}",
        status_code=200,
        content_type="application/json",
        input_tokens=0,
        output_tokens=0,
        ttl_seconds=60,
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert cache.clear() == 1
    assert cache.clear() == 0


def test_semantic_keys_are_deterministic_scoped_and_do_not_expose_content() -> None:
    secret = "credential-secret"
    prompt = "private prompt"
    first = make_semantic_compatibility_key(
        namespace="openai:a",
        identity=secret,
        task="support",
        compatibility_body=b'{"model":"x"}',
        embedding_version="v1",
    )

    assert first == make_semantic_compatibility_key(
        namespace="openai:a",
        identity=secret,
        task="support",
        compatibility_body=b'{"model":"x"}',
        embedding_version="v1",
    )
    assert first != make_semantic_compatibility_key(
        namespace="openai:a",
        identity=secret,
        task="billing",
        compatibility_body=b'{"model":"x"}',
        embedding_version="v1",
    )
    entry_key = make_semantic_entry_key(compatibility_key=first, query=prompt)
    assert secret not in first
    assert prompt not in entry_key
