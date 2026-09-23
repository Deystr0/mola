from __future__ import annotations

import math

from optimizer.embeddings import HashingEmbeddingBackend, cosine_similarity


def test_hashing_embedding_is_deterministic_normalized_and_local() -> None:
    backend = HashingEmbeddingBackend(dimensions=256)

    first = backend.embed("Explain Python list comprehensions.")
    second = backend.embed("Explain Python list comprehensions.")

    assert first == second
    assert backend.version == "hashing-word-char-v1:256"
    assert all(0 <= index < 256 for index in first)
    assert math.isclose(sum(value * value for value in first.values()), 1.0)


def test_hashing_embedding_separates_related_and_unrelated_queries() -> None:
    backend = HashingEmbeddingBackend()
    query = backend.embed("How do I reset my password?")
    related = backend.embed("How can I reset my password?")
    unrelated = backend.embed("What is the weather today?")

    assert cosine_similarity(query, related) > 0.75
    assert cosine_similarity(query, unrelated) < 0.25
    assert cosine_similarity({}, related) == 0
