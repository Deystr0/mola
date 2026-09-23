from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from typing import Protocol, TypeAlias

Embedding: TypeAlias = dict[int, float]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class EmbeddingBackend(Protocol):
    @property
    def version(self) -> str: ...

    def embed(self, text: str) -> Embedding: ...


class HashingEmbeddingBackend:
    """Small deterministic CPU embedding with no model files or external services."""

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    @property
    def version(self) -> str:
        return f"hashing-word-char-v1:{self.dimensions}"

    def embed(self, text: str) -> Embedding:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        tokens = _TOKEN_RE.findall(normalized)
        if not tokens:
            return {}

        values: defaultdict[int, float] = defaultdict(float)
        for token in tokens:
            self._add(values, f"w:{token}", 1.0)
            padded = f"^{token}$"
            for index in range(max(0, len(padded) - 2)):
                self._add(values, f"c:{padded[index : index + 3]}", 0.15)
        for left, right in zip(tokens, tokens[1:], strict=False):
            self._add(values, f"b:{left}:{right}", 0.65)

        magnitude = math.sqrt(sum(value * value for value in values.values()))
        if magnitude == 0:
            return {}
        return {index: value / magnitude for index, value in values.items() if value}

    def _add(self, values: defaultdict[int, float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        index = number % self.dimensions
        sign = -1.0 if number & (1 << 63) else 1.0
        values[index] += sign * weight


def cosine_similarity(left: Embedding, right: Embedding) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    similarity = sum(value * right.get(index, 0.0) for index, value in left.items())
    return max(-1.0, min(1.0, similarity))
