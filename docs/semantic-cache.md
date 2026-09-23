# Semantic cache

V0.7 provides an opt-in local cache that can reuse a response for a sufficiently similar latest
human query. It is intentionally conservative: exact-cache lookup runs first, and semantic
comparison is allowed only inside an otherwise exact compatibility scope.

## Configuration

```yaml
optimization:
  cache:
    semantic:
      enabled: false
      database_path: ~/.local/share/llm-optimizer/semantic-cache.db
      ttl_seconds: 900
      similarity_threshold: 0.75
      max_candidates: 200
      embedding_dimensions: 1024
```

The cache is disabled by default because enabling it stores successful provider responses and
sparse embeddings on local disk. The threshold accepts values from `0` through `1`; higher values
require a closer match. Expired entries are deleted during normal searches and writes.

## Local embeddings and search

`HashingEmbeddingBackend` normalizes the query and hashes word unigrams, word bigrams, and character
trigrams into a fixed-size sparse vector. Search uses cosine similarity. This implementation is
deterministic, CPU-only, and makes no network request or model download.

This is a compact lexical-semantic representation, not a neural sentence model. It handles close
paraphrases and wording changes but may miss conceptually equivalent queries that share little
vocabulary. It may also produce false positives near a permissive threshold. Keep the default
threshold or raise it for higher-risk workloads.

SQLite search evaluates the newest `max_candidates` entries in the exact compatibility scope. V0.7
does not use an approximate-nearest-neighbor index.

## Compatibility and eligibility

Only the latest human `user` message may vary. A candidate must have the same:

- native provider and configured endpoint;
- credential/account scope, represented only by a one-way identity hash;
- model and generation parameters;
- preceding messages and system/developer context;
- input, compression, and output-control result;
- embedding implementation and dimensions;
- optional caller-supplied task namespace.

Requests are eligible only when they are non-streaming, tool-free JSON requests containing a
non-empty human query. Only status `200`, valid JSON, `application/json` responses are stored.
Malformed bodies, tool definitions, tool calls/results, and mixed non-text human content bypass the
semantic cache.

## Freshness protection

The automatic detector bypasses both semantic and exact caches for queries that appear
freshness-sensitive, including current or latest news, weather, forecasts, prices, exchange rates,
stock data, scores, schedules, availability, and release dates. Relative-time words such as
`today`, `tomorrow`, `now`, and `this week` also trigger the bypass.

Use this local request header when the application knows a task is freshness-sensitive even if the
query does not contain a detected term:

```text
x-optimizer-semantic-freshness: sensitive
```

`auto` and `stable` are accepted, but `stable` never overrides an automatically detected freshness
term. This deliberately prevents callers from accidentally forcing reuse of an obviously stale
answer.

## Request controls and response headers

- `x-optimizer-cache-control: no-store` bypasses exact and semantic reads and writes.
- `x-optimizer-cache-control: refresh` skips reads and replaces eligible entries.
- `x-optimizer-semantic-task: <namespace>` isolates candidates by an application-defined task.
- `x-optimizer-semantic-freshness: auto|stable|sensitive` supplies freshness intent.

Task namespaces must contain 1-128 letters, numbers, dots, colons, slashes, underscores, or hyphens.
Invalid semantic control values return `400 invalid_semantic_cache_control` before a provider call.
All control headers are consumed locally and never forwarded.

A reused response has `x-optimizer-cache: SEMANTIC_HIT` and includes
`x-optimizer-semantic-similarity`. Misses, refreshes, bypasses, disabled caches, and failures use the
same `MISS`, `REFRESH`, `BYPASS`, `DISABLED`, and `ERROR` states as exact caching.

## Storage, telemetry, and invalidation

The semantic database is created with owner-only (`0600`) permissions. It contains cached response
bodies and sparse vectors, but not raw requests, prompts, headers, or credentials. Telemetry records
semantic hit counts and avoided provider cost without copying cached content.

Clear semantic entries even while the feature is disabled:

```bash
optimizer cache clear --kind semantic
```

Use `--kind all` to clear exact and semantic entries. A database or embedding error fails open to a
normal provider call and disables semantic caching for the running process.
