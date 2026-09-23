# Exact cache

V0.5 provides an opt-in SQLite cache for byte-identical, compatible requests. It is designed to
avoid a provider call when the same finalized request has already produced a reusable response.

## Configuration

```yaml
optimization:
  cache:
    exact:
      enabled: false
      database_path: ~/.local/share/llm-optimizer/cache.db
      ttl_seconds: 3600
```

The cache is disabled by default. Enabling it stores provider response bodies on local disk. TTL is
measured from the successful cache write; expired entries are deleted during normal reads and
writes.

## Eligibility

A request is eligible only when all of the following are true:

- exact caching is enabled;
- the request is non-streaming;
- the finalized request body is a JSON object;
- `tools`, `functions`, and `tool_choice` are absent;
- the request does not specify `x-optimizer-cache-control: no-store`.

Only status `200` responses with valid JSON and an `application/json` content type are stored.
Provider errors, non-JSON bodies, streams, malformed requests, and tool-bearing requests are never
stored.

The key includes the provider name and endpoint, a one-way hash of the relevant
credential/account/version headers, and the exact request bytes after enabled input optimization and
output-budget control. This prevents reuse across credentials, accounts, provider endpoints, or
different finalized payloads.

## Request controls

The optional local header accepts two values:

- `x-optimizer-cache-control: no-store` bypasses both lookup and storage;
- `x-optimizer-cache-control: refresh` skips lookup and replaces the entry after a cacheable provider
  response.

An unknown value returns `400 invalid_cache_control` before a provider call. The header is never
forwarded upstream.

`x-optimizer-cache` reports `HIT`, `MISS`, `REFRESH`, `BYPASS`, `DISABLED`, or `ERROR`. Cache hits
record zero provider tokens and zero actual cost for that request while preserving the cached usage
as baseline cost, so `optimizer stats` reports the avoided cost.

When semantic caching is enabled, its freshness detector can conservatively suppress exact-cache
reads and writes too. This prevents an identical request for current news, weather, prices, scores,
or schedules from receiving a stale exact-cache response.

## Invalidation and failure behavior

Clear every entry, including entries left behind while the feature is disabled, with:

```bash
optimizer cache clear
```

Use `optimizer cache clear --kind all` to clear both exact and semantic caches.

The database file is created with owner-only permissions. A cache initialization or read/write error
disables caching for the running process and fails open to normal provider traffic. The gateway never
serves partially read data and never treats an error as a cache hit.
