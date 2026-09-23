# Privacy

The default runtime sends request content only to the provider selected by the route. It does not
send telemetry to the project maintainers or any third party.

The telemetry SQLite database stores only:

- request identifier and timestamp;
- provider and model name;
- completion status and upstream status code;
- latency;
- input and output token counts reported by the provider;
- configured baseline and actual cost;
- cache-hit state and exact/semantic hit type;
- original and optimized input-token estimates;
- optimization kind, JSON path, estimated token delta, and non-content detail;
- selected output policy, provider parameter, action, and numeric limit through audit-event metadata.
- rule/model routes, confidence, model names, and content-free structural routing features.
- context-pruning component scores, message counts, paths, and estimated token deltas.
- validation attempt route, model, status, reason, usage, latency, and configured cost;
- escalation count and cost.

The telemetry schema has no columns for prompts, responses, removed content, API keys, tool output,
repositories, or conversation history. Structured logs contain the same safe request metadata and
never include headers or bodies. The telemetry database is created or corrected to owner-only
(`0600`) permissions.

The exact cache is disabled by default. When enabled, its separate SQLite database stores successful
provider response bodies until their configured expiry or manual invalidation. It does not store
request bodies, prompts, headers, or credential values. Cache keys are one-way hashes derived from
the exact final request bytes plus a hashed provider credential/account scope. The cache database is
created with owner-only (`0600`) permissions. Anyone who can read that database can read cached
responses, so place it only on trusted local storage and run `optimizer cache clear` when needed.

The semantic cache is also disabled by default and uses a separate owner-only SQLite database. It
stores successful provider response bodies plus sparse, locally computed feature vectors. It does
not store raw prompts, request bodies, headers, or credentials; compatibility and entry identifiers
are one-way hashes. Embeddings can still reveal limited information about their source text through
inference, and cached responses remain readable to anyone who can read the database. Keep the file
on trusted local storage and clear it with `optimizer cache clear --kind semantic` when needed. No
embedding content or cache body is copied into telemetry.

Routing telemetry stores estimated input size, message count, tool count, media/stream flags,
selected and executed routes, configured model names, shadow predictions, confidence, and fallback
reason. It never stores prompt text or feature values derived from prompt vocabulary. Optional
decision-model artifacts are read only from the configured local path and are never uploaded.

Tool-output compression is disabled by default and does not add storage. When enabled, it changes
recognized tool-result text in memory before forwarding the request to the selected provider. The
original tool output is not written to telemetry or a recovery database. Audit events store only the
JSON path, compression mode, command family, omitted-line count, and estimated token delta. Because
V0.6 has no recovery store, use `lite` or the per-request `off` mode when full output may be needed.

Context intelligence is disabled by default and does not add a content store. When enabled, it
scores historical turns in memory and forwards a request with selected low-scoring turns removed.
Telemetry stores the five numeric component scores and affected message indexes, but never stores
the scored or removed text. Scores are coarse derived metadata and can disclose limited structural
information, so disable telemetry if even that local metadata is outside your privacy requirements.

Provider credentials are read from the incoming request or from the configured environment-variable
name. Configuration stores the name, never the credential value.

The optional gateway API key is also read only from its configured environment variable. It is not
stored in telemetry or forwarded to providers. The in-memory rate limiter uses only a SHA-256
principal hash, which is discarded when the process exits.

The server binds to `127.0.0.1` by default. Non-loopback configured binds require remote
authentication. TLS and distributed denial-of-service protection remain deployment responsibilities.
