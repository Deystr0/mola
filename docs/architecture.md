# Architecture

## V1.0 request path

```text
client
  -> localhost FastAPI route
  -> optional deterministic input optimizer and context intelligence
  -> optional deterministic tool-output compressor
  -> optional rule router plus shadow/evidence-gated active decision model
  -> optional explicit output-budget controller
  -> optional exact-cache lookup
  -> optional local embedding and semantic-cache lookup
  -> provider adapter -> OpenAI, Anthropic, OpenRouter, or Ollama (cache miss/bypass only)
  -> deterministic response validator
  -> optional same-provider cheap -> mid -> frontier escalation
  -> optional exact-cache and semantic-cache writes
  -> response forwarding without content transformation
  -> metadata-only telemetry record
```

The gateway has two native protocol routes. It does not translate an OpenAI payload into an
Anthropic payload or the reverse. With input optimization disabled—the default—the original request
body is forwarded unchanged. Enabled V0.3 transformations operate on supported message structures
without changing provider protocols.

`optimizer.providers.Provider` is the boundary between HTTP ingress and provider-specific
authentication, headers, and upstream paths. `TelemetryStore` owns the local request ledger.
`CostCalculator` consumes replaceable model prices from configuration. These boundaries support
later providers and storage implementations without a plugin loader in the initial release.

`InputOptimizer` owns immutable-context classification and deterministic transformations. Its result
contains the forwarded body, request-level token estimates, and content-free audit events. A runtime
optimizer error fails open to the original request body. V0.11 adds a local deterministic scorer
that may remove only complete historical turns after a configured budget is exceeded. Scoring runs
before compression, routing, and cache-key construction.

`OutputBudgetController` selects only explicitly requested or configured policies. It modifies the
provider's pre-generation token limit without inspecting prompt content. A runtime controller error
fails open; an unknown explicit policy returns a local `400` before any provider call.

`ToolOutputCompressor` examines only OpenAI tool messages and Anthropic `tool_result` blocks. It
links each result to its preceding invocation, recognizes a bounded set of command families, and
passes through unknown, short, malformed, already-compressed, or non-shrinking inputs. It never
modifies human, system, developer, assistant, or tool-schema content.

`ExactCacheBackend` is the storage boundary for V0.5. `SQLiteExactCache` stores eligible successful
JSON response bodies in a database separate from metadata-only telemetry. Keys cover provider
endpoint, credential/account scope, and the exact final request bytes. Streaming and tool-bearing
requests bypass the cache.

`SemanticCacheBackend` is the V0.7 similarity-search boundary. `SQLiteSemanticCache` stores sparse
local embeddings and successful JSON response bodies separately from telemetry and the exact cache.
`HashingEmbeddingBackend` creates deterministic CPU-only vectors without a model download or
external call. Semantic candidates share an exact compatibility key; only their latest human query
varies. The exact cache always has lookup priority.

`RuleRouter` is the V0.8 production decision boundary. It uses structural, content-free features and
explicit per-provider model mappings. `DecisionRouter` adds V0.9 shadow predictions and V0.10 active
selection. `LocalDecisionModel` loads a versioned JSON linear classifier, runs softmax inference on
the CPU, and cannot become active unless validation evidence, confidence, explicit-override, and
target-availability gates all pass. Cache hits are recorded as the executed `CACHE` route regardless
of the model route selected before lookup.

`ResponseValidator` is the V0.12 quality gate. It validates protocol structure, non-empty output,
truncation, refusal, and requested JSON mode without another model. Progressive escalation reuses
the provider adapter and explicit route mappings; it never switches providers. Security middleware
owns gateway authentication, per-process rate limiting, request size, and active-request limits.

## Failure behavior

- Missing credentials produce a local `401` response.
- Upstream connection failures produce `502`.
- Upstream timeouts produce `504`.
- Provider HTTP errors and bodies pass through with their original status.
- Streaming chunks pass through without content transformation; a stream transport failure terminates the response and
  records a failed stream.
- Unavailable telemetry fails open: provider traffic continues and the gateway emits a safe warning.
- Unavailable exact-cache storage fails open: caching is disabled for the process and provider
  traffic continues.
- Unavailable semantic-cache storage or embedding generation fails open: semantic caching is
  disabled for the process and provider traffic continues.
- Missing route mappings preserve the caller model. Invalid explicit routes return local `400`.
  Cache-only misses return local `409` and never call a provider.
- Invalid decision-model artifacts disable model inference. Low confidence, insufficient validation
  evidence, explicit routes, cache predictions, and unavailable targets fall back to rule routing.
- An unexpected tool-compression failure passes through the unmodified post-input-optimization
  body. An unknown explicit mode returns local `400 invalid_tool_compression_mode`.

The gateway does not retry upstream transport failures, timeouts, or provider HTTP errors. When
response validation and progressive escalation are both explicitly enabled, a failed validation
can trigger another billable generation request to a higher configured route of the **same**
provider, up to the configured attempt limit. Streaming requests are not escalated. Keep
escalation disabled for workflows where duplicate generation or downstream side effects would be
unsafe. There is no automatic cross-provider retry or failover.

## Explicit non-goals

- Learned context scoring or automatic summarization
- Automatic provider switching
- Prompt-semantic task classification
- Stop-sequence or response-format rewriting
- Decision-model training or bundled weights
- Remote telemetry
- Automatic cross-provider failover
