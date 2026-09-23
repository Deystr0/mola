# Context intelligence

V0.11 provides deterministic, local, relevance-based context pruning. It is disabled by default,
makes no model or provider calls, and requires `max_estimated_tokens` when enabled.

## Configuration

```yaml
optimization:
  input:
    max_estimated_tokens: 32000
    context:
      enabled: true
      min_recent_turns: 2
      max_prunable_score: 0.35
```

`min_recent_turns` protects the newest historical turn groups regardless of score.
`max_prunable_score` is inclusive and ranges from `0` to `1`. Raising it permits increasingly
important context to be removed; the conservative default is `0.35`.

## Chunking and scoring

A chunk is one complete historical human turn: its user message plus following assistant and tool
messages up to the next human turn. Anthropic user-role messages containing only `tool_result`
blocks remain part of the preceding turn. This prevents tool invocations and results from being
separated. Unknown message roles cause the pruning pass to fail safe without removing anything.

Each chunk receives five normalized component scores:

- relevance: current-request token coverage found in the chunk;
- dependency: conversational pairing and tool-call/result structure;
- recency: position among historical chunks;
- uniqueness: token features not present in another historical chunk;
- importance: explicit constraint, error, security, code, file, and tool signals.

The fixed score is `0.40 relevance + 0.15 dependency + 0.15 recency + 0.10 uniqueness + 0.20 importance`.
Eligible chunks are removed from lowest score upward, with original order as the deterministic
tie-breaker. Before deleting a chunk, the optimizer removes JSON formatting whitespace and records
`request_json_compacted`; if that lossless representation fits the budget, no history is removed.
Pruning stops as soon as the approximate request budget is met.

## Immutable context

The pass never removes or edits:

- system or developer messages;
- the current human request;
- assistant or tool messages trailing that request;
- the configured recent-turn floor;
- chunks scoring above `max_prunable_score`;
- top-level tool definitions, schemas, model settings, output settings, or structured constraints;
- malformed requests or requests with unsupported message roles.

Existing exact-deduplication and line-ending transforms run first. Context intelligence then runs
before tool-output compression, routing, and exact/semantic cache-key generation.

## Observability and conservative failure

Every removed group emits `context_chunk_pruned` with original message indexes, the five scores,
message count, and estimated token delta. Lossless JSON compaction is separately observable, and
removed content is never included. If the budget cannot be met without crossing a safety boundary,
the request is forwarded with the protected context intact and `context_budget_unmet` records
`action=conservative_stop` plus a content-free reason such as `recent_turn_floor`,
`score_threshold`, or `unsupported_shape_or_request`.

The budget uses the same deterministic four-bytes-per-token approximation as the rest of the input
optimizer. Provider-reported usage remains authoritative. V0.11 does not summarize, rewrite, embed,
or recover removed context, and it does not guarantee that a protected request can fit a provider's
actual model-specific context window.
