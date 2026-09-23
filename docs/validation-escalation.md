# Validation and progressive escalation

V0.12 adds deterministic response validation and optional same-provider model escalation. Both are
disabled by default and make no judge-model calls.

```yaml
optimization:
  routing:
    enabled: true
    models:
      openai:
        cheap: cheap-model
        mid: mid-model
        frontier: frontier-model
  validation:
    enabled: true
    require_non_empty: true
    reject_truncated: true
    reject_refusal: true
    validate_json_mode: true
    escalation:
      enabled: true
      max_attempts: 3
```

For OpenAI-compatible responses, validation checks HTTP success, JSON shape, a non-empty assistant
message or tool call, truncation, refusal, and valid JSON content when the request selected JSON
mode. Anthropic validation accepts non-empty text or `tool_use` blocks and rejects `max_tokens`
truncation. It does not infer factual correctness.

When the selected, configured route is `cheap` or `mid`, a failed successful-HTTP response can move
through `cheap → mid → frontier`. Missing model mappings are skipped. Explicit provider errors,
transport failures, cache-only routes, tool/local/frontier routes, and streaming requests do not
trigger another call. Escalation never changes provider or endpoint.

Only the final passing response is written to cache. Validation-enabled cache hits are rechecked;
an invalid exact entry is deleted and an invalid semantic candidate is bypassed. If the final tier
still fails, its response is returned with `x-optimizer-validation: fail` and telemetry status
`validation_failed`.

Responses expose validation state, reason, escalation count, and priced escalation cost. SQLite
stores one content-free row per attempt: route, model, status, validation reason, usage, configured
cost, and latency. Request totals include every paid attempt; `escalation_cost_usd` counts the failed
attempts before the final response.

Limitations: streaming is marked `bypass`; semantic entries rejected by validation remain until TTL
expiry; deterministic validation detects structural failure, not truth, style quality, or task
correctness.
