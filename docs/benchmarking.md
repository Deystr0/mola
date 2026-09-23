# Benchmarking

The benchmark runner compares recorded direct and optimized executions without making provider
calls:

```bash
optimizer benchmark benchmarks/results.jsonl
```

Each JSONL row contains `direct` and `optimized` objects with externally measured `success`,
`cost_usd`, `input_tokens`, `output_tokens`, and `latency_ms`. See
[`benchmarks/README.md`](../benchmarks/README.md) for the exact schema.

The report includes total cost, success rate, and cost per successful task alongside tokens and
median latency. Failed optimized tasks therefore cannot be hidden behind lower token usage. The
runner rejects absent quality labels, negative metrics, malformed rows, and non-finite costs.

This repository does not ship fabricated outcome data and normal tests never call paid providers.
Live collection is intentionally operator-controlled and opt-in because success criteria and
credentials are task-specific. Record provider-reported usage and configured prices, use identical
task sets on both sides, disclose retries and escalation, and publish negative results.
