# Benchmarks

`optimizer benchmark PATH` compares recorded direct and optimized runs without making provider
calls. The input is JSON Lines with one task per line. Store metadata only unless the benchmark data
is intentionally public:

```json
{"id":"task-1","direct":{"success":true,"cost_usd":0.01,"input_tokens":100,"output_tokens":20,"latency_ms":900},"optimized":{"success":true,"cost_usd":0.006,"input_tokens":60,"output_tokens":18,"latency_ms":700}}
```

Every side requires an externally measured `success` boolean, configured-provider cost, input and
output tokens, and latency. The report includes success rate and cost per successful task so lower
token use cannot conceal a quality regression. This runner does not invent success labels and does
not contact a provider. Generate live results with your own opt-in harness and credentials, then
compare them locally.
