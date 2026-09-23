# MOLA - Local LLM Optimizer

Local LLM Optimizer is a private, local-first gateway placed between an application and an LLM
provider. It forwards OpenAI, Anthropic, OpenRouter, and Ollama requests, records metadata-only usage locally, and can
apply conservative input, tool-output, output-budget, caching, and model-routing optimizations when
explicitly enabled.

The primary product metric is **cost per successful task**, not token reduction in isolation.

This is an alpha-stage project. Published tags and assets appear under
[GitHub Releases](https://github.com/Deystr0/mola/releases). See the
[changelog](https://github.com/Deystr0/mola/blob/main/CHANGELOG.md),
[contributing guide](https://github.com/Deystr0/mola/blob/main/CONTRIBUTING.md),
[security policy](https://github.com/Deystr0/mola/blob/main/SECURITY.md), and
[release checklist](https://github.com/Deystr0/mola/blob/main/docs/releasing.md).

## Current scope

- OpenAI-native `POST /v1/chat/completions`
- Anthropic-native `POST /v1/messages`
- OpenRouter-compatible `POST /openrouter/v1/chat/completions`
- Ollama-compatible `POST /ollama/v1/chat/completions`
- Streaming and non-streaming body forwarding without JSON reserialization
- Caller-supplied credentials or environment-variable fallback
- SQLite request, latency, token, status, and configured-cost telemetry
- `optimizer init`, `optimizer serve`, `optimizer doctor`, `optimizer stats`, and
  `optimizer cache clear`
- Opt-in exact deduplication and tool-output line-ending normalization
- Opt-in relevance-based context intelligence with conservative turn-level pruning
- Opt-in deterministic tool-output compression with `lite`, `full`, and `ultra` modes
- Explicit output-budget policies applied before provider generation
- Opt-in, TTL-bound SQLite exact and local semantic caches for compatible non-streaming requests
- Opt-in deterministic `CACHE`, `TOOL`, `LOCAL`, `CHEAP`, `MID`, and `FRONTIER` routing
- Optional local shadow or evidence-gated active decision model
- Deterministic response validation and optional same-provider progressive escalation
- Optional gateway authentication, per-process rate limiting, and runtime resource limits
- Non-root Docker/Compose deployment and metadata-only benchmark comparison
- Immutable system/developer instructions, tool schemas, and current user requests
- No remote telemetry, learned context rewriting, automatic provider switching, or bundled ML weights

## Requirements

- Python 3.11 or newer
- CPU only; no GPU or local model is required

The production Docker image installs the exact dependencies and artifact hashes recorded in
`uv.lock`. Local editable development installs may use the commands below; contributors should
run `uv lock --check` before changing dependencies or preparing a release.

## Install and run

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e '.[dev]'

optimizer init
optimizer doctor
optimizer serve
```

The server binds to `127.0.0.1:4000` by default. Configuration is created at
`~/.config/llm-optimizer/optimizer.yaml`; telemetry is stored at
`~/.local/share/llm-optimizer/telemetry.db`.

Set provider credentials in the process environment:

```bash
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export OPENROUTER_API_KEY='...'
```

Credentials supplied by an SDK are forwarded instead, so existing clients can normally change
only their base URL.

## OpenAI example

```python
from openai import OpenAI

client = OpenAI(api_key="...", base_url="http://127.0.0.1:4000/v1")
response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

## Anthropic example

```python
from anthropic import Anthropic

client = Anthropic(api_key="...", base_url="http://127.0.0.1:4000")
response = client.messages.create(
    model="your-model",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.content[0].text)
```

## Cost configuration

Provider prices change, so the project does not ship guessed or stale prices. Add the models you
use to `optimizer.yaml` as USD per one million tokens:

```yaml
pricing:
  models:
    your-model:
      input_per_million: 1.00
      output_per_million: 4.00
```

Then inspect local totals:

```bash
optimizer stats
```

Unknown models still contribute request and token totals and are clearly reported as unpriced.

## Safe input optimization

Every input optimization is disabled by default. Enable only the deterministic transformations you
want in `optimizer.yaml`:

```yaml
optimization:
  input:
    deduplicate_messages: true
    deduplicate_content_blocks: true
    deduplicate_tool_outputs: true
    normalize_tool_output_line_endings: true
    max_estimated_tokens: 32000
    context:
      enabled: false
      min_recent_turns: 2
      max_prunable_score: 0.35
```

The latest user message, all system/developer messages, top-level tool definitions, and structured
constraints are never modified. The token budget remains audit-only unless V0.11 context
intelligence is explicitly enabled. Each transformation is stored as metadata containing its kind,
JSON path, and estimated token delta—never the removed content.

Responses include `x-optimizer-request-id`, `x-optimizer-events`, and
`x-optimizer-estimated-input-tokens-removed` headers. `optimizer stats` reports the aggregate
estimated reduction separately from provider-reported token usage.

## Context intelligence

V0.11 can remove low-value historical turns when a request exceeds `max_estimated_tokens`. It is
disabled by default and requires an explicit budget:

```yaml
optimization:
  input:
    max_estimated_tokens: 32000
    context:
      enabled: true
      min_recent_turns: 2
      max_prunable_score: 0.35
```

The local deterministic scorer combines relevance to the current request, dependency structure,
recency, uniqueness, and importance. It removes only complete historical turn groups, keeping tool
invocations and results together. A lossless JSON-whitespace pass runs before semantic removal, so
formatting alone cannot cause a historical turn to be dropped. System/developer instructions, the
current user request and its trailing active tool sequence, top-level tool schemas, structured
settings, recent-turn floor, and chunks above the configured score threshold remain intact. If
those protections make the budget
impossible, the gateway forwards the safest remaining request and records `context_budget_unmet`.
See [context intelligence](docs/context-intelligence.md) for the scoring and safety contract.

## Output budget controller

V0.4 never infers task type from prompt text. Enable the controller and configure the ceilings you
want:

```yaml
optimization:
  output:
    enabled: true
    default_policy: null
    openai_parameter: max_completion_tokens
    policies:
      machine_to_machine: {max_tokens: 512}
      tool_call: {max_tokens: 1024}
      code: {max_tokens: 4096}
      analysis: {max_tokens: 4096}
      user_answer: {max_tokens: 1024}
      documentation: {max_tokens: 4096}
```

These values are examples, not project defaults. Select a policy on each request:

```text
x-optimizer-output-policy: user_answer
```

The controller sets a missing limit, caps a higher limit, and preserves a lower limit. It leaves
invalid or ambiguous provider settings untouched and records the decision without storing content.
Use `default_policy` only when every request without a header should share one policy.

## Exact cache

V0.5 can reuse the byte-identical JSON response for an identical compatible request. It is disabled
by default because enabling it stores provider response bodies locally:

```yaml
optimization:
  cache:
    exact:
      enabled: true
      database_path: ~/.local/share/llm-optimizer/cache.db
      ttl_seconds: 3600
```

Only non-streaming, tool-free requests are eligible. Cache keys include the provider endpoint,
credential/account scope, and the exact request bytes after enabled input and output controls. API
keys and prompts are not stored in the cache database. Responses expose `x-optimizer-cache` with
`MISS`, `HIT`, `REFRESH`, `BYPASS`, `DISABLED`, or `ERROR`.

Use `x-optimizer-cache-control: no-store` to bypass both reading and writing, or `refresh` to skip a
read and replace the entry with a successful JSON response. Clear all entries with:

```bash
optimizer cache clear
```

Clear semantic entries separately with `optimizer cache clear --kind semantic`, or clear both
caches with `optimizer cache clear --kind all`.

## Semantic cache

V0.7 can reuse a cached response when the latest human query is sufficiently similar and every
other compatibility input is unchanged. It is opt-in and runs entirely on the local CPU:

```yaml
optimization:
  cache:
    semantic:
      enabled: true
      database_path: ~/.local/share/llm-optimizer/semantic-cache.db
      ttl_seconds: 900
      similarity_threshold: 0.75
      max_candidates: 200
      embedding_dimensions: 1024
```

The built-in embedding is deterministic feature hashing over normalized word and character
features. It requires no model download, GPU, network request, or embedding API. Exact cache lookup
still runs first. Semantic reuse requires the same provider endpoint, credential/account scope,
model and generation settings, conversation context, and optional task namespace. Only the latest
human query may vary.

Freshness-sensitive queries such as current news, weather, prices, scores, schedules, or requests
containing words such as `today` and `latest` bypass both caches whenever semantic caching is
enabled. You can explicitly force the conservative bypass with:

```text
x-optimizer-semantic-freshness: sensitive
```

Use `x-optimizer-semantic-task: support` to isolate responses by a caller-defined task namespace.
A semantic hit returns `x-optimizer-cache: SEMANTIC_HIT` and an
`x-optimizer-semantic-similarity` score. See the detailed safety contract and limitations in
[semantic cache](docs/semantic-cache.md).

## Tool-output compression

V0.6 compresses large, recognized command results already present in an LLM request. It does not
rewrite the provider's generated response. Enable it in `optimizer.yaml`:

```yaml
optimization:
  compression:
    tool_output:
      enabled: true
      mode: ultra
      min_characters: 2000
```

`ultra` is the default mode. Select `lite`, `full`, `ultra`, or the safe `off` bypass per request:

```text
x-optimizer-tool-compression: lite
```

The compressor recognizes Git, grep/ripgrep, pytest, PHPUnit, Jest, npm, pnpm, ESLint, TypeScript,
PHPStan, and Composer command output. It keeps errors, failures, warnings, stack frames, diagnostic
locations, command-specific structure, and head/tail context. It emits explicit omission markers and
uses the original bytes whenever the candidate is not smaller or the command cannot be identified.

Responses report the selected mode and number of changed tool results through
`x-optimizer-tool-compression` and `x-optimizer-tool-outputs-compressed`. Compression events contain
only the mode, command family, omitted-line count, JSON path, and estimated token delta.

## Deterministic model routing

V0.8 selects an explainable route from structural request features. Routing is disabled by default
and never invents model names:

```yaml
optimization:
  routing:
    enabled: true
    default_route: mid
    cheap_max_estimated_tokens: 600
    frontier_min_estimated_tokens: 8000
    frontier_min_messages: 20
    models:
      openai:
        local: local-model
        tool: tool-model
        cheap: cheap-model
        mid: mid-model
        frontier: frontier-model
      anthropic:
        local: null
        tool: null
        cheap: null
        mid: null
        frontier: null
```

Tool-bearing requests select `TOOL`; media, large inputs, and long conversations select
`FRONTIER`; small requests select `CHEAP`; remaining requests use the configured default. Override
the rule for one request with `x-optimizer-route: cache|tool|local|cheap|mid|frontier`. `CACHE` is a
cache-only route: a miss returns locally without incurring a provider call. If a selected model is
not configured, the caller's original model is preserved and the response reports that routing was
not applied.

## Optional decision model

V0.9–V0.10 can load a versioned local JSON artifact containing a multiclass linear model. Shadow
mode records its prediction without changing rule execution. Active mode can control routing only
when its declared validation sample count and accuracy satisfy configured evidence thresholds, its
prediction exceeds the confidence threshold, and the target route has a configured model.
Otherwise the rule router remains in control.

```yaml
optimization:
  routing:
    decision_model:
      enabled: true
      mode: shadow  # change to active only after validation
      artifact_path: ~/.config/llm-optimizer/decision-model.json
      min_confidence: 0.80
      min_validation_samples: 100
      min_validation_accuracy: 0.80
```

Inference uses only numeric structural features, runs on the CPU, and has no network dependency.
The project intentionally ships no guessed weights or trainer. See [routing](docs/routing.md) and
[decision model](docs/decision-model.md) for the complete contracts.

## Validation and progressive escalation

V0.12 validates non-streaming responses without an LLM judge. With routing mappings and escalation
enabled, an invalid cheap response can progress through `cheap → mid → frontier` within the same
provider. Every attempt and its configured cost are recorded; only the final passing response is
cached. Streaming is explicitly bypassed.

```yaml
optimization:
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

See [validation and escalation](docs/validation-escalation.md) for gates, telemetry, and limitations.

## Remote access and rate limiting

Localhost behavior remains unchanged. Any configured non-loopback bind requires gateway
authentication. Clients send a dedicated `x-optimizer-api-key`, while provider credentials continue
to use their native headers.

```yaml
security:
  remote_auth:
    enabled: true
    api_key_env: OPTIMIZER_API_KEY
  rate_limit:
    enabled: true
    requests_per_minute: 60
    burst: 10
```

Rate and concurrency limits are per process. TLS and distributed protection belong at the reverse
proxy or platform boundary. See [security](docs/security.md).

Provider destinations use exact hostname allowlists and reject unsafe URL forms. Custom private
or HTTP endpoints require explicit configuration. Authentication guesses are rate limited before
key verification, and chunked bodies are bounded while being read.

## Docker

```bash
cp .env.example .env
# Set OPTIMIZER_API_KEY and any provider keys.
docker compose up --build -d
```

The image runs non-root; Compose drops capabilities, uses a read-only root filesystem, persists
SQLite under `/data`, and publishes on host loopback by default. See [deployment](docs/deployment.md).

## Benchmarks

Compare recorded direct and optimized results without a provider call:

```bash
optimizer benchmark benchmarks/results.jsonl
```

The report includes quality, cost per successful task, tokens, and median latency. It never invents
success labels. See [benchmarking](docs/benchmarking.md).

## Development

```bash
venv/bin/ruff format --check .
venv/bin/ruff check .
venv/bin/pytest
```

Normal tests use mock providers and never make paid API calls.

GitHub Actions audit Python dependencies, scan the built container, export a CycloneDX SBOM, and
attest tagged release artifacts. The [security and deployment guide](docs/security.md) covers the
NGINX TLS example and the remaining platform controls.

See [architecture](docs/architecture.md), [input optimization](docs/optimization.md),
[context intelligence](docs/context-intelligence.md),
[output budgets](docs/output-budgets.md), [exact cache](docs/exact-cache.md),
[semantic cache](docs/semantic-cache.md), [tool-output compression](docs/tool-output-compression.md),
[routing](docs/routing.md), [decision model](docs/decision-model.md), [privacy](docs/privacy.md), and
[provider behavior](docs/providers.md), [validation and escalation](docs/validation-escalation.md),
[security](docs/security.md), [deployment](docs/deployment.md), and
[benchmarking](docs/benchmarking.md) for the current contracts and limitations.
