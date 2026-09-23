# Deterministic routing

V0.8 adds an opt-in rule router with six execution labels: `CACHE`, `TOOL`, `LOCAL`, `CHEAP`, `MID`,
and `FRONTIER`. The router runs after input/tool-output optimization and before output control and
cache identity generation, so a routed model is part of the cache key and cost accounting.

## Rules

Without an explicit override, rules are evaluated in this order:

1. A request containing tool definitions, calls, or results selects `TOOL`.
2. A request containing image, document, or audio blocks selects `FRONTIER`.
3. Input at or above `frontier_min_estimated_tokens` selects `FRONTIER`.
4. A conversation at or above `frontier_min_messages` selects `FRONTIER`.
5. Input at or below `cheap_max_estimated_tokens` with at most four messages selects `CHEAP`.
6. Remaining requests select `default_route` (`MID` by default).

These rules inspect structure and counts, not prompt meaning. They are deterministic for identical
finalized input. `LOCAL` is explicit-only because structural heuristics cannot safely decide that a
task is locally answerable.

## Model mappings and safety

Each provider has optional `local`, `tool`, `cheap`, `mid`, and `frontier` model names. A selected
mapping replaces only the JSON `model` field. If it is absent, malformed input prevents rewriting,
or routing is disabled, the caller's payload is preserved. The gateway never invents a provider
model name.

`LOCAL` identifies a configured model reachable through the current provider adapter. It does not
claim that a cloud endpoint is local and does not add a text-generation runtime. Point an
OpenAI-compatible provider configuration at a trusted local endpoint when a genuinely local model
is desired.

## Request and response controls

Use one local header to override automatic rules:

```text
x-optimizer-route: cache|tool|local|cheap|mid|frontier
```

An invalid value returns `400 invalid_route` before a provider call. The header is never forwarded.
Explicit overrides win over the active decision model.

`CACHE` performs the normal exact/semantic lookups using the model that the automatic rule would
have selected. A hit is returned normally. A miss returns `409 cache_miss` without calling the
provider; a lookup failure returns `503 cache_unavailable`.

Responses can include:

- `x-optimizer-route`: executed route, or selected route when nothing executed;
- `x-optimizer-route-decision`: pre-execution decision;
- `x-optimizer-route-source`: `heuristic`, `explicit`, or `decision_model`;
- `x-optimizer-route-applied`: whether a cache or configured model route executed;
- `x-optimizer-routed-model`: effective request model.

## Training data

Enabled routing writes a `routing_decisions` row joined one-to-one with request telemetry. It stores
the rule, selected and executed routes, source/reason, original and routed model names, estimated
input tokens, message/tool counts, media/stream flags, model prediction/confidence/mode, agreement,
application state, and fallback reason. Prompt text and request bodies are excluded.
