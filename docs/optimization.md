# Input optimization

Input optimization contains only opt-in, deterministic transformations. All switches default to
false.

## Immutable input

The optimizer never changes:

- system messages;
- developer messages;
- the latest user message;
- top-level tool definitions or schemas;
- model, output, streaming, and structured-response settings;
- unsupported or malformed request shapes.

## Available transformations

### Duplicate messages

`deduplicate_messages` removes a later message only when it is adjacent, byte-equivalent after JSON
parsing, contains plain text, belongs to older conversation history, and has no tool-call structure.

### Repeated context blocks

`deduplicate_content_blocks` removes later exact duplicate text blocks within one older message. Tool
blocks and the latest user message are excluded.

### Redundant tool outputs

`deduplicate_tool_outputs` removes only adjacent, exactly equal tool results. OpenAI tool messages and
Anthropic `tool_result` blocks are supported. Non-adjacent or changed results are retained.

### Tool-output line endings

`normalize_tool_output_line_endings` changes CRLF to LF only inside older tool output. Lone carriage
returns, ordinary conversation text, and current tool results are retained.

### Token budget

`max_estimated_tokens` uses a deterministic four-bytes-per-token approximation. It remains an
audit-only guard by default: an exceeded budget records `token_budget_exceeded` with `action=none`
and forwards the full request.

When `context.enabled` is true, V0.11 may remove complete low-scoring historical turns until the
budget is met. It never cuts content inside a message. See [context intelligence](context-intelligence.md).

## Audit records

Each event records:

- request ID and event sequence;
- transformation kind;
- JSON path;
- estimated tokens before and after;
- a content-free explanation.

Raw removed content is never persisted. Aggregate estimated input-token reduction is visible through
`optimizer stats`. Provider-reported usage remains the authoritative billing signal.
