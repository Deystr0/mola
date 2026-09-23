# Provider contracts

## Gateway routes

- OpenAI: `POST /v1/chat/completions`
- Anthropic: `POST /v1/messages`
- OpenRouter: `POST /openrouter/v1/chat/completions`
- Ollama: `POST /ollama/v1/chat/completions`

OpenRouter and Ollama use the OpenAI-compatible request/response protocol. OpenRouter forwards only
the allowlisted `Authorization`, `HTTP-Referer`, `X-Title`, `Accept`, and idempotency headers. Ollama
does not require a credential; if `OLLAMA_API_KEY` or an incoming provider `Authorization` value is
present it is forwarded for compatible secured deployments.

Each provider's shipped hostname is allowed by default. To use a custom compatible endpoint,
include its exact hostname in `allowed_hosts`. HTTP and private networks need explicit opt-ins:

```yaml
providers:
  openai:
    base_url: http://127.0.0.1:8000/v1
    api_key_env: LOCAL_COMPATIBLE_KEY
    allowed_hosts: [127.0.0.1]
    allow_private_network: true
    allow_insecure_http: true
```

These opt-ins authorize only the configured destination; client requests cannot supply an
upstream URL. Link-local IP targets, including cloud metadata addresses, are always rejected.

## OpenAI

- Local route: `POST /v1/chat/completions`
- Upstream route: `{openai.base_url}/chat/completions`
- Credential: incoming `Authorization` header, then the configured environment variable
- Forwarded optional headers: `OpenAI-Organization`, `OpenAI-Project`, `Idempotency-Key`

## Anthropic

- Local route: `POST /v1/messages`
- Upstream route: `{anthropic.base_url}/messages`
- Credential: incoming `x-api-key` header, then the configured environment variable
- Forwarded optional headers: `anthropic-version`, `anthropic-beta`
- Default version when absent: `2023-06-01`

Only response headers with defined provider semantics are passed back. Hop-by-hop headers, cookies,
and upstream server metadata are dropped.

Streaming usage is recorded only when the provider includes usage in its server-sent events. The
gateway does not modify requests to force usage reporting because the current contract is transparent
forwarding.

With all optimization switches disabled, request bodies are forwarded unchanged. Enabled input
switches apply only the deterministic transformations documented in `optimization.md`; enabled
output policies may set or cap the documented provider token-limit field.

`x-optimizer-output-policy` is a local control header and is never forwarded upstream. For OpenAI,
the controller respects the limit field already present and otherwise uses the configured
`openai_parameter`. For Anthropic it uses `max_tokens`.

`x-optimizer-cache-control`, `x-optimizer-semantic-freshness`, and
`x-optimizer-semantic-task` are local and never forwarded. Exact and semantic cache identity varies
by provider endpoint and the relevant credential/account/version headers. Raw header values are
never stored.

`x-optimizer-tool-compression` is a local control header and is never forwarded. The compressor
supports OpenAI `role: tool` messages and Anthropic `tool_result` blocks without changing either
provider's protocol shape.

`x-optimizer-route` is local and never forwarded. Routing can replace only the payload's `model`
field; it never translates provider protocols or credentials. `LOCAL` means a caller-configured
model reachable through that route's existing provider adapter. V0.10 does not switch endpoints or
create an in-process text-generation runtime.
