# Output budgets

V0.4 controls output cost before generation by setting or capping the provider's output-token limit.
It never generates a large response and compresses it afterward.

## Policies

The supported policy names are:

- `machine_to_machine`
- `tool_call`
- `code`
- `analysis`
- `user_answer`
- `documentation`

The project supplies no default token counts because appropriate limits depend on the application and
model. Each configured policy is independently adjustable.

## Selection

Requests select a policy with:

```text
x-optimizer-output-policy: POLICY_NAME
```

When the header is absent, `default_policy` is used if configured. The controller does not inspect
prompts or guess task type. Unknown explicit policies return `400 invalid_output_policy` without
calling the provider.

## Provider behavior

For OpenAI requests, an existing `max_completion_tokens` or `max_tokens` field is respected. When
neither exists, `openai_parameter` selects the field to add. If both fields exist, the request is
considered ambiguous and remains untouched.

Anthropic requests always use `max_tokens`.

For either provider:

- a missing limit is set to the policy ceiling;
- a larger valid limit is capped;
- an equal or smaller valid limit is preserved;
- null, boolean, string, zero, negative, or ambiguous limits are left untouched.

## Observability

The selected policy and ceiling are returned in `x-optimizer-output-policy` and
`x-optimizer-output-limit`. The local `x-optimizer-output-policy` request header is stripped before
forwarding.

Every decision is stored in the metadata-only optimization event ledger. Output-limit values occupy
the event's numeric before/after fields; raw request and response content is never stored. Actual
provider-reported output tokens remain the billing signal.
