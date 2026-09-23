# Optional local decision model

V0.9 adds shadow inference and V0.10 adds evidence-gated active inference. Both modes use the same
small multiclass linear classifier and deterministic numeric feature vector. The model does not
generate text and does not inspect prompt vocabulary.

## Features

Artifacts must provide weights for exactly these normalized features:

- `log_input_tokens`: `log(1 + estimated_input_tokens)`;
- `message_count`: capped at 100 and divided by 10;
- `tool_count`: capped at 20 and divided by 5;
- `has_media`: zero or one;
- `streaming`: zero or one.

The classifier computes one linear score for every route and converts the six scores to
probabilities with softmax.

## Artifact format

Artifacts are local UTF-8 JSON files with `schema_version: 1`, declared validation evidence, and
exactly six route objects (`cache`, `tool`, `local`, `cheap`, `mid`, `frontier`). Every route object
contains a finite numeric `bias` and a `weights` object containing every feature above. Unknown,
missing, non-finite, or malformed values reject the artifact.

```json
{
  "schema_version": 1,
  "validation": {"samples": 500, "accuracy": 0.91},
  "routes": {
    "cheap": {
      "bias": 0.2,
      "weights": {
        "log_input_tokens": -0.8,
        "message_count": -0.2,
        "tool_count": -1.0,
        "has_media": -2.0,
        "streaming": 0.0
      }
    }
  }
}
```

The abbreviated example shows one route; a loadable artifact must include all six. The gateway
trusts the artifact's evidence metadata, so artifact provenance and validation methodology remain
the operator's responsibility. The project ships no pretrained weights and does not train models.

## Shadow mode

With `mode: shadow`, the rule router always controls production. The model route, confidence, and
agreement with the rule are stored and exposed in response headers. This produces outcome-linked
training/evaluation data without changing provider traffic.

## Active mode gates

With `mode: active`, a prediction controls the model field only when all conditions hold:

- no explicit `x-optimizer-route` override is present;
- artifact validation samples and accuracy meet configured minimums;
- prediction confidence meets `min_confidence`;
- the prediction is not `CACHE` (normal cache lookup already runs first);
- the predicted provider model mapping exists.

Any failed condition falls back to the deterministic rule and records a specific reason. Invalid or
unreadable artifacts disable model inference while leaving rule routing available. There are no
automatic retries or escalations in V0.10.
