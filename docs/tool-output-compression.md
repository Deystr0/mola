# Tool-output compression

V0.6 reduces large command results before they become input tokens for the next provider call. The
implementation is deterministic, local, dependency-free, and disabled by default.

## Configuration and selection

```yaml
optimization:
  compression:
    tool_output:
      enabled: false
      mode: ultra
      min_characters: 2000
```

The configured mode applies when a request has no override. `ultra` is the default:

- `lite` keeps 64 head lines, 32 tail lines, and three context lines around every signal line;
- `full` keeps 32 head lines, 16 tail lines, and two context lines around every signal line;
- `ultra` keeps eight head lines, eight tail lines, and one context line around every signal line.

Use `x-optimizer-tool-compression: lite`, `full`, or `ultra` to select a mode for one request. Use
`off` for an explicit pass-through. An unknown value returns `400 invalid_tool_compression_mode`
before any provider call. The header is never forwarded upstream.

## Supported inputs

The compressor links a tool result to the preceding OpenAI `tool_calls` or Anthropic `tool_use`
entry. It recognizes these command families from the tool name and structured command arguments:

- Git;
- grep and ripgrep;
- pytest;
- PHPUnit;
- Jest;
- npm and npx;
- pnpm and pnpx;
- ESLint;
- TypeScript `tsc`;
- PHPStan;
- Composer.

Unknown tools, invalid request JSON, unsupported result structures, results below `min_characters`,
and one-line payloads pass through unchanged. A candidate is accepted only when it is smaller than
the original. Existing optimizer omission markers make the transformation idempotent.

## Preserved signal

Every mode retains detected errors, failures, warnings, exception names, tracebacks, non-zero exit
statuses, test assertions, stack frames, compiler diagnostic locations, and command-specific Git
structure. ANSI display sequences and obsolete carriage-return progress frames are removed from a
compressed result. Every omitted run is replaced with a marker containing its exact line count,
mode, and command family.

The compressor is intentionally lossy and V0.6 does not store a recoverable copy. Clients that need
complete output must select `off`, keep compression disabled, or rerun the tool. Runtime failures
fail open to the unchanged request body.

The design follows the same conservative principles documented by
[Caveman](https://github.com/juliusbrussee/caveman): deterministic local processing, explicit
omission markers, preservation of actionable failure signal, and pass-through when a transformation
does not produce a real reduction. This project uses an independent Python implementation and does
not include Caveman's recovery store.
