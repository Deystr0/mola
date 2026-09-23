# Remote access and production security

The default remains loopback-only without gateway authentication. A non-loopback configured bind is
rejected unless remote authentication is enabled.

```yaml
server:
  host: 0.0.0.0
security:
  remote_auth:
    enabled: true
    api_key_env: OPTIMIZER_API_KEY
    allow_health_unauthenticated: true
    trusted_proxy_ips: []
  rate_limit:
    enabled: true
    requests_per_minute: 60
    burst: 10
    authentication_requests_per_minute: 30
    authentication_burst: 5
  limits:
    max_request_bytes: 10485760
    max_concurrent_requests: 100
    concurrency_wait_seconds: 0.25
```

Clients authenticate to the gateway with `x-optimizer-api-key`. The dedicated header is never
forwarded upstream, leaving `Authorization` and `x-api-key` available for provider credentials. The
gateway secret is read only from the configured environment variable and compared in constant time.
Authentication attempts consume a separate per-peer token bucket before the key comparison,
including missing and incorrect keys. This protection remains active when the normal request quota
is disabled. The response is `429` with `Retry-After` after the authentication burst is exhausted.

Rate limiting uses in-memory token buckets keyed by one-way hashes of peer address and credential
scope. The proxy address header `X-Optimizer-Client-IP` is ignored unless the direct socket peer is
listed in `trusted_proxy_ips` as an exact IP address. The proxy must overwrite this header on every
request. Uvicorn's generic forwarded-header trust is disabled. Limits and concurrency are per
process. For multiple replicas, deploy a shared limiter at a trusted edge or keep one gateway
instance. The gateway reads request bodies incrementally and stops at `max_request_bytes` even if
`Content-Length` is absent or false. Oversized requests return `413`, exhausted buckets return
`429`, and saturated concurrency returns `503` before a provider call.

Provider destinations are configured, never selected from request URLs. Built-in hosts are
allowlisted per provider; a custom host must be listed in that provider's `allowed_hosts`. The
configured URL must use HTTP(S), have a normal hostname and port, and contain no credentials,
query, fragment, or parent path segment. Remote HTTP and private addresses require separate
`allow_insecure_http` and `allow_private_network` opt-ins. Link-local, multicast, and unspecified
IP targets are always rejected. Ollama's shipped local endpoint opts into HTTP and private
network access. Redirect following is disabled in the HTTP client. This protects the default
configuration and request path; an operator who explicitly allowlists an internal DNS name is
authorizing that destination. DNS rebinding or host-level egress policy needs network enforcement
outside this process.

Telemetry and cache databases are owner-only (`0600`); initialization corrects the permissions of
an existing telemetry file. Run the gateway with a dedicated OS account and protect its data
directory and backups as well.

`/health/live` reports process liveness. `/health/ready` also checks telemetry storage and returns
`503` on failure. Health endpoints can be public for orchestrator checks; set
`allow_health_unauthenticated: false` to protect them.

The sample [NGINX configuration](../deploy/nginx/optimizer.conf) terminates TLS, limits request
rate and concurrent connections per client IP, and forwards only to the loopback gateway. It is a
single-edge example. Certificate issuance/rotation, host firewall rules, a shared limiter across
edge replicas, upstream denial-of-service mitigation, and secret distribution must be configured
on the actual deployment platform. Do not expose the gateway's plain HTTP port to the internet.

The [security workflow](../.github/workflows/security.yml) checks `uv.lock`, audits Python
dependencies, and scans
the built image for high and critical vulnerabilities, generates a CycloneDX container SBOM, and
runs tests across supported Python versions. [Dependabot](../.github/dependabot.yml) updates uv,
Docker, and GitHub Actions dependencies. Tagged artifacts are signed with GitHub build provenance
by the [release workflow](../.github/workflows/release-attestation.yml), then published to GitHub
Releases. These workflows start only after the repository is pushed to GitHub; a release tag must
match `pyproject.toml` and refer to a commit on `main`.
