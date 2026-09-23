# Docker and production deployment

The supplied image runs as a non-root user, drops writable application state into `/data`, and has
an unauthenticated readiness healthcheck. Compose additionally drops Linux capabilities, enables
`no-new-privileges`, and makes the root filesystem read-only.
The Dockerfile pins its Python and uv base images by digest and installs production dependencies
from `uv.lock` with `uv sync --locked --no-dev --no-editable`. Rebuild after an intentional lockfile
update; record the resulting image digest for deployment and rollback. A tag workflow publishes
wheel, sdist, checksums, and an SBOM to GitHub Releases, but does not push a container image.

```bash
cp .env.example .env
# Set a long random OPTIMIZER_API_KEY and any provider keys in .env.
docker compose up --build -d
curl http://127.0.0.1:4000/health/ready
```

Send `x-optimizer-api-key` on provider requests. Compose publishes only on host loopback by default,
while the process binds inside the container. Configuration is mounted read-only from
`docker/optimizer.yaml`; telemetry and caches use the `optimizer-data` volume.

The Docker Ollama URL uses `host.docker.internal`, including a Linux `host-gateway` mapping. Change
it to the Ollama service name when both containers share a Compose network.

For a VPS, place a maintained reverse proxy or load balancer in front of the container for TLS,
public rate limiting, request logging policy, and certificate management. The example
[`deploy/nginx/optimizer.conf`](../deploy/nginx/optimizer.conf) includes TLS, HSTS, per-IP request
and connection limits, and streaming response forwarding. Replace `optimizer.example.com`, install
and rotate a valid certificate at the shown paths, then run `nginx -t` before reloading NGINX. Keep
port `4000` bound to host loopback; allow only SSH and ports `80`/`443` through the host firewall,
and check externally that port `4000` is closed.

The NGINX example overwrites `X-Optimizer-Client-IP`. To apply the gateway's authentication quota
per original client, add the exact direct peer IP seen by the gateway to
`security.remote_auth.trusted_proxy_ips` in `docker/optimizer.yaml`. For a local non-container
NGINX-to-gateway connection this is `127.0.0.1`; Docker's bridge gateway address depends on the
host and network. Never trust a broad CIDR or client-supplied forwarding header. NGINX's limit
zone is shared across its workers on one host; multiple edge hosts need an external shared limiter
or managed edge service. Provider egress and volumetric DDoS protection also need platform rules.

Back up `/data` only if the local telemetry/cache retention policy allows it. Cached response
databases may contain model outputs even though telemetry never contains prompts or responses.
