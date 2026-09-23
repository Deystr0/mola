# Release checklist

GitHub Releases is the only publishing destination. Package metadata version and tag must agree:
`pyproject.toml` version `X.Y.Z` corresponds to tag `vX.Y.Z`. No tag is created by the build
workflow. See [SECURITY.md](../SECURITY.md) for private vulnerability reporting.

## Before tagging

- [ ] Review and approve the commit on `main`; confirm there are no known critical bugs or
      accidentally committed secrets.
- [ ] Update `CHANGELOG.md`, package version, and `uv.lock`; run `uv lock --check`.
- [ ] Run `venv/bin/ruff format --check .`, `venv/bin/ruff check .`, and `venv/bin/pytest`.
- [ ] Confirm the GitHub security workflow passed its dependency audit, container scan, and SBOM
      job for the exact commit.
- [ ] Build and smoke-test the container in staging using the actual deployment configuration.
- [ ] Confirm rollback image and configuration are available; document any SQLite schema changes.

## Publish

Create and push an annotated `vX.Y.Z` tag from the reviewed `main` commit. The tag workflow checks
the version and lock, runs tests and scans, builds wheel/sdist and the container SBOM, records
GitHub artifact attestations, then creates a GitHub Release with those files. It does not publish
to PyPI or push a container image. Never retag an existing release.

## After publication

- [ ] Verify the release page contains the wheel, sdist, and CycloneDX SBOM, and verify their
      GitHub attestations.
- [ ] Deploy the pinned image in a staging environment, then production under TLS and the edge
      controls described in [deployment.md](deployment.md).
- [ ] Smoke-test `/health/ready`, one non-streaming provider request, one streaming request, and
      rejection of an invalid gateway key. Monitor provider errors, gateway `5xx`, latency, and
      cost/usage records against the pre-deploy baseline.
- [ ] Roll back the deployment to the prior image digest and configuration if authentication is
      bypassed, gateway-attributable `5xx` persists, telemetry is corrupted, or provider requests
      are duplicated unexpectedly. Do not delete the release tag to perform an application rollback.
