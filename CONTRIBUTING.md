# Contributing

Contributions are welcome through GitHub issues and pull requests. For a behavioral change, open
an issue first so the API contract and test cases can be agreed before implementation. Security
issues belong in the private channel described in [SECURITY.md](SECURITY.md).

## Local setup

Use Python 3.11 or newer. From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e '.[dev]'
venv/bin/ruff format --check .
venv/bin/ruff check .
venv/bin/pytest
```

No provider API key is required for the tests; provider traffic is mocked. Add focused tests for
any behavior change and preserve the default pass-through behavior when optimizations are disabled.
Do not commit `.env`, `venv/`, databases, prompts, responses, or the private project handoff.

## Dependency and image changes

The production image uses `uv.lock` and a pinned Python/uv image digest. Install uv, run `uv lock`
after changing `pyproject.toml`, and verify with `uv lock --check`. Review the lockfile diff,
including transitive version changes. Build the image with `docker build -t mola:local .`; CI also
audits dependencies, scans the image, and generates a software bill of materials.

## Pull requests

Explain the user-visible behavior, add or update tests and docs, and include the three local check
results above. Keep unrelated changes separate. Do not create a release tag in a pull request;
follow [docs/releasing.md](docs/releasing.md) for a release.
