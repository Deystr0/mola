# Security policy

## Supported versions

This project has not published a stable 1.0 release. Security fixes target the current `main`
branch and the latest tagged pre-1.0 release when a patch is practical. Older pre-1.0 tags are not
guaranteed support. Check [releases](https://github.com/Deystr0/mola/releases) for the latest tag.

## Report a vulnerability

Use [GitHub's private vulnerability reporting](https://github.com/Deystr0/mola/security/advisories)
and select **Report a vulnerability**. Include affected version or commit, a minimal reproduction,
impact, and a safe contact method. Do not post exploit details in a public issue or pull request.
Maintainers will triage the report privately and coordinate disclosure through a security advisory.
There is no promised response-time SLA.

For deployment-specific risks and controls, see [docs/security.md](docs/security.md). Never include
API keys, prompts, cached responses, or telemetry databases in a report unless explicitly requested
through a secure channel.
