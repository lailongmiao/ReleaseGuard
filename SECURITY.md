# Security Policy

## Supported version

Security fixes are applied to the latest revision on the default branch. Older
revisions and downstream forks are not maintained by this project.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability or exposed secret.
Report it privately through
[GitHub Security Advisories](https://github.com/lailongmiao/ReleaseGuard/security/advisories/new).

Include the affected revision, impact, reproduction steps, and any suggested
mitigation. Do not include live credentials or private repository data.

## Deployment boundaries

ReleaseGuard is published for local development and engineering research. It
has not been certified for production use. Before exposing an instance:

- Generate a unique `JWT_SECRET_KEY` of at least 32 characters.
- Keep model, GitHub, database, and tracing credentials outside Git.
- Disable debug mode and restrict CORS to trusted origins.
- Use separate database credentials and network policies per environment.
- Review whether Langfuse tracing may receive sensitive prompts or source data.
- Keep real-model and real-repository feature flags disabled until access,
  retention, redaction, and budget policies have been reviewed.

The current command path is limited to Dry Run / `SIMULATED` behavior. It must
not be treated as authorization to approve, merge, deploy, or publish changes.
