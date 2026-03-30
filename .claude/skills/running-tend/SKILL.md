# Running Tend — numbagg

Tend-specific CI guidance. Project conventions are in CLAUDE.md.

## CI workflows

- **Test** — the main CI workflow (`test.yaml`). Runs tests, linting,
  benchmarks. tend-ci-fix watches this workflow.

## Dependency management

Dependencies are managed in `pyproject.toml` with `uv`. The tend-weekly
workflow handles dependency updates.
