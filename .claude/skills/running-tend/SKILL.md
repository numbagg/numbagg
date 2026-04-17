# Running Tend — numbagg

Tend-specific CI guidance. Project conventions are in CLAUDE.md.

## CI workflows

- **Test** — the main CI workflow (`test.yaml`). Runs tests, linting,
  benchmarks. tend-ci-fix watches this workflow.

## Nightly rolling survey

`nightly-survey-files.sh` outputs empty on roughly 5 of 28 days — this
repo only tracks ~50 files, so several daily buckets have no files
assigned. Empty output is expected; treat it as "no survey today" and
move on to the next step rather than re-running the script or debugging
the shell.

## Dependency management

Dependencies are managed in `pyproject.toml` with `uv`. The tend-weekly
workflow handles dependency updates.
