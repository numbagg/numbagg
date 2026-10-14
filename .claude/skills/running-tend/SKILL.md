---
name: running-tend
description: numbagg-specific guidance for tend CI workflows. Adds a standing exception for filing issues in other repos, which CI workflow to watch, the command timeout the long benchmark job needs when polling CI, nightly-survey expectations, and dependency management on top of the bundled tend-ci-runner skills. Use when operating in CI.
---

# Running Tend — numbagg

Tend-specific CI guidance. Project conventions are in AGENTS.md.

## Filing issues in other repos

Standing exception granted: file directly in agent-equipped targets (per
**Filing issues** in the bundled `/tend-ci-runner:act-in-other-repos` skill)
without asking permission here first. The default rule (open an issue here asking
permission first) still applies when the target shows no agent signals.

## CI workflows

- **Test** — the main CI workflow (`test.yaml`). Runs tests, linting,
  benchmarks. tend-ci-fix watches this workflow.

## CI polling

The `Test` workflow's `benchmark` job runs ~17 min, so a gated poll here
takes that long; that is expected. Run the bundled
`/tend-ci-runner:monitor-ci` poll once, in the foreground, with a command
timeout above 20 min. The poll has no time limit of its own, so a shorter
command timeout ends it before the benchmark settles and leaves the verdict
unverified.

## Nightly rolling survey

`nightly_survey_files.py` outputs no files on several of its 28
buckets — this repo tracks only a few dozen files, so some daily
buckets have no files assigned. Empty output is expected; treat it
as "no survey today" and
move on to the next step rather than re-running the script or debugging
it.

## Dependency management

Dependencies are managed in `pyproject.toml` with `uv`. The tend-weekly
workflow handles dependency updates.
