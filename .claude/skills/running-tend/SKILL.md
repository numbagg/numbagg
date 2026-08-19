---
name: running-tend
description: numbagg-specific guidance for tend CI workflows. Adds which CI workflow to watch, two-pass polling for the long benchmark job, nightly-survey expectations, and dependency management on top of the generic tend-* skills. Use when operating in CI.
---

# Running Tend — numbagg

Tend-specific CI guidance. Project conventions are in CLAUDE.md.

## Filing issues in other repos

Standing exception granted: file directly in agent-equipped targets (per
**Filing Issues in Other Repos** in the bundled `running-in-ci` skill) without
asking permission here first. The default rule (open an issue here asking
permission first) still applies when the target shows no agent signals.

## CI workflows

- **Test** — the main CI workflow (`test.yaml`). Runs tests, linting,
  benchmarks. tend-ci-fix watches this workflow.

## CI polling takes two passes

The `benchmark` job runs ~17 min, longer than one pass of the bundled
`running-in-ci` CI Monitoring loop, whose iteration cap is sized to fit
the harness's 10-min Bash maximum. That is expected here: run the
bundled loop as written, and when the first pass reports checks still
running, simply invoke it again. Two passes normally cover the benchmark.

Do **not** substitute an unbounded `while :; do …; done` — it cannot
finish inside the 10-min Bash cap, so the harness kills it mid-poll with
exit 143 and the poll has to be restarted anyway
([30789131037](https://github.com/numbagg/numbagg/actions/runs/30789131037)).
See [#599](https://github.com/numbagg/numbagg/issues/599) and
[#614](https://github.com/numbagg/numbagg/pull/614) for the original,
now-superseded rationale for the unbounded loop.

Keep the loop in the **foreground** (no `run_in_background: true`): a
backgrounded poll is killed when the run's turn ends (~1–2 min), long
before the benchmark finishes, so the dismiss-on-CI-failure follow-up
never runs.

## Nightly rolling survey

`nightly-survey-files.sh` outputs empty on roughly 5 of 28 days — this
repo only tracks ~50 files, so several daily buckets have no files
assigned. Empty output is expected; treat it as "no survey today" and
move on to the next step rather than re-running the script or debugging
the shell.

## Dependency management

Dependencies are managed in `pyproject.toml` with `uv`. The tend-weekly
workflow handles dependency updates.
