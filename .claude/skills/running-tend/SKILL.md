---
name: running-tend
description: numbagg-specific guidance for tend CI workflows. Adds a standing exception for filing issues in other repos, which CI workflow to watch, two-pass polling for the long benchmark job, nightly-survey expectations, and dependency management on top of the generic tend-* skills. Use when operating in CI.
---

# Running Tend — numbagg

Tend-specific CI guidance. Project conventions are in CLAUDE.md.

## Filing issues in other repos

Standing exception granted: file directly in agent-equipped targets (per
**Other Repos** in the bundled `/tend-ci-runner:running-in-ci` skill) without
asking permission here first. The default rule (open an issue here asking
permission first) still applies when the target shows no agent signals.

## CI workflows

- **Test** — the main CI workflow (`test.yaml`). Runs tests, linting,
  benchmarks. tend-ci-fix watches this workflow.

## CI polling takes two passes

The `benchmark` job runs ~17 min, longer than one pass of the bundled
`/tend-ci-runner:running-in-ci` CI Monitoring loop, whose iteration cap is sized to fit
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

`nightly_survey_files.py` outputs no files on several of its 28
buckets — this repo tracks only a few dozen files, so some daily
buckets have no files assigned. Empty output is expected; treat it
as "no survey today" and
move on to the next step rather than re-running the script or debugging
it.

## Dependency management

Dependencies are managed in `pyproject.toml` with `uv`. The tend-weekly
workflow handles dependency updates.

### Don't propose a `ty` pin or a Dependabot `ignore` entry

`ty` bumps pass `lint` again as of 0.0.80. The red streak from 0.0.75 to
0.0.78 was the upstream false positive
[astral-sh/ty#2585](https://github.com/astral-sh/ty/issues/2585) — `ty`
resolves a subscript or member access through a `TypeVar` against a
single member of the TypeVar's bound instead of distributing it over the
union — and it was cleared on the repo side by
[#800](https://github.com/numbagg/numbagg/pull/800), which added the
constrained `FloatArrayT`/`NumericArrayT` TypeVars in `numbagg/utils.py`
and moved `decorators.py`, `funcs.py`, `moving.py` and `moving_exp.py`
onto them. 0.0.73 and 0.0.74 were red for a second bug, `ty`
mis-checking numba's `GUFunc.__call__` stub
([astral-sh/ty#4352](https://github.com/astral-sh/ty/issues/4352)),
which upstream closed on 2026-08-25. #2585 is still open upstream;
numbagg simply no longer writes the annotation that trips it.

So a red `ty` bump is no longer the expected outcome. Review one on its
merits: read the diagnostics before attributing them to #2585, and don't
wave a real regression through as a known upstream bug.

Capping `ty` in `pyproject.toml` or adding a Dependabot `ignore` entry
stays declined, and that decision outlived the failures that prompted it:

- [#755](https://github.com/numbagg/numbagg/pull/755), closed 2026-08-28:
  "We'll let a later Dependabot update re-propose the upgrade once
  upstream is fixed, rather than add local suppressions or weaken the
  annotations."
- [#785](https://github.com/numbagg/numbagg/pull/785), a bot PR capping
  `ty>=0.0.2,<0.0.73`, closed without comment on 2026-09-06.

`pyproject.toml` carries a bare `ty>=0.0.2` and should keep it.
Recommending a cap or an `ignore` entry in a review body is the same
declined proposal in a different place.
