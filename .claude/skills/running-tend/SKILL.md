---
name: running-tend
description: numbagg-specific guidance for tend CI workflows. Adds a standing exception for filing issues in other repos, which CI workflow to watch, two-pass polling for the long benchmark job, nightly-survey expectations, and dependency management on top of the generic tend-* skills. Use when operating in CI.
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

### The red `ty` bumps are a settled maintainer decision — don't propose a pin

Every `ty` release after 0.0.72 fails the `lint` job against unchanged
numbagg source. From 0.0.75 on it is a single upstream bug: `ty` rejects
indexing and member access through a `TypeVar` whose bound is a union —
`FloatArray`/`NumericArray` are declared in `numbagg/utils.py`, and the
`TypeVar`s bound to them live in `funcs.py`, `moving.py`, `moving_exp.py`
and `decorators.py` ([astral-sh/ty#2585](https://github.com/astral-sh/ty/issues/2585),
open since January). 0.0.73 is red for a separate bug in numba's
`GUFunc.__call__` stub ([astral-sh/ty#4352](https://github.com/astral-sh/ty/issues/4352),
named in #755's closing comment, closed upstream 2026-08-25 and gone
from 0.0.75 on), and 0.0.74 hits both. So each Dependabot `ty` bump arrives
red, and the obvious-looking remedies — a `<0.0.73` cap in `pyproject.toml`,
a Dependabot `ignore` entry, or widening the `TypeVar` bounds — have already
been declined here:

- [#755](https://github.com/numbagg/numbagg/pull/755), closed 2026-08-28 with
  the reasoning stated: "We'll let a later Dependabot update re-propose the
  upgrade once upstream is fixed, rather than add local suppressions or
  weaken the annotations."
- [#785](https://github.com/numbagg/numbagg/pull/785), a bot PR capping
  `ty>=0.0.2,<0.0.73`, closed without comment on 2026-09-06 — after a bot
  review had already flagged that the cap removes the Dependabot
  re-proposal #755 was counting on, and the author-side run kept it anyway.

Handle a red `ty` bump as a review, not as a problem to solve: confirm the
failure is one of these upstream bugs rather than a real regression in the
diff, say so, withhold approval, and leave the PR for the maintainer.
Recommending a cap or an `ignore` entry in a review body is the same declined
proposal in a different place — report the status and stop there.

This is a record of a decision, not a rule of its own: a maintainer
instruction supersedes it, and it lapses on its own once #2585 is fixed and
the next bump goes green.
