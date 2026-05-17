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

## CI polling cap

The `Test` workflow's `benchmark` job consistently runs 16–17 min wall, and
the full required-check set on bot-opened PRs that trigger the matrix
(workflow-regen, dependency bumps) finishes in 17–20 min. The bundled
`running-in-ci` polling recipe's default `for i in $(seq 1 15)` (15-min cap)
reliably fires one tick short of completion. **Use `seq 1 22` instead** when
polling such PRs.

The bot's fallback path (single-shot `gh pr checks` after cap fires) is
correct — but it has historically carried bugs in hand-rolled continuation
loops, so avoiding the fallback is preferred. See tracking issue
[#599](https://github.com/numbagg/numbagg/issues/599) for the cumulative
evidence.

## Nightly rolling survey

`nightly-survey-files.sh` outputs empty on roughly 5 of 28 days — this
repo only tracks ~50 files, so several daily buckets have no files
assigned. Empty output is expected; treat it as "no survey today" and
move on to the next step rather than re-running the script or debugging
the shell.

## Dependency management

Dependencies are managed in `pyproject.toml` with `uv`. The tend-weekly
workflow handles dependency updates.
