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

Benchmark runs ~17 min, exceeding the bundled `running-in-ci` 15-iter
cap. Use an unbounded loop instead — e.g. `until [ "$(pending)" -eq 0 ];
do sleep 60; done` (after defining `pending()` per the bundled recipe) —
so the poll completes rather than aborting at 15 minutes.

The claude-interactive harness blocks long leading `sleep` and
auto-backgrounds the loop, then delivers a single notification when it
exits. That is fine — the dismiss-on-CI-failure follow-up runs in
response to that notification. Earlier guidance to "poll in the
foreground, not `run_in_background: true`" applied to the prior
`claude-code-action` harness and no longer holds; don't fight the
auto-background.

## Nightly rolling survey

`nightly-survey-files.sh` outputs empty on roughly 5 of 28 days — this
repo only tracks ~50 files, so several daily buckets have no files
assigned. Empty output is expected; treat it as "no survey today" and
move on to the next step rather than re-running the script or debugging
the shell.

## Dependency management

Dependencies are managed in `pyproject.toml` with `uv`. The tend-weekly
workflow handles dependency updates.
