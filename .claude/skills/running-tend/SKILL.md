# Running Tend — numbagg

Tend-specific CI guidance. Project conventions are in CLAUDE.md.

## CI workflows

- **Test** — the main CI workflow (`test.yaml`). Runs tests, linting,
  benchmarks. tend-ci-fix watches this workflow.

## Dependency management

Dependencies are managed in `pyproject.toml` with `uv`. The tend-weekly
workflow handles dependency updates.

## Editing files under `.claude/skills/` or `.github/workflows/`

The generated tend workflows don't grant `workflows: write`, so the
claude-code-action sandbox mounts `.github/workflows/` and `.claude/skills/`
as read-only. Direct writes fail with `Read-only file system` or permission
errors — this affects nightly Step 5 (`uvx tend@latest init`) and any
review-runs PR that modifies `.claude/skills/`.

Work around by cloning into `$TMPDIR`, making edits there, and pushing:

```bash
WORK="$TMPDIR/repo-edit" && mkdir -p "$WORK" && cd "$WORK"
git clone "https://x-access-token:${GH_TOKEN}@github.com/numbagg/numbagg.git" repo
cd repo && git checkout -b "$BRANCH"
# Edit files, e.g. uvx tend@latest init, or edit .claude/skills/...
git -c user.name=numbagg-bot \
    -c user.email="272061629+numbagg-bot@users.noreply.github.com" \
    commit -am "..."
git push origin "$BRANCH"
```

Use `gh api repos/numbagg/numbagg/pulls/<N> -X PATCH -f title=... -F body=@...`
to update an existing PR's title/body — `gh pr edit` triggers a GraphQL query
requiring the `read:org` scope, which `BOT_TOKEN` does not have.
