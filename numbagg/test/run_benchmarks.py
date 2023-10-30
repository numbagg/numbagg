"""
Run the benchmarks and write the results to a markdown file at `.benchmarks/benchmark-output.md`.
"""

import subprocess
from pathlib import Path

import jq
import pandas as pd
from tabulate import tabulate


def run():
    json_path = Path(".benchmarks/benchmark.json")
    json_path.parent.mkdir(exist_ok=True, parents=True)
    subprocess.run(
        [
            "pytest",
            "numbagg/test/test_benchmark.py",
            "--benchmark-only",
            f"--benchmark-json={json_path}",
        ]
    )

    json = jq.compile(
        '.benchmarks | map(.params + {group, library: .params.library, func: .params.func | match("\\\\[numbagg.(.*?)\\\\]").captures[0].string, time: .stats.mean, })'
    ).input(text=json_path.read_text())

    df = pd.DataFrame.from_dict(json.first())

    df = df.set_index(["func", "library", "size"])["time"].unstack("library")

    # We want to order all `move_exp` functions together, rather than have them between
    # `move_count` and `move_mean`

    # But it's crazy difficult to sort a multiindex with a custom key in pandas...
    sorted_index = sorted(
        [(lib, size) for lib, size in df.index], key=lambda x: x[0].rsplit("_", 1)
    )
    df = (
        df.reindex(pd.MultiIndex.from_tuples(sorted_index, names=df.index.names))
        .reset_index()
        .assign(ratio=lambda df: df.eval("pandas/numbagg"))
        .assign(func=lambda x: x["func"].map(lambda x: f"`{x}`"))
    )

    # Surprisingly difficult to get pandas to print a nice-looking table...
    df = (
        df.assign(ratio=lambda x: x["ratio"].map("{:.2f}x".format))
        .assign(numbagg=lambda x: (x.numbagg * 1000))
        .assign(pandas=lambda x: (x.pandas * 1000))
    )
    full = (
        df.assign(func=lambda x: x["func"].where(lambda x: ~x.duplicated(), ""))
        .assign(numbagg=lambda x: (x.numbagg).map("{:>6.2f}ms".format))
        .assign(pandas=lambda x: (x.pandas).map("{:>6.2f}ms".format))
    )

    summary = (
        df.query("size == 10_000_000")
        .drop(columns="size")
        .assign(numbagg=lambda x: (x.numbagg).map("{:>6.0f}ms".format))
        .assign(pandas=lambda x: (x.pandas).map("{:>6.0f}ms".format))
    )

    text = ""
    for df in [full, summary]:
        values = df.to_dict(index=False, orient="split")["data"]
        markdown_table = tabulate(
            values,
            headers=df.columns,
            disable_numparse=True,
            colalign=["left"]
            + (["right"] if "size" in df.columns else [])
            + ["right", "right", "right"],
            tablefmt="pipe",
        )
        text += markdown_table
        text += "\n\n"
    Path(".benchmarks/benchmark-output.md").write_text(text)
    print(text)


if __name__ == "__main__":
    run()
