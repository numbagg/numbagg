"""
Run the benchmarks and write the results to a markdown file at `.benchmarks/benchmark-output.md`.
"""

import subprocess
from pathlib import Path

import jq
import numpy as np
import pandas as pd
from tabulate import tabulate


def run():
    json_path = Path(".benchmarks/benchmark.json")
    # json_path.parent.mkdir(exist_ok=True, parents=True)
    # subprocess.run(
    #     [
    #         "pytest",
    #         "numbagg/test/test_benchmark.py",
    #         "--benchmark-only",
    #         f"--benchmark-json={json_path}",
    #     ]
    # )

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
        .assign(pandas_ratio=lambda df: df.eval("pandas/numbagg"))
        .assign(bottleneck_ratio=lambda df: df.eval("bottleneck/numbagg"))
        .assign(func=lambda x: x["func"].map(lambda x: f"`{x}`"))
    )

    # Surprisingly difficult to get pandas to print a nice-looking table...
    df = (
        df.assign(
            pandas_ratio=lambda x: x["pandas_ratio"].map(
                lambda x: f"{x:.2f}x" if not np.isnan(x) else "n/a"
            )
        )
        .assign(
            bottleneck_ratio=lambda x: x["bottleneck_ratio"].map(
                lambda x: f"{x:.2f}x" if not np.isnan(x) else "n/a"
            )
        )
        .assign(
            numbagg=lambda x: (x.numbagg * 1000).map(
                lambda x: f"{x:>6.2f}ms" if not np.isnan(x) else "n/a"
            )
        )
        .assign(
            pandas=lambda x: (x.pandas * 1000).map(
                lambda x: f"{x:>6.2f}ms" if not np.isnan(x) else "n/a"
            )
        )
        .assign(
            bottleneck=lambda x: (x.bottleneck * 1000).map(
                lambda x: f"{x:>6.2f}ms" if not np.isnan(x) else "n/a"
            )
        )
    ).reindex(
        columns=[
            "func",
            "size",
            "numbagg",
            "pandas",
            "bottleneck",
            "pandas_ratio",
            "bottleneck_ratio",
        ]
    )
    full = df.assign(func=lambda x: x["func"].where(lambda x: ~x.duplicated(), ""))

    summary = df.query("size == 10_000_000").drop(columns="size")

    text = ""
    for df in [full, summary]:
        values = df.to_dict(index=False, orient="split")["data"]
        markdown_table = tabulate(
            values,
            headers=df.columns,
            disable_numparse=True,
            colalign=["left"] + ["right"] * (len(df.columns) - 1),
            tablefmt="pipe",
        )
        text += markdown_table
        text += "\n\n"
    Path(".benchmarks/benchmark-output.md").write_text(text)
    print(text)


if __name__ == "__main__":
    run()
