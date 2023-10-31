"""
Run the benchmarks and write the results to a markdown file at `.benchmarks/benchmark-output.md`.
"""

import subprocess
from pathlib import Path

import jq
import numpy as np
import pandas as pd
from tabulate import tabulate

RUN = True
# RUN = False


def _sort_key(x):
    # The third part of this finds the final number in `shape` and puts bigger
    # numbers first, so we get the biggest final axis (which favors bottleneck over
    # numbagg but is probably a better example)
    print(x[1])

    print(tuple(reversed(x[1].split(" ,", 1))))
    return (
        x[0].rsplit("_", 1),  # func
        x[2],  # size
        tuple(reversed(x[1].split(" ,", 1))),
    )  # [:-1] + "Z")


def run():
    json_path = Path(".benchmarks/benchmark.json")
    json_path.parent.mkdir(exist_ok=True, parents=True)
    if RUN:
        # pytest numbagg/test/test_benchmark.py --benchmark-only --benchmark-json=.benchmarks/benchmark.json
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

    df = df.assign(size=lambda x: x["shape"].map(lambda x: np.prod(x))).assign(
        shape=lambda x: x["shape"].map(lambda x: tuple(x)).astype(str)
    )
    df = df.set_index(["func", "library", "shape", "size"])["time"].unstack("library")

    # We want to order all `move_exp` functions together, rather than have them between
    # `move_count` and `move_mean`

    # But it's crazy difficult to sort a multiindex with a custom key in pandas...
    sorted_index = sorted(
        [(func, shape, size) for func, shape, size in df.index],
        # The third part of this finds the final number in `shape` and puts bigger
        # numbers first, so we get the biggest final axis (which favors bottleneck over
        # numbagg but is probably a better example)
        key=_sort_key,
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
            ),
            bottleneck_ratio=lambda x: x["bottleneck_ratio"].map(
                lambda x: f"{x:.2f}x" if not np.isnan(x) else "n/a"
            ),
            numbagg=lambda x: (x.numbagg * 1000).map(
                lambda x: f"{x:.0f}ms" if not np.isnan(x) else "n/a"
            ),
            pandas=lambda x: (x.pandas * 1000).map(
                lambda x: f"{x:.0f}ms" if not np.isnan(x) else "n/a"
            ),
            bottleneck=lambda x: (x.bottleneck * 1000).map(
                lambda x: f"{x:.0f}ms" if not np.isnan(x) else "n/a"
            ),
        )
    ).reindex(
        columns=[
            "func",
            "shape",
            "size",
            "numbagg",
            "pandas",
            "bottleneck",
            "pandas_ratio",
            "bottleneck_ratio",
        ]
    )
    full = df.assign(func=lambda x: x["func"].where(lambda x: ~x.duplicated(), ""))

    # Take the biggest of each of 2D or >2D
    summary_2d = (
        df[lambda x: x["shape"].map(lambda x: x.count(",")) == 1]
        .groupby(by="func", sort=False)
        .last()
        .reset_index()
        .drop(columns=("size"))
    )
    summary_nd = (
        df[lambda x: x["shape"].map(lambda x: x.count(",")) > 1]
        .groupby(by="func", sort=False)
        .last()
        .reset_index()
        .drop(columns="size")
    )

    text = ""
    for title, df in (("2D", summary_2d), ("ND", summary_nd), ("All", full)):
        shapes = df["shape"].unique()
        if len(shapes) == 1:
            shape = shapes[0]
            df = df.drop(columns="shape")
        else:
            shape = None
        values = df.to_dict(index=False, orient="split")["data"]
        markdown_table = tabulate(
            values,
            headers=df.columns,
            disable_numparse=True,
            colalign=["left"] + ["right"] * (len(df.columns) - 1),
            tablefmt="pipe",
        )
        text += f"### {title}\n\n"
        if shape:
            text += f"Arrays of shape `{shape}`\n\n"
        text += ""
        text += markdown_table
        text += "\n\n"
    Path(".benchmarks/benchmark-output.md").write_text(text)
    print(text)


if __name__ == "__main__":
    run()
