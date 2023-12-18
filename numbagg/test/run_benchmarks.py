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


def _sort_key(x):
    return (
        x[0].rsplit("_", 1),  # func
        x[2],  # size
        x[4],  # ndim
        x[3],  # length
    )


def run():
    json_path = Path(".benchmarks/benchmark.json")
    json_path.parent.mkdir(exist_ok=True, parents=True)
    if RUN:
        # pytest numbagg/test/test_benchmark.py --benchmark-only --benchmark-json=.benchmarks/benchmark.json
        subprocess.run(
            [
                "pytest",
                "-vv",
                "numbagg/test/test_benchmark.py",
                # If iterating on a single function, adding this will filter to a functions
                # "-k=test_benchmark_main[nanargmin",
                "-k=test_benchmark_main",
                "--benchmark-enable",
                "--benchmark-only",
                "--run-nightly",
                f"--benchmark-json={json_path}",
            ],
            check=True,
        )

    json = jq.compile(
        '.benchmarks[] | select(.name | index("test_benchmark_main[")) | .params + {group, library: .params.library, func: .params.func | match("\\\\[numbagg.(.*?)\\\\]").captures[0].string, time: .stats.median, }'
    ).input(text=json_path.read_text())

    df = pd.DataFrame.from_dict(json.all())

    df = (
        df.assign(size=lambda x: x["shape"].map(lambda x: np.prod(x)))
        .assign(length=lambda x: x["shape"].map(lambda x: x[-1]))
        .assign(ndim=lambda x: x["shape"].map(lambda x: len(x)))
        .assign(shape=lambda x: x["shape"].map(lambda x: tuple(x)).astype(str))
    )
    df = df.set_index(["func", "library", "shape", "size", "length", "ndim"])[
        "time"
    ].unstack("library")

    # We want to order all `move_exp` functions together, rather than have them between
    # `move_count` and `move_mean`

    # But it's crazy difficult to sort a multiindex with a custom key in pandas...
    sorted_index = sorted(
        [
            (func, shape, size, length, ndim)
            for (func, shape, size, length, ndim) in df.index
        ],
        # The third part of this finds the final number in `shape` and puts bigger
        # numbers first, so we get the biggest final axis (which favors bottleneck over
        # numbagg but is probably a better example)
        key=_sort_key,
    )
    df = (
        df.reindex(pd.MultiIndex.from_tuples(sorted_index, names=df.index.names))
        .reset_index()
        .assign(numbagg_ratio=lambda df: df.eval("numbagg/numbagg"))
        .assign(pandas_ratio=lambda df: df.eval("pandas/numbagg"))
        .assign(bottleneck_ratio=lambda df: df.eval("bottleneck/numbagg"))
        .assign(numpy_ratio=lambda df: df.eval("numpy/numbagg"))
        .assign(func=lambda x: x["func"].map(lambda x: f"`{x}`"))
    )

    # Surprisingly difficult to get pandas to print a nice-looking table...
    df = (
        df.assign(
            numbagg_ratio=lambda x: x["numbagg_ratio"].map(
                lambda x: f"{x:.2f}x" if not np.isnan(x) else "n/a"
            ),
            pandas_ratio=lambda x: x["pandas_ratio"].map(
                lambda x: f"{x:.2f}x" if not np.isnan(x) else "n/a"
            ),
            bottleneck_ratio=lambda x: x["bottleneck_ratio"].map(
                lambda x: f"{x:.2f}x" if not np.isnan(x) else "n/a"
            ),
            numpy_ratio=lambda x: x["numpy_ratio"].map(
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
            numpy=lambda x: (x.numpy * 1000).map(
                lambda x: f"{x:.0f}ms" if not np.isnan(x) else "n/a"
            ),
        )
        .reset_index(drop=True)[
            [
                "func",
                "shape",
                "size",
                "numbagg",
                "pandas",
                "bottleneck",
                "numpy",
                "numbagg_ratio",
                "pandas_ratio",
                "bottleneck_ratio",
                "numpy_ratio",
            ]
        ]
        .rename_axis(columns=None)
    )

    def make_summary_df(df, nd: int):
        # Take the biggest of a dimension
        shape = (
            df[lambda x: x["shape"].astype(str).map(lambda x: x.count(" ")) == (nd - 1)]
            .sort_values(by="size")["shape"]
            .iloc[-1]
        )

        return (
            df.query(f"shape == '{shape}'")
            .reset_index()
            .set_index(["func", "shape"])
            .unstack("shape")
            .pipe(
                lambda x: x[
                    [
                        c
                        for c in x.columns
                        # Want to confirm that numpy is handled correctly before showing
                        # in summary
                        if c[0].endswith("ratio")
                        and c[0] not in ["numbagg_ratio", "numpy_ratio"]
                    ]
                ]
            )
        )

    summary_1d = make_summary_df(df, 1)
    summary_2d = make_summary_df(df, 2)
    summary = pd.concat([summary_1d, summary_2d], axis=1).fillna("n/a")
    summary = summary.reset_index()

    values = summary.to_dict(index=False, orient="split")["data"]  # type: ignore[unused-ignore,call-overload]
    summary_markdown = tabulate(
        values,
        headers=["func"]
        + [f"{c[0].removesuffix('_ratio')}<br>`{c[1]}`" for c in summary.columns[1:]],
        disable_numparse=True,
        colalign=["left"] + ["right"] * (len(summary.columns) - 1),
        tablefmt="pipe",
    )

    full = df.assign(
        func=lambda x: x.reset_index()["func"].where(lambda x: ~x.duplicated(), "")
    )
    values = full.to_dict(index=False, orient="split")["data"]  # type: ignore[unused-ignore,call-overload]
    full_markdown = tabulate(
        values,
        headers=full.columns,
        disable_numparse=True,
        colalign=["left"] + ["right"] * (len(full.columns) - 1),
        tablefmt="pipe",
    )

    text = f"""
### Summary benchmark

Two benchmarks summarize numbagg's performance — one with a 1D array with no
parallelization, and one with a 2D array with the potential for parallelization.
Numbagg's relative performance is much higher where parallelization is possible.

The values in the table are numbagg's performance as a multiple of other libraries for a
given shaped array, calculated over the final axis. (so 1.00x means numbagg is equal,
higher means numbagg is faster.)

{summary_markdown}

### Full benchmarks

<details>

{full_markdown}

</details>

    """
    Path(".benchmarks/benchmark-output.md").write_text(text)
    print(text)


if __name__ == "__main__":
    run()
