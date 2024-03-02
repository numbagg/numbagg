"""
Run the benchmarks and write the results to a markdown file at `.benchmarks/benchmark-output.md`.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import jq
import numpy as np
import pandas as pd
from tabulate import tabulate

import numbagg


def _sort_key(x):
    return (
        x[0].rsplit("_", 1),  # func
        x[2],  # size
        x[4],  # ndim
        x[3],  # length
    )


def run(k_filter, run_tests, extra_args):
    json_path = Path(".benchmarks/benchmark.json")
    json_path.parent.mkdir(exist_ok=True, parents=True)
    if run_tests:
        subprocess.run(
            [
                "pytest",
                "-vv",
                "numbagg/test/test_benchmark.py",
                f"-k={k_filter}",
                "--benchmark-enable",
                "--benchmark-only",
                "--run-nightly",
                f"--benchmark-json={json_path}",
            ]
            + extra_args,
            check=True,
        )

    json = jq.compile(
        r'.benchmarks[] | select(.name | index("test_benchmark_main[")) | .params + {group, library: .params.library, func: .params.func | match("\\[numbagg.(.*?)\\]").captures[0].string, time: .stats.median, }'
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
        .assign(
            func=lambda x: x["func"].map(
                lambda x: f"`{x}`{'[^5]' if not getattr(numbagg, x).supports_parallel else ''}"
            )
        )
    )

    # Do numbagg last, so the division works below
    libraries = [c for c in ["pandas", "bottleneck", "numpy"] if c in df.columns] + [
        "numbagg"
    ]

    for library in libraries:
        df[f"{library}_ratio"] = (df[library] / df["numbagg"]).map(
            lambda x: f"{x:.2f}x" if not np.isnan(x) else "n/a"
        )
        df[library] = (df[library] * 1000).map(
            lambda x: f"{x:.0f}ms" if not np.isnan(x) else "n/a"
        )

    # Surprisingly difficult to get pandas to print a nice-looking table...
    df = df.reset_index(drop=True)[
        [
            "func",
            "shape",
            "size",
            "ndim",
        ]
        + list(libraries)
        + [f"{library}_ratio" for library in libraries]
    ].rename_axis(columns=None)

    def make_summary_df(df, nd: int):
        shape = df[lambda x: x["ndim"] == nd].sort_values(by="size")["shape"].iloc[-1]

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
                        if c[0].endswith("ratio") and c[0] not in ["numbagg_ratio"]
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
        + [
            # Kinda a horrible expression; we're converting the string to a tuple with
            # `eval` and then formatting its elements as scientific notation.
            # f"`({', '.join(f'{s:.0e}' for s in eval(c[1]))})`<br>{c[0].removesuffix('_ratio')}".replace(
            #     "e+0", "e"
            # )
            f"{len(eval(c[1]))}D<br>{c[0].removesuffix('_ratio')}".replace("e+0", "e")
            for c in summary.columns[1:]
        ],
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

Two benchmarks summarize numbagg's performance — the first with a 1D array of 10M elements without
parallelization, and a second with a 2D array of 100x10K elements with parallelization. Numbagg's relative
performance is much higher where parallelization is possible. A wider range of arrays is
listed in the full set of benchmarks below.

The values in the table are numbagg's performance as a multiple of other libraries for a
given shaped array calculated over the final axis. (so 1.00x means numbagg is equal,
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
    parser = argparse.ArgumentParser(
        description="Run main benchmarks and output results. Pass any additional options after a `--`; for example `python run_benchmarks.py -- --benchmark-max-time=10`"
    )
    parser.add_argument(
        "-k",
        "--filter",
        default="test_benchmark_main",
        help="Filter for pytest -k option; for example `test_benchmark_main and group_nanmean and numbagg`",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        default=True,
        help="Run the tests (default: True)",
    )
    parser.add_argument(
        "--no-run-tests",
        action="store_false",
        dest="run_tests",
        help="Do not run the tests",
    )

    # Split arguments at '--'
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args, remaining_args = sys.argv[:idx], sys.argv[idx + 1 :]
    else:
        args, remaining_args = sys.argv, []

    parsed_args = parser.parse_args(args[1:])

    run(parsed_args.filter, parsed_args.run_tests, remaining_args)
