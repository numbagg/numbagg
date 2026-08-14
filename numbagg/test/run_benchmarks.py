"""
Run the benchmarks and write the results to a markdown file at `.benchmarks/benchmark-output.md`.
"""

import argparse
import subprocess
import sys
from pathlib import Path

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


# pytest-benchmark can't serialize the function objects it's parametrized over, so it
# records them as `UNSERIALIZABLE[<repr>]`. Decorated functions have a `__repr__` of
# `numbagg.<name>`; plain wrappers like `nanmedian` fall back to the default
# `<function nanmedian at 0x...>`. Match both, or their rows are silently dropped; any
# other repr form raises rather than vanishing the same way.
_JQ_PROGRAM = r"""
.benchmarks[]
| select(.name | test("test_benchmark_(main|matrix)\\["))
| .params + {
    group,
    library: .params.library,
    func: (
      .params.func
      | (
          match("\\[numbagg\\.(.*?)\\]")
          // match("<function (.*?) at ")
          // error("benchmark row has an unrecognized func repr: \(.)")
        )
      | .captures[0].string
    ),
    time: .stats.median,
  }
"""


def _func_label(func_name: str) -> str:
    """Markdown label for a function, with the footnote marker its type calls for."""
    if "matrix" in func_name:
        return f"`{func_name}`[^6]"
    func = getattr(numbagg, func_name)
    # `nanmedian` is a plain wrapper around `nanquantile` rather than a decorated
    # gufunc, so it has no `supports_parallel` attribute. It inherits the
    # parallelism of the function it delegates to, so default to parallel.
    if getattr(func, "supports_parallel", True):
        return f"`{func_name}`"
    return f"`{func_name}`[^5]"


def markdown_tables(records: list[dict]) -> tuple[str, str]:
    """The summary and full benchmark tables as markdown, from `_JQ_PROGRAM`'s output."""

    df = pd.DataFrame(records)

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
        .assign(func=lambda x: x["func"].map(_func_label))
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
        """Ratios at one dimensionality, indexed by func with a column per shape.

        Matrix functions have no 1D benchmark — the smallest input they accept is 2D —
        so the 1D column falls back to their largest 2D shape. They're absent from the
        2D column, because no comparison library accepts more than two dimensions and
        so there's nothing to compare a parallelized matrix run against.
        """

        def ratios_at_largest_shape(func_df, ndim):
            filtered = func_df[lambda x: x["ndim"] == ndim]
            if filtered.empty:
                return None

            # Use largest array shape for performance comparison
            shape = filtered.sort_values(by="size")["shape"].iloc[-1]
            return (
                func_df.query(f"shape == '{shape}'")
                .set_index(["func", "shape"])
                .unstack("shape")  # Pivot: functions as rows, shapes as columns
                .pipe(
                    lambda x: x[
                        [
                            c
                            for c in x.columns
                            if c[0].endswith("ratio") and c[0] != "numbagg_ratio"
                        ]
                    ]
                )
            )

        is_matrix = df["func"].str.contains("matrix", na=False)
        results = [ratios_at_largest_shape(df[~is_matrix], nd)]
        if nd == 1:
            results.append(ratios_at_largest_shape(df[is_matrix], 2))
        results = [r for r in results if r is not None]
        return pd.concat(results, axis=0) if results else pd.DataFrame()

    def summary_value(summary_df, func, lib):
        """The ratio for a function/library pair, from whichever shape column holds it.

        Concatenating the matrix and non-matrix frames leaves one column per shape, and
        each function has a value in exactly one of them.
        """
        if func not in summary_df.index:
            return "n/a"
        values = summary_df.loc[
            func, [c for c in summary_df.columns if c[0].removesuffix("_ratio") == lib]
        ].dropna()
        return values.iloc[0] if len(values) else "n/a"

    summaries = [(nd, make_summary_df(df, nd)) for nd in (1, 2)]

    summary_funcs = set().union(*(summary_df.index for _, summary_df in summaries))
    # Rows follow the order `_sort_key` established above
    funcs = [func for func in df["func"].drop_duplicates() if func in summary_funcs]
    # Columns follow the library order of the full benchmark table below

    summary_libs = [
        lib
        for lib in libraries
        if lib != "numbagg"
        and any(
            c[0].removesuffix("_ratio") == lib
            for _, summary_df in summaries
            for c in summary_df.columns
        )
    ]

    if funcs and summary_libs:
        summary_markdown = tabulate(
            [
                [func]
                + [
                    summary_value(summary_df, func, lib)
                    for _, summary_df in summaries
                    for lib in summary_libs
                ]
                for func in funcs
            ],
            headers=["func"]
            + [f"{nd}D<br>{lib}" for nd, _ in summaries for lib in summary_libs],
            disable_numparse=True,
            colalign=["left"] + ["right"] * (len(summaries) * len(summary_libs)),
            tablefmt="pipe",
        )
    else:
        summary_markdown = "No benchmark data available for summary."

    full = df.assign(
        func=lambda x: x.reset_index()["func"].where(lambda x: ~x.duplicated(), "")
    )
    values = full.to_dict(index=False, orient="split")["data"]
    full_markdown = tabulate(
        values,
        headers=full.columns,
        disable_numparse=True,
        colalign=["left"] + ["right"] * (len(full.columns) - 1),
        tablefmt="pipe",
    )

    return summary_markdown, full_markdown


def run(k_filter, run_tests, extra_args):
    import jq  # ty:ignore[unresolved-import]

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

    records = jq.compile(_JQ_PROGRAM).input(text=json_path.read_text()).all()
    summary_markdown, full_markdown = markdown_tables(records)

    text = f"""
### Summary benchmark

Two benchmarks summarize numbagg's performance — the first with a 1D array of 10M elements without
parallelization, and a second with a 2D array of 100x10K elements with parallelization[^6]. Numbagg's relative
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
        default="test_benchmark_main or test_benchmark_matrix",
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
