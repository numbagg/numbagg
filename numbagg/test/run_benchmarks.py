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


def run(k_filter, run_tests, extra_args):
    import jq  # type: ignore[import-not-found]

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
        r'.benchmarks[] | select(.name | test("test_benchmark_(main|matrix)\\[")) | .params + {group, library: .params.library, func: .params.func | match("\\[numbagg.(.*?)\\]").captures[0].string, time: .stats.median, }'
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
                lambda func_name: (
                    f"`{func_name}`{'[^6]' if 'matrix' in func_name else '[^5]' if not getattr(numbagg, func_name).supports_parallel else ''}"
                )
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
        """Create summary DataFrame for benchmark results.

        Matrix functions require special handling to appear in main summary columns:
        - nd=1 (1D column): Use their LARGEST 2D matrix shape
        - nd=2 (2D column): Use their LARGEST 3D matrix shape (demonstrates parallelization)

        This allows matrix functions to demonstrate parallelization without separate columns.
        """

        def process_functions(func_df, target_ndim, source_df=None):
            """Process a subset of functions with target dimensionality."""
            if source_df is None:
                source_df = func_df

            filtered = func_df[lambda x: x["ndim"] == target_ndim]
            if filtered.empty:
                return None

            # Use largest array shape for performance comparison
            shape = filtered.sort_values(by="size")["shape"].iloc[-1]
            return (
                source_df.query(f"shape == '{shape}'")
                .reset_index()
                .set_index(["func", "shape"])
                .unstack("shape")  # Pivot: functions as rows, shapes as columns
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

        # Split data by matrix vs non-matrix functions
        matrix_df = df[df["func"].str.contains("matrix", na=False)]
        non_matrix_df = df[~df["func"].str.contains("matrix", na=False)]

        results = []

        # Process non-matrix functions: use regular dimensionality (1D→1D, 2D→2D)
        if not non_matrix_df.empty:
            regular_result = process_functions(non_matrix_df, nd)
            if regular_result is not None:
                results.append(regular_result)

        # Process matrix functions: use special dimensionality mapping
        if not matrix_df.empty:
            matrix_target_ndim = {1: 2, 2: 3}.get(nd)  # 1D→2D, 2D→3D
            if matrix_target_ndim:
                matrix_result = process_functions(
                    matrix_df, matrix_target_ndim, matrix_df
                )
                if matrix_result is not None:
                    results.append(matrix_result)

        # Combine all results into single DataFrame
        return pd.concat(results, axis=0) if results else pd.DataFrame()

    def get_column_value(summary_df, func, lib, dimension, matrix_shape_exclusions):
        """Extract column value for a function/library pair with matrix function handling."""
        matching_cols = [
            col for col in summary_df.columns if col[0].removesuffix("_ratio") == lib
        ]

        if not matching_cols or func not in summary_df.index:
            return "n/a"

        # For matrix functions, try to find matrix-specific column first
        value = None
        if "matrix" in func:
            matrix_cols = [
                col
                for col in matching_cols
                if not any(exclusion in col[1] for exclusion in matrix_shape_exclusions)
            ]
            if matrix_cols:
                value = summary_df.loc[func, matrix_cols[0]]

        # Fallback to first column if no matrix-specific column found
        if value is None:
            value = summary_df.loc[func, matching_cols[0]]

        return value if not pd.isna(value) else "n/a"

    def process_dimension_data(
        summary_df, func, dimension, all_libs, matrix_shape_exclusions
    ):
        """Process data for a single dimension (1D or 2D) for a specific function."""
        if summary_df.empty:
            return {f"{dimension}_{lib}": "n/a" for lib in all_libs}

        return {
            f"{dimension}_{lib}": get_column_value(
                summary_df, func, lib, dimension, matrix_shape_exclusions
            )
            for lib in all_libs
        }

    # Create summaries including matrix functions in main 1D/2D columns
    summary_1d = make_summary_df(df, 1)
    summary_2d = make_summary_df(df, 2)

    # Matrix function shape exclusion patterns
    matrix_exclusions = {"1D": ["(10000000,)"], "2D": ["(100, 100000)"]}

    # Combine summaries properly - reorganize to have 1D/2D structure
    if summary_1d.empty and summary_2d.empty:
        summary = pd.DataFrame()
    else:
        # Extract unique libraries and functions
        libs_1d = (
            set(col[0].removesuffix("_ratio") for col in summary_1d.columns)
            if not summary_1d.empty
            else set()
        )
        libs_2d = (
            set(col[0].removesuffix("_ratio") for col in summary_2d.columns)
            if not summary_2d.empty
            else set()
        )
        all_libs = sorted(libs_1d | libs_2d)
        all_functions = set(summary_1d.index if not summary_1d.empty else []) | set(
            summary_2d.index if not summary_2d.empty else []
        )

        # Create properly structured summary
        summary_data = []
        for func in all_functions:
            row = {"func": func}

            # Process 1D and 2D dimensions
            for dim, summary_df, exclusions in [
                ("1D", summary_1d, matrix_exclusions["1D"]),
                ("2D", summary_2d, matrix_exclusions["2D"]),
            ]:
                dim_data = process_dimension_data(
                    summary_df, func, dim, all_libs, exclusions
                )
                row.update(dim_data)

            summary_data.append(row)

        summary = pd.DataFrame(summary_data).set_index("func")

    if not summary.empty:
        summary = summary.reset_index()
        values = summary.to_dict(index=False, orient="split")["data"]

        # Generate headers from the new column structure (1D_pandas, 2D_pandas, etc.)
        headers = ["func"]
        for col in summary.columns[1:]:
            # Column names are like "1D_pandas", "2D_numpy"
            dimension, library = col.split("_", 1)
            headers.append(f"{dimension}<br>{library}")

        summary_markdown = tabulate(
            values,
            headers=headers,
            disable_numparse=True,
            colalign=["left"] + ["right"] * (len(summary.columns) - 1),
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
