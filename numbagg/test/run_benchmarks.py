import subprocess
from pathlib import Path

import jq
import pandas as pd
from tabulate import tabulate


def run():
    # subprocess.run(
    #     [
    #         "pytest",
    #         "numbagg/test/test_benchmark.py",
    #         "--benchmark-only",
    #         "--benchmark-json=benchmark.json",
    #     ]
    # )

    json = jq.compile(
        '.benchmarks | map(.params + {group, library: .params.library, func: .params.func | match("\\\\[numbagg.(.*?)\\\\]").captures[0].string, time: .stats.mean, })'
    ).input(text=Path("benchmark.json").read_text())

    df = (
        pd.DataFrame.from_dict(json.first())
        .set_index(["func", "library", "size"])["time"]
        .unstack("library")
        .assign(ratio=lambda df: df.eval("pandas/numbagg"))
        .reset_index()
        .assign(func=lambda x: x["func"].map(lambda x: f"`{x}`"))
    )

    # Surprisingly difficult to get pandas to print a nice-looking table...
    df = (
        df.assign(ratio=lambda x: x["ratio"].map("{:.2f}x".format))
        .assign(numbagg=lambda x: (x.numbagg * 1000).map("{:>6.2f}ms".format))
        .assign(pandas=lambda x: (x.pandas * 1000).map("{:>6.2f}ms".format))
    )
    full = df.assign(func=lambda x: x["func"].where(lambda x: ~x.duplicated(), ""))

    summary = df.query("size == 10_000_000").drop(columns="size")

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
        print(markdown_table)


if __name__ == "__main__":
    run()
