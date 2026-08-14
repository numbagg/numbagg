"""Tests for the benchmark-table generator in `run_benchmarks.py`.

These cover the parts that silently mishandled functions which aren't decorated
gufuncs — `nanmedian` is a plain wrapper around `nanquantile`, so it has neither
`numbagg.<name>` as its `__repr__` nor a `supports_parallel` attribute — and the
assembly of the summary table, whose rows and columns have to come out in a stable
order for the README's committed copy to be diffable.
"""

import json

import pytest

from .run_benchmarks import _JQ_PROGRAM, _func_label, markdown_tables


@pytest.fixture(scope="module")
def jq():
    # A module-level `importorskip` would skip the `_func_label` tests too, which
    # don't need jq.
    return pytest.importorskip("jq", reason="jq isn't installed on Windows")


def _benchmark_json(func_repr: str) -> str:
    return json.dumps(
        {
            "benchmarks": [
                {
                    "name": "test_benchmark_main[func-shape0-numbagg]",
                    "group": "group0",
                    "params": {
                        "func": f"UNSERIALIZABLE[{func_repr}]",
                        "shape": [3, 1000],
                        "library": "numbagg",
                    },
                    "stats": {"median": 0.001},
                }
            ]
        }
    )


@pytest.mark.parametrize(
    ("func_repr", "expected"),
    [
        # Decorated gufuncs define `__repr__` as `numbagg.<name>`.
        ("numbagg.nanmean", "nanmean"),
        ("numbagg.move_exp_nanmean", "move_exp_nanmean"),
        ("numbagg.nancorrmatrix", "nancorrmatrix"),
        # Plain functions fall back to CPython's default repr.
        ("<function nanmedian at 0x7f650182e8e0>", "nanmedian"),
    ],
)
def test_jq_extracts_func_name(jq, func_repr, expected):
    records = jq.compile(_JQ_PROGRAM).input(text=_benchmark_json(func_repr)).all()
    assert [r["func"] for r in records] == [expected]


def test_jq_keeps_plain_function_rows(jq):
    # Previously the regex only matched `[numbagg.<name>]`, so a `nanmedian` row
    # produced no output at all and vanished from the generated tables.
    records = (
        jq.compile(_JQ_PROGRAM)
        .input(text=_benchmark_json("<function nanmedian at 0x7f650182e8e0>"))
        .all()
    )
    assert len(records) == 1
    assert records[0]["shape"] == [3, 1000]
    assert records[0]["library"] == "numbagg"


def test_jq_raises_on_unrecognized_repr(jq):
    # A repr form neither pattern covers should fail loudly rather than drop the row.
    with pytest.raises(ValueError, match="unrecognized func repr"):
        (
            jq.compile(_JQ_PROGRAM)
            .input(text=_benchmark_json("<built-in function foo>"))
            .all()
        )


@pytest.mark.parametrize(
    ("func_name", "expected"),
    [
        # Parallelized gufunc — no marker.
        ("nanmean", "`nanmean`"),
        # Not parallelized — footnote 5.
        ("nanargmax", "`nanargmax`[^5]"),
        # Matrix function — footnote 6.
        ("nancorrmatrix", "`nancorrmatrix`[^6]"),
        # Plain wrapper with no `supports_parallel`; previously raised
        # AttributeError here.
        ("nanmedian", "`nanmedian`"),
    ],
)
def test_func_label(func_name, expected):
    assert _func_label(func_name) == expected


# The shapes and comparison libraries each function is benchmarked over, mirroring
# `test_benchmark.py` and `conftest.py`: matrix functions have no 1D benchmark, and
# only pandas to compare against.
_BENCHMARKED = {
    "move_mean": ([(10_000_000,), (100, 100_000)], ["numbagg", "pandas", "bottleneck"]),
    "move_exp_nanmean": ([(10_000_000,), (100, 100_000)], ["numbagg", "pandas"]),
    "nancorrmatrix": ([(5, 100_000), (3_000, 5, 1_000)], ["numbagg", "pandas"]),
    "nanmean": ([(10_000_000,), (100, 100_000)], ["numbagg", "pandas", "numpy"]),
    "nansum": ([(10_000_000,), (100, 100_000)], ["numbagg", "pandas", "numpy"]),
}


@pytest.fixture
def records():
    """One record per (func, library, shape), as `_JQ_PROGRAM` emits them.

    Every comparison library takes twice as long as numbagg, so each populated cell
    reads `2.00x` and anything else is a gap in the benchmark rather than a slow run.
    """
    return [
        {
            "func": func,
            "library": library,
            "shape": list(shape),
            "time": 0.001 if library == "numbagg" else 0.002,
            "group": f"{func}|{shape}",
        }
        for func, (shapes, libraries) in _BENCHMARKED.items()
        for shape in shapes
        for library in libraries
        # No comparison library takes more than 2 dimensions
        if library == "numbagg" or len(shape) <= 2
    ]


def _cells(markdown: str) -> list[list[str]]:
    """The header row and body rows of a pipe-format markdown table."""
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in markdown.splitlines()
        if line.startswith("|")
    ]
    return [rows[0]] + rows[2:]  # rows[1] is tabulate's alignment separator


def test_summary_header(records):
    header, *_ = _cells(markdown_tables(records)[0])
    # Libraries in the order the full table below uses, not alphabetical
    assert header == [
        "func",
        "1D<br>pandas",
        "1D<br>bottleneck",
        "1D<br>numpy",
        "2D<br>pandas",
        "2D<br>bottleneck",
        "2D<br>numpy",
    ]


def test_summary_row_order(records):
    # Previously the rows came from a set, so the order — and hence the README diff of
    # a regenerated table — was arbitrary. It should follow `_sort_key`, which groups
    # `move_exp_*` after all the plain `move_*` functions.
    _, *body = _cells(markdown_tables(records)[0])
    assert [row[0] for row in body] == [
        "`move_mean`",
        "`move_exp_nanmean`",
        "`nancorrmatrix`[^6]",
        "`nanmean`",
        "`nansum`",
    ]


def test_summary_matrix_function_uses_its_2d_shape(records):
    # The 1D column falls back to a matrix function's largest 2D shape; its 2D column
    # is empty because nothing else takes the 3D input.
    _, *body = _cells(markdown_tables(records)[0])
    rows = {row[0]: row[1:] for row in body}
    assert rows["`nancorrmatrix`[^6]"] == ["2.00x", "n/a", "n/a", "n/a", "n/a", "n/a"]
    assert rows["`nanmean`"] == ["2.00x", "n/a", "2.00x", "2.00x", "n/a", "2.00x"]


def test_full_table_keeps_every_shape(records):
    header, *body = _cells(markdown_tables(records)[1])
    shapes = [row[header.index("shape")] for row in body]
    assert sorted(set(shapes)) == [
        "(100, 100000)",
        "(10000000,)",
        "(3000, 5, 1000)",
        "(5, 100000)",
    ]
