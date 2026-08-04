"""Tests for the benchmark-table generator in `run_benchmarks.py`.

These cover the parts that silently mis-handled functions which aren't decorated
gufuncs — `nanmedian` is a plain wrapper around `nanquantile`, so it has neither
`numbagg.<name>` as its `__repr__` nor a `supports_parallel` attribute.
"""

import json

import pytest

from .run_benchmarks import _func_label

jq = pytest.importorskip("jq", reason="jq isn't installed on Windows")

from .run_benchmarks import _JQ_PROGRAM  # noqa: E402


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
def test_jq_extracts_func_name(func_repr, expected):
    records = jq.compile(_JQ_PROGRAM).input(text=_benchmark_json(func_repr)).all()
    assert [r["func"] for r in records] == [expected]


def test_jq_keeps_plain_function_rows():
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
