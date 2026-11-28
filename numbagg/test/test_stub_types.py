"""Pin the return types the `funcs.pyi` aggregation overloads give type checkers.

`assert_type` is a no-op at runtime; the `ty` pre-commit hook is what checks these.
Running them under pytest also checks each runtime result against the declared type.
"""

import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

import numbagg

if sys.version_info >= (3, 11):
    from typing import assert_type
else:  # `typing_extensions` reaches the dev environment through mypy.
    from typing_extensions import assert_type

_F64: NDArray[np.float64] = np.ones((2, 3))
_F32: NDArray[np.float32] = np.ones((2, 3), dtype=np.float32)
_I64: NDArray[np.int64] = np.ones((2, 3), dtype=np.int64)


def test_axis_none_returns_the_scalar_type():
    assert isinstance(assert_type(numbagg.nansum(_F64), np.float64), np.float64)
    assert isinstance(assert_type(numbagg.nansum(_I64), np.int64), np.int64)
    assert isinstance(assert_type(numbagg.nanmean(_F32), np.float32), np.float32)
    assert isinstance(assert_type(numbagg.nanvar(_F64), np.float64), np.float64)
    assert isinstance(assert_type(numbagg.nanstd(_F32), np.float32), np.float32)
    assert isinstance(assert_type(numbagg.nanmax(_F32), np.float32), np.float32)
    assert isinstance(assert_type(numbagg.nanmin(_I64), np.int64), np.int64)
    assert isinstance(assert_type(numbagg.nanargmax(_F64), np.int64), np.int64)
    assert isinstance(assert_type(numbagg.nanargmin(_F32), np.int64), np.int64)
    assert isinstance(assert_type(numbagg.nancount(_F64), np.int64), np.int64)
    assert isinstance(assert_type(numbagg.count(_F64), np.int64), np.int64)
    assert isinstance(assert_type(numbagg.allnan(_F64), np.bool_), np.bool_)
    assert isinstance(assert_type(numbagg.anynan(_I64), np.bool_), np.bool_)


def test_explicit_axis_returns_scalar_or_array():
    result = assert_type(numbagg.nansum(_F32, axis=0), np.float32 | NDArray[np.float32])
    assert isinstance(result, np.ndarray) and result.dtype == np.float32
    assert isinstance(
        assert_type(
            numbagg.nanmean(_F64, axis=(0, 1)), np.float64 | NDArray[np.float64]
        ),
        np.float64,
    )
    assert_type(numbagg.nanvar(_F64, ddof=0, axis=-1), np.float64 | NDArray[np.float64])
    assert_type(numbagg.nanmax(_F64, axis=0), np.float64 | NDArray[np.float64])
    assert_type(numbagg.nanargmax(_F64, axis=0), np.int64 | NDArray[np.int64])
    assert_type(numbagg.nancount(_F64, axis=0), np.int64 | NDArray[np.int64])
    assert_type(numbagg.allnan(_F64, axis=0), np.bool_ | NDArray[np.bool_])


def test_unreadable_dtype_falls_back_to_any():
    # `nanmean` has only float loops, so an integer input is cast to float64 — not the
    # input's scalar type, which the first overload would claim.
    assert isinstance(assert_type(numbagg.nanmean(_I64), Any), np.float64)
    untyped: Any = _F64
    assert_type(numbagg.nansum(untyped), Any)
