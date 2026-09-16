from collections.abc import Iterable
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from .utils import FloatArray, NumericArray

_T = TypeVar("_T", bound=NumericArray)

def allnan(arrays, /, *, axis: int | tuple[int, ...] | None = None): ...
def anynan(arrays, /, *, axis: int | tuple[int, ...] | None = None): ...
def nancount(arrays, /, *, axis: int | tuple[int, ...] | None = None): ...
def nansum(arrays, /, *, axis: int | tuple[int, ...] | None = None): ...
def nanmean(arrays, /, *, axis: int | tuple[int, ...] | None = None): ...
def nanvar(arrays, /, *, ddof: int = 1, axis: int | tuple[int, ...] | None = None): ...
def nanstd(arrays, /, *, ddof: int = 1, axis: int | tuple[int, ...] | None = None): ...
def nanargmax(arr: NDArray[Any], *args, axis: tuple[int, ...] | int | None = None): ...
def nanargmin(arr: NDArray[Any], *args, axis: tuple[int, ...] | int | None = None): ...
def nanmax(arr: NDArray[Any], *args, axis: tuple[int, ...] | int | None = None): ...
def nanmin(arr: NDArray[Any], *args, axis: tuple[int, ...] | int | None = None): ...

# `**kwargs` reaches the gufunc for these two as well, so `out=` fixes the result
# dtype: `nanmedian(a, axis=-1, out=<int64 array>, casting="unsafe")` gives back int64,
# and a boolean `out` likewise. The shape doesn't carry through either — the quantile
# axis is moved to the front, so the return is a reshaped view of `out`, not `out`
# itself. Only "some ndarray" holds for every call form they accept, which is what the
# matrix declarations below say too.
def nanquantile(
    a: NDArray[np.float64],
    quantiles: float | Iterable[float],
    axis: int | tuple[int, ...] | None = None,
    **kwargs,
) -> np.ndarray: ...
def nanmedian(
    a: NDArray[np.float64], *, axis: int | tuple[int, ...] | None = None, **kwargs
) -> np.ndarray: ...
def bfill(
    arr: _T,
    *,
    limit: int | None = None,
    axis: int = -1,
) -> _T: ...
def ffill(
    arr: _T,
    *,
    limit: int | None = None,
    axis: int = -1,
) -> _T: ...

# `(..., vars, obs) -> (..., vars, vars)`: the trailing `obs` axis is replaced by a
# second `vars` axis, so the input's shape does not carry through. Nor does its dtype:
# `**kwargs` reaches the gufunc, so `dtype=` picks the loop, and `out=` returns the
# supplied array itself — `nancovmatrix(a_float32, out=<int64 array>, casting="unsafe")`
# hands back that int64 array. Only "some ndarray" holds for every call form the runtime
# accepts, which is what `move_covmatrix` / `move_corrmatrix` declare too.
def nancovmatrix(a: FloatArray, **kwargs) -> np.ndarray: ...
def nancorrmatrix(a: FloatArray, **kwargs) -> np.ndarray: ...

count = nancount
