from collections.abc import Iterable
from typing import Any, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from .utils import AxisLike, FloatArray, NumericArray

_T = TypeVar("_T", bound=NumericArray)

_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_FloatT = TypeVar("_FloatT", np.float64, np.float32)

# Return types follow numpy's own `sum` stub: `axis=None` reduces to the input's scalar
# type, an explicit axis returns that scalar or an array of it (a scalar when the axes
# named cover every dimension), and anything else gets `Any`. As in numpy's stubs,
# per-dtype widening is not modelled: `nanmax` on int32 returns int64 at runtime, but
# the stub says int32. Functions whose result dtype is fixed declare it directly —
# counts and indices are int64, the NaN tests bool. `nanmean`, `nanvar` and `nanstd`
# only have float loops, so an integer input, which the gufunc casts to float64, falls
# through to `Any`.
@overload
def allnan(arrays: NDArray[Any], /, *, axis: None = None) -> np.bool_: ...
@overload
def allnan(
    arrays: NDArray[Any], /, *, axis: AxisLike
) -> np.bool_ | NDArray[np.bool_]: ...
@overload
def allnan(arrays: Any, /, *, axis: AxisLike | None = None) -> Any: ...
@overload
def anynan(arrays: NDArray[Any], /, *, axis: None = None) -> np.bool_: ...
@overload
def anynan(
    arrays: NDArray[Any], /, *, axis: AxisLike
) -> np.bool_ | NDArray[np.bool_]: ...
@overload
def anynan(arrays: Any, /, *, axis: AxisLike | None = None) -> Any: ...
@overload
def nancount(arrays: NDArray[Any], /, *, axis: None = None) -> np.int64: ...
@overload
def nancount(
    arrays: NDArray[Any], /, *, axis: AxisLike
) -> np.int64 | NDArray[np.int64]: ...
@overload
def nancount(arrays: Any, /, *, axis: AxisLike | None = None) -> Any: ...
@overload
def nansum(arrays: NDArray[_ScalarT], /, *, axis: None = None) -> _ScalarT: ...
@overload
def nansum(
    arrays: NDArray[_ScalarT], /, *, axis: AxisLike
) -> _ScalarT | NDArray[_ScalarT]: ...
@overload
def nansum(arrays: Any, /, *, axis: AxisLike | None = None) -> Any: ...
@overload
def nanmean(arrays: NDArray[_FloatT], /, *, axis: None = None) -> _FloatT: ...
@overload
def nanmean(
    arrays: NDArray[_FloatT], /, *, axis: AxisLike
) -> _FloatT | NDArray[_FloatT]: ...
@overload
def nanmean(arrays: Any, /, *, axis: AxisLike | None = None) -> Any: ...
@overload
def nanvar(
    arrays: NDArray[_FloatT], /, *, ddof: int = 1, axis: None = None
) -> _FloatT: ...
@overload
def nanvar(
    arrays: NDArray[_FloatT], /, *, ddof: int = 1, axis: AxisLike
) -> _FloatT | NDArray[_FloatT]: ...
@overload
def nanvar(arrays: Any, /, *, ddof: int = 1, axis: AxisLike | None = None) -> Any: ...
@overload
def nanstd(
    arrays: NDArray[_FloatT], /, *, ddof: int = 1, axis: None = None
) -> _FloatT: ...
@overload
def nanstd(
    arrays: NDArray[_FloatT], /, *, ddof: int = 1, axis: AxisLike
) -> _FloatT | NDArray[_FloatT]: ...
@overload
def nanstd(arrays: Any, /, *, ddof: int = 1, axis: AxisLike | None = None) -> Any: ...
@overload
def nanargmax(arr: NDArray[Any], *args, axis: None = None) -> np.int64: ...
@overload
def nanargmax(
    arr: NDArray[Any], *args, axis: AxisLike
) -> np.int64 | NDArray[np.int64]: ...
@overload
def nanargmax(arr: Any, *args, axis: AxisLike | None = None) -> Any: ...
@overload
def nanargmin(arr: NDArray[Any], *args, axis: None = None) -> np.int64: ...
@overload
def nanargmin(
    arr: NDArray[Any], *args, axis: AxisLike
) -> np.int64 | NDArray[np.int64]: ...
@overload
def nanargmin(arr: Any, *args, axis: AxisLike | None = None) -> Any: ...
@overload
def nanmax(arr: NDArray[_ScalarT], *args, axis: None = None) -> _ScalarT: ...
@overload
def nanmax(
    arr: NDArray[_ScalarT], *args, axis: AxisLike
) -> _ScalarT | NDArray[_ScalarT]: ...
@overload
def nanmax(arr: Any, *args, axis: AxisLike | None = None) -> Any: ...
@overload
def nanmin(arr: NDArray[_ScalarT], *args, axis: None = None) -> _ScalarT: ...
@overload
def nanmin(
    arr: NDArray[_ScalarT], *args, axis: AxisLike
) -> _ScalarT | NDArray[_ScalarT]: ...
@overload
def nanmin(arr: Any, *args, axis: AxisLike | None = None) -> Any: ...

# `**kwargs` reaches the gufunc for these two as well, so `out=` fixes the result
# dtype: `nanmedian(a, axis=-1, out=<int64 array>, casting="unsafe")` gives back int64,
# and a boolean `out` likewise. The shape doesn't carry through either — the quantile
# axis is moved to the front, so the return is a reshaped view of `out`, not `out`
# itself. Only "some ndarray" holds for every call form they accept, which is what the
# matrix declarations below say too.
def nanquantile(
    a: NDArray[np.float64],
    quantiles: float | Iterable[float],
    axis: AxisLike | None = None,
    **kwargs,
) -> np.ndarray: ...
def nanmedian(
    a: NDArray[np.float64], *, axis: AxisLike | None = None, **kwargs
) -> np.ndarray: ...
def bfill(
    arr: _T,
    *,
    limit: int | None = None,
    axis: AxisLike = -1,
) -> _T: ...
def ffill(
    arr: _T,
    *,
    limit: int | None = None,
    axis: AxisLike = -1,
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
