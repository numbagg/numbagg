from __future__ import annotations

from typing import TypeVar

import numpy as np
from numba import bool_, float32, float64, int32, int64, njit
from numpy.typing import NDArray

from numbagg.decorators import (
    _ENABLE_CACHE,
    ndaggregate,
    ndfill,
    ndmatrix,
    ndquantile,
    ndreduce,
)

from .utils import FloatArray, NumericArray

T = TypeVar("T", bound=NumericArray)
F = TypeVar("F", bound=FloatArray)


@ndaggregate.wrap(
    signature=[
        (int32[:], bool_[:]),
        (int64[:], bool_[:]),
        (float32[:], bool_[:]),
        (float64[:], bool_[:]),
    ]
)
def allnan(a: NumericArray, out: NumericArray) -> None:
    out[0] = True
    for ai in a:
        if not np.isnan(ai):
            out[0] = False
            return


@ndaggregate.wrap(
    signature=[
        (int32[:], bool_[:]),
        (int64[:], bool_[:]),
        (float32[:], bool_[:]),
        (float64[:], bool_[:]),
    ]
)
def anynan(a: NumericArray, out: NumericArray) -> None:
    out[0] = False
    for ai in a:
        if np.isnan(ai):
            out[0] = True
            return


@ndaggregate.wrap(
    signature=[
        (int32[:], int64[:]),
        (int64[:], int64[:]),
        (float32[:], int64[:]),
        (float64[:], int64[:]),
    ]
)
def nancount(a: T, out: T) -> None:
    non_missing = 0
    for ai in a:
        if not np.isnan(ai):
            non_missing += 1
    out[0] = non_missing


@ndaggregate.wrap(
    signature=[
        (int32[:], int32[:]),
        (int64[:], int64[:]),
        (float32[:], float32[:]),
        (float64[:], float64[:]),
    ]
)
def nansum(a, out):
    asum = a.dtype.type(0)
    for ai in a:
        if not np.isnan(ai):
            asum += ai
    out[0] = asum


@ndaggregate.wrap(
    signature=[
        (float32[:], float32[:]),
        (float64[:], float64[:]),
    ]
)
def nanmean(a, out):
    asum = 0.0
    count = 0
    for ai in a:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > 0:
        out[0] = asum / count
    else:
        out[0] = np.nan


@ndaggregate.wrap(
    signature=[
        (float32[:], int32, float32[:]),
        (float64[:], int64, float64[:]),
    ],
    supports_ddof=True,
)
def nanvar(a: F, ddof: int, out: F) -> None:
    # Running two loops might seem inefficient, but it's 3x faster than a Welford's
    # algorithm. And if we don't compute the mean first, we get numerical instability
    # (which our tests capture so is easy to observe).

    asum = 0
    count = 0
    for ai in a:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > ddof:
        amean = asum / count
        asum = 0
        for ai in a:
            if not np.isnan(ai):
                ai -= amean
                asum += ai * ai
        out[0] = asum / (count - ddof)
    else:
        out[0] = np.nan


@ndaggregate.wrap(
    signature=[
        (float32[:], int32, float32[:]),
        (float64[:], int64, float64[:]),
    ],
    supports_ddof=True,
)
def nanstd(a: F, ddof: int, out: F) -> None:
    asum = 0
    count = 0
    for ai in a:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > ddof:
        amean = asum / count
        asum = 0
        for ai in a:
            if not np.isnan(ai):
                ai -= amean
                asum += ai * ai
        out[0] = np.sqrt(asum / (count - ddof))
    else:
        out[0] = np.nan


@ndreduce.wrap(
    [int64(int32), int64(int64), int64(float32), int64(float64)],
    # https://github.com/numba/numba/issues/7350
    supports_parallel=False,
)
def nanargmax(a):
    if not a.size:
        raise ValueError("All-NaN slice encountered")
    amax = -np.inf
    idx = -1
    for i, ai in enumerate(a.flat):
        # Much slower, by 3-4x to use this construction:
        # if not np.isnan(ai) and (ai > ammax or idx == -1):
        if ai > amax or (idx == -1 and not np.isnan(ai)):
            amax = ai
            idx = i
    if idx == -1:
        raise ValueError("All-NaN slice encountered")
    return idx


@ndreduce.wrap(
    [int64(int32), int64(int64), int64(float32), int64(float64)],
    # https://github.com/numba/numba/issues/7350
    supports_parallel=False,
)
def nanargmin(a):
    if not a.size:
        raise ValueError("All-NaN slice encountered")
    amin = np.inf
    idx = -1
    for i, ai in enumerate(a.flat):
        if ai < amin or (idx == -1 and not np.isnan(ai)):
            amin = ai
            idx = i
    if idx == -1:
        raise ValueError("All-NaN slice encountered")
    return idx


@ndreduce.wrap(
    [int64(int32), int64(int64), float32(float32), float64(float64)],
    # https://github.com/numba/numba/issues/7350
    supports_parallel=False,
)
def nanmax(a):
    if not a.size:
        raise ValueError(
            "zero-size array to reduction operation fmax which has no identity"
        )
    amax = -np.inf
    all_missing = True
    for ai in a.flat:
        # If we check for `isnan` here, the function becomes much slower (by about 4x!)
        if ai >= amax:
            amax = ai
            all_missing = False
    if all_missing:
        amax = np.nan
    return amax


@ndreduce.wrap(
    [int64(int32), int64(int64), float32(float32), float64(float64)],
    # https://github.com/numba/numba/issues/7350
    supports_parallel=False,
)
def nanmin(a):
    if not a.size:
        raise ValueError(
            "zero-size array to reduction operation fmin which has no identity"
        )
    amin = np.inf
    all_missing = True
    for ai in a.flat:
        if ai <= amin:
            amin = ai
            all_missing = False
    if all_missing:
        amin = np.nan
    return amin


@ndquantile.wrap(([(float64[:], float64[:], float64[:])], "(n),(m)->(m)"))
def nanquantile(
    arr: NDArray[np.float64], quantile: NDArray[np.float64], out: NDArray[np.float64]
) -> None:
    nans = np.isnan(arr)
    valid_obs = arr.size - np.sum(nans)

    if valid_obs == 0:
        out[:] = np.nan
        return

    # replace NaN with maximum
    max_val = np.nanmax(arr)

    # and we need to use `where` to avoid modifying the original array
    arr = np.where(nans, max_val, arr)

    # two columns for indexes — floor and ceiling
    indexes = np.zeros((len(quantile), 2), dtype=np.int32)
    # store ranks as floats
    ranks = np.zeros(len(quantile), dtype=np.float64)

    for i in range(len(quantile)):
        if np.isnan(quantile[i]):
            continue
        rank = (valid_obs - 1) * quantile[i]
        ranks[i] = rank
        indexes[i] = [int(np.floor(rank)), int(np.ceil(rank))]

    # `partition` is similar to a `sort`, but only ensures that the indexes passed to
    # kth are in the correct positions
    unique_indices = np.unique(indexes)
    sorted = np.partition(arr, kth=unique_indices)

    for i in range(len(quantile)):
        if np.isnan(quantile[i]):
            out[i] = np.nan
            continue
        # linear interpolation (like numpy percentile) takes the fractional part of
        # desired position
        proportion = ranks[i] - indexes[i, 0]

        floor_val, ceil_val = sorted[indexes[i]]

        result = floor_val + proportion * (ceil_val - floor_val)

        out[i] = result


@ndfill.wrap()
def bfill(a: T, limit: int, out: T) -> None:
    """Backward fill missing values."""
    lives_remaining = limit
    current = np.nan
    # Ugly `range` expression, but can't do 'enumerate(reversed(a))', and adding a
    # `list` will cause a copy.
    for i in range(len(a) - 1, -1, -1):
        val = a[i]
        if np.isnan(val):  # Always False for integers, True for float NaN
            if lives_remaining <= 0:
                current = np.nan
            lives_remaining -= 1
        else:
            lives_remaining = limit
            current = val
        out[i] = current


@ndfill.wrap()
def ffill(a: T, limit: int, out: T) -> None:
    """Forward fill missing values."""
    lives_remaining = limit
    current = np.nan
    for i, val in enumerate(a):
        if np.isnan(val):  # Always False for integers, True for float NaN
            if lives_remaining <= 0:
                current = np.nan
            lives_remaining -= 1
        else:
            lives_remaining = limit
            current = val
        out[i] = current


count = nancount


def nanmedian(
    a: NDArray[np.float64], *, axis: int | tuple[int, ...] | None = None, **kwargs
) -> NDArray[np.float64]:
    return nanquantile(a, quantiles=0.5, axis=axis, **kwargs)


# The packed matrix kernels accumulate in float64, including for float32 input.
# Raw moments subtract terms close to `scale`. Use 64 times machine epsilon as
# an empirical trigger for a shifted rescan, not as a forward-error bound for
# the accumulated sums.
_STATIC_MOMENT_FALLBACK_EPSILON = 64.0 * np.finfo(np.float64).eps

# Recompute a pair in shifted coordinates if the roundoff estimate exceeds the
# accuracy target for the output dtype.
_STATIC_FLOAT32_ACCURACY = 1e-4
_STATIC_FLOAT64_ACCURACY = 1e-8


@njit(cache=_ENABLE_CACHE)
def _nancorr_pair_stable(a, i, j):
    """Return one pair's shifted float64 correlation."""
    n_obs = a.shape[1]
    count = 0
    shift_i = 0.0
    shift_j = 0.0
    sum_i = 0.0
    sum_j = 0.0
    sum_sq_i = 0.0
    sum_sq_j = 0.0
    sum_ij = 0.0
    for k in range(n_obs):
        value_i = np.float64(a[i, k])
        value_j = np.float64(a[j, k])
        if np.isnan(value_i) or np.isnan(value_j):
            continue
        if count == 0:
            shift_i = value_i
            shift_j = value_j
        shifted_i = value_i - shift_i
        shifted_j = value_j - shift_j
        sum_i += shifted_i
        sum_j += shifted_j
        sum_sq_i += shifted_i * shifted_i
        sum_sq_j += shifted_j * shifted_j
        sum_ij += shifted_i * shifted_j
        count += 1

    value = np.nan
    if count > 1:
        centered_i = sum_sq_i - sum_i * sum_i / count
        centered_j = sum_sq_j - sum_j * sum_j / count
        denominator = np.sqrt(centered_i) * np.sqrt(centered_j)
        if denominator > 0.0:
            if i == j:
                value = 1.0
            elif count == 2:
                centered_ij = sum_ij - sum_i * sum_j / count
                if centered_ij > 0.0:
                    value = 1.0
                elif centered_ij < 0.0:
                    value = -1.0
            else:
                value = (sum_ij - sum_i * sum_j / count) / denominator
                if value > 1.0:
                    value = 1.0
                elif value < -1.0:
                    value = -1.0
    return value


@njit(cache=_ENABLE_CACHE)
def _nancov_pair_stable(a, i, j):
    """Return one pair's shifted float64 covariance."""
    n_obs = a.shape[1]
    count = 0
    shift_i = 0.0
    shift_j = 0.0
    sum_i = 0.0
    sum_j = 0.0
    sum_ij = 0.0
    for k in range(n_obs):
        value_i = np.float64(a[i, k])
        value_j = np.float64(a[j, k])
        if np.isnan(value_i) or np.isnan(value_j):
            continue
        if count == 0:
            shift_i = value_i
            shift_j = value_j
        shifted_i = value_i - shift_i
        shifted_j = value_j - shift_j
        sum_i += shifted_i
        sum_j += shifted_j
        sum_ij += shifted_i * shifted_j
        count += 1

    if count > 1:
        return (sum_ij - sum_i * sum_j / count) / (count - 1)
    return np.nan


@ndmatrix.wrap(
    signature=(
        [(float32[:, :], float32[:, :]), (float64[:, :], float64[:, :])],
        "(n,m)->(n,n)",
    )
)
def nancorrmatrix(a: F, out: F) -> None:
    """
    Compute correlation matrix treating NaN as missing values.

    Matrix Function Dimensional Conventions:

    Due to NumPy gufunc constraints, matrix functions have fixed axis assignments:

    Static Matrix Functions (nancorrmatrix, nancovmatrix):
    - vars_axis: -2 (variables dimension gets duplicated into n×n matrix)
    - obs_axis: -1 (observations dimension gets reduced)
    - Input signature: (..., vars, obs) -> (..., vars, vars)

    Moving Matrix Functions (move_corrmatrix, etc.):
    - obs_axis: -2 (observations dimension preserved as time axis)
    - vars_axis: -1 (variables dimension duplicated to end as matrix dims)
    - Input signature: (..., obs, vars) -> (..., obs, vars, vars)

    This asymmetry exists because:
    - Static: gufunc "(vars,obs)->(vars,vars)" needs obs at end to reduce
    - Moving: gufunc "(obs,vars)->(obs,vars,vars)" needs vars at end to add matrix dims

    Parameters
    ----------
    a : array_like
        Input array with shape (..., vars, obs) where:
        - vars (axis=-2): variables to compute correlations between
        - obs (axis=-1): observations to aggregate over (gets reduced)

    Returns
    -------
    ndarray
        Shape (..., vars, vars) - correlation matrix with same leading
        dimensions as input, plus vars×vars correlation matrix at the end.

    Examples
    --------
    >>> import numpy as np
    >>> import numbagg as nb
    >>> # Standard: 3 variables, 100 observations
    >>> data = np.random.randn(3, 100)
    >>> corr = nb.nancorrmatrix(data)
    >>> corr.shape
    (3, 3)
    >>>
    >>> # Broadcasting: batch of correlation matrices
    >>> data_3d = np.random.randn(5, 3, 100)
    >>> corr_3d = nb.nancorrmatrix(data_3d)
    >>> corr_3d.shape
    (5, 3, 3)
    >>>
    >>> # Wrong arrangement: transpose first
    >>> data_wrong = np.random.randn(100, 3)  # obs, vars
    >>> data_correct = data_wrong.T  # vars, obs
    >>> corr = nb.nancorrmatrix(data_correct)

    Notes
    -----
    - Uses pairwise complete observations (like pandas.DataFrame.corr)
    - Cache-friendly implementation: processes observations sequentially for better locality
    - Unlike NumPy's corrcoef, this broadcasts over arbitrary leading dimensions
    - For other dimension arrangements, transpose your data first
    - axis parameter removed - dimensions are now fixed for consistency
    """
    n_vars, n_obs = a.shape
    accuracy_tolerance = (
        _STATIC_FLOAT32_ACCURACY if a.itemsize == 4 else _STATIC_FLOAT64_ACCURACY
    )
    n_pairs = n_vars * (n_vars + 1) // 2
    sums_i = np.zeros(n_pairs, dtype=np.float64)
    sums_j = np.zeros(n_pairs, dtype=np.float64)
    sums_sq_i = np.zeros(n_pairs, dtype=np.float64)
    sums_sq_j = np.zeros(n_pairs, dtype=np.float64)
    sums_ij = np.zeros(n_pairs, dtype=np.float64)
    counts = np.zeros(n_pairs, dtype=np.int64)

    for k in range(n_obs):
        obs = a[:, k]
        pair = 0
        for i in range(n_vars):
            value_i = np.float64(obs[i])
            if not np.isnan(value_i):
                for j in range(i, n_vars):
                    value_j = np.float64(obs[j])
                    if not np.isnan(value_j):
                        sums_i[pair] += value_i
                        sums_j[pair] += value_j
                        sums_sq_i[pair] += value_i * value_i
                        sums_sq_j[pair] += value_j * value_j
                        sums_ij[pair] += value_i * value_j
                        counts[pair] += 1
                    pair += 1
            else:
                pair += n_vars - i

    pair = 0
    for i in range(n_vars):
        for j in range(i, n_vars):
            value = np.nan
            count = counts[pair]
            if count > 1:
                mean_i = sums_i[pair] / count
                mean_j = sums_j[pair] / count
                centered_i = sums_sq_i[pair] - mean_i * mean_i * count
                centered_j = sums_sq_j[pair] - mean_j * mean_j * count
                centered_ij = sums_ij[pair] - mean_i * mean_j * count
                scale_i = abs(sums_sq_i[pair]) + abs(mean_i * mean_i * count)
                scale_j = abs(sums_sq_j[pair]) + abs(mean_j * mean_j * count)
                scale_ij = abs(sums_ij[pair]) + abs(mean_i * mean_j * count)
                denominator = np.sqrt(centered_i) * np.sqrt(centered_j)
                requires_stable = (
                    not np.isfinite(centered_i)
                    or not np.isfinite(centered_j)
                    or _STATIC_MOMENT_FALLBACK_EPSILON * scale_i
                    > accuracy_tolerance * abs(centered_i)
                    or _STATIC_MOMENT_FALLBACK_EPSILON * scale_j
                    > accuracy_tolerance * abs(centered_j)
                    or _STATIC_MOMENT_FALLBACK_EPSILON * scale_ij
                    > accuracy_tolerance * np.sqrt(abs(centered_i * centered_j))
                )
                if requires_stable:
                    value = _nancorr_pair_stable(a, i, j)
                elif denominator > 0.0:
                    if i == j:
                        value = 1.0
                    elif count == 2:
                        if centered_ij > 0.0:
                            value = 1.0
                        elif centered_ij < 0.0:
                            value = -1.0
                    else:
                        value = centered_ij / denominator
                        if value > 1.0:
                            value = 1.0
                        elif value < -1.0:
                            value = -1.0
            out[i, j] = value
            out[j, i] = value
            pair += 1


@ndmatrix.wrap(
    signature=(
        [(float32[:, :], float32[:, :]), (float64[:, :], float64[:, :])],
        "(n,m)->(n,n)",
    )
)
def nancovmatrix(a: F, out: F) -> None:
    """
    Compute covariance matrix treating NaN as missing values.

    Dimension conventions:
    - Input: (n_vars, n_obs) - variables as rows, observations as columns
    - Output: (n_vars, n_vars) - square covariance matrix
    - Broadcasting: Supports arbitrary leading dimensions via NumPy's gufunc system

    Uses pairwise complete observations (like pandas.DataFrame.cov).
    Cache-friendly implementation: processes observations sequentially for better locality.

    Unlike NumPy's cov, this function broadcasts over higher dimensions:

    Examples:
    - 2D: (3, 100) -> (3, 3)
    - 3D: (batch=5, vars=3, obs=100) -> (5, 3, 3)
    - 4D: (2, 5, 3, 100) -> (2, 5, 3, 3)
    """
    n_vars, n_obs = a.shape
    accuracy_tolerance = (
        _STATIC_FLOAT32_ACCURACY if a.itemsize == 4 else _STATIC_FLOAT64_ACCURACY
    )
    n_pairs = n_vars * (n_vars + 1) // 2
    sums_i = np.zeros(n_pairs, dtype=np.float64)
    sums_j = np.zeros(n_pairs, dtype=np.float64)
    sums_ij = np.zeros(n_pairs, dtype=np.float64)
    counts = np.zeros(n_pairs, dtype=np.int64)

    for k in range(n_obs):
        obs = a[:, k]
        pair = 0
        for i in range(n_vars):
            value_i = np.float64(obs[i])
            if not np.isnan(value_i):
                for j in range(i, n_vars):
                    value_j = np.float64(obs[j])
                    if not np.isnan(value_j):
                        sums_i[pair] += value_i
                        sums_j[pair] += value_j
                        sums_ij[pair] += value_i * value_j
                        counts[pair] += 1
                    pair += 1
            else:
                pair += n_vars - i

    pair = 0
    for i in range(n_vars):
        for j in range(i, n_vars):
            value = np.nan
            count = counts[pair]
            if count > 1:
                mean_i = sums_i[pair] / count
                mean_j = sums_j[pair] / count
                centered = sums_ij[pair] - mean_i * mean_j * count
                scale = abs(sums_ij[pair]) + abs(mean_i * mean_j * count)
                if not np.isfinite(
                    centered
                ) or _STATIC_MOMENT_FALLBACK_EPSILON * scale > accuracy_tolerance * abs(
                    centered
                ):
                    value = _nancov_pair_stable(a, i, j)
                else:
                    value = centered / (count - 1)
            out[i, j] = value
            out[j, i] = value
            pair += 1
