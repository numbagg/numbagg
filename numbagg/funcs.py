from __future__ import annotations

from typing import TypeVar

import numpy as np
from numba import bool_, float32, float64, int32, int64
from numpy.typing import NDArray

from numbagg.decorators import (
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
    for ai in a.flat:
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
    for ai in a.flat:
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
    for ai in a.flat:
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
    for ai in a.flat:
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
    # ddof = 1
    for ai in a:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > ddof:
        amean = asum / count
        asum = 0
        for ai in a.flat:
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
        for ai in a.flat:
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
    """
    Backward fill missing values.
    For floats, fills NaN values. For integers, returns unchanged (no NaN possible).
    """
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
    """
    Forward fill missing values.
    For floats, fills NaN values. For integers, returns unchanged (no NaN possible).
    """
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


def nanmedian(a: NDArray[np.float64], **kwargs) -> NDArray[np.float64]:
    return nanquantile(a, quantiles=0.5, **kwargs)


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

    # Allocate arrays for all pairs - optimized for cache locality
    sums_i = np.zeros((n_vars, n_vars), dtype=a.dtype)
    sums_j = np.zeros((n_vars, n_vars), dtype=a.dtype)
    sums_sq_i = np.zeros((n_vars, n_vars), dtype=a.dtype)
    sums_sq_j = np.zeros((n_vars, n_vars), dtype=a.dtype)
    sums_ij = np.zeros((n_vars, n_vars), dtype=a.dtype)
    counts = np.zeros((n_vars, n_vars), dtype=np.int64)

    # Single pass through observations (excellent cache locality)
    for k in range(n_obs):
        # Load entire observation into cache once
        obs = a[:, k]

        # Process all variable pairs for this observation
        for i in range(n_vars):
            val_i = obs[i]
            if not np.isnan(val_i):
                for j in range(i, n_vars):  # Only upper triangle
                    val_j = obs[j]
                    if not np.isnan(val_j):
                        sums_i[i, j] += val_i
                        sums_j[i, j] += val_j
                        sums_sq_i[i, j] += val_i * val_i
                        sums_sq_j[i, j] += val_j * val_j
                        sums_ij[i, j] += val_i * val_j
                        counts[i, j] += 1

    # Compute final correlations from accumulated statistics
    for i in range(n_vars):
        for j in range(i, n_vars):
            count = counts[i, j]
            if count > 1:  # Need at least 2 observations for correlation
                mean_i = sums_i[i, j] / count
                mean_j = sums_j[i, j] / count

                # Variances (sample variance with ddof=1)
                var_i = (sums_sq_i[i, j] / count) - (mean_i * mean_i)
                var_j = (sums_sq_j[i, j] / count) - (mean_j * mean_j)
                var_i_unbiased = var_i * count / (count - 1)
                var_j_unbiased = var_j * count / (count - 1)

                # Covariance (sample covariance with ddof=1)
                cov = (sums_ij[i, j] / count) - (mean_i * mean_j)
                cov_unbiased = cov * count / (count - 1)

                # Correlation
                if var_i_unbiased > 0 and var_j_unbiased > 0:
                    corr = cov_unbiased / np.sqrt(var_i_unbiased * var_j_unbiased)
                    out[i, j] = corr
                    out[j, i] = corr  # Symmetric
                else:
                    out[i, j] = np.nan
                    out[j, i] = np.nan
            else:
                out[i, j] = np.nan
                out[j, i] = np.nan


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

    # Allocate arrays for all pairs - optimized for cache locality
    sums_i = np.zeros((n_vars, n_vars), dtype=a.dtype)
    sums_j = np.zeros((n_vars, n_vars), dtype=a.dtype)
    sums_ij = np.zeros((n_vars, n_vars), dtype=a.dtype)
    counts = np.zeros((n_vars, n_vars), dtype=np.int64)

    # Single pass through observations (excellent cache locality)
    for k in range(n_obs):
        # Load entire observation into cache once
        obs = a[:, k]

        # Process all variable pairs for this observation
        for i in range(n_vars):
            val_i = obs[i]
            if not np.isnan(val_i):
                for j in range(i, n_vars):  # Only upper triangle
                    val_j = obs[j]
                    if not np.isnan(val_j):
                        sums_i[i, j] += val_i
                        sums_j[i, j] += val_j
                        sums_ij[i, j] += val_i * val_j
                        counts[i, j] += 1

    # Compute final covariances from accumulated statistics
    for i in range(n_vars):
        for j in range(i, n_vars):
            count = counts[i, j]
            if count > 1:
                mean_i = sums_i[i, j] / count
                mean_j = sums_j[i, j] / count
                # Covariance: E[XY] - E[X]E[Y], scaled to unbiased estimator
                cov = (sums_ij[i, j] / count) - (mean_i * mean_j)
                cov_unbiased = cov * count / (count - 1)
                out[i, j] = cov_unbiased
                out[j, i] = cov_unbiased  # Symmetric
            else:
                out[i, j] = np.nan
                out[j, i] = np.nan
