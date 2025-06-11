from __future__ import annotations

import numpy as np
from numba import bool_, float32, float64, int32, int64
from numpy.typing import NDArray

from numbagg.decorators import ndaggregate, ndfill, ndquantile, ndreduce

from .utils import FloatArray, NumericArray, NumericScalar


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
def nancount[T: NumericArray](a: T, out: T) -> None:
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
def nansum[T: NumericArray](a: T, out: T) -> None:
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
def nanmean[T: FloatArray](a: T, out: T) -> None:
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
def nanvar[T: FloatArray](
    a: T,
    ddof: int,
    out: T,
) -> None:
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
def nanstd[T: FloatArray](a: T, ddof: int, out: T) -> None:
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
def nanargmax(a: NumericScalar) -> int:
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
def nanargmin(a: NumericScalar) -> int:
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
def nanmax(a: NumericScalar):
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
def nanmin(a: NumericScalar):
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
def nanquantile(arr: NDArray[np.float64], quantile: NDArray[np.float64], out: NDArray[np.float64]) -> None:
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
def bfill[T: FloatArray](a: T, limit: int, out: T) -> None:
    lives_remaining = limit
    current = np.nan
    # Ugly `range` expression, but can't do 'enumerate(reversed(a))', and adding a
    # `list` will cause a copy.
    for i in range(len(a) - 1, -1, -1):
        val = a[i]
        if np.isnan(val):
            if lives_remaining <= 0:
                current = np.nan
            lives_remaining -= 1
        else:
            lives_remaining = limit
            current = val
        out[i] = current


@ndfill.wrap()
def ffill[T: FloatArray](a: T, limit: int, out: T) -> None:
    lives_remaining = limit
    current = np.nan
    for i, val in enumerate(a):
        if np.isnan(val):
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
