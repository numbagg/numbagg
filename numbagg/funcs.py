from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numba import bool_, float32, float64, guvectorize, int32, int64

from numbagg.decorators import ndfill, ndreduce


@ndreduce([bool_(int32), bool_(int64), bool_(float32), bool_(float64)])
def allnan(a):
    f = True
    for ai in a.flat:
        if not np.isnan(ai):
            f = False
            break
    return f


@ndreduce([bool_(int32), bool_(int64), bool_(float32), bool_(float64)])
def anynan(a):
    f = False
    for ai in a.flat:
        if np.isnan(ai):
            f = True
            break
    return f


@ndreduce([int64(int32), int64(int64), int64(float32), int64(float64)])
def nancount(a):
    non_missing = 0
    for ai in a.flat:
        if not np.isnan(ai):
            non_missing += 1
    return non_missing


@ndreduce([int32(int32), int64(int64), float32(float32), float64(float64)])
def nansum(a):
    asum = 0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
    return asum


@ndreduce([float32(float32), float64(float64)])
def nanmean(a):
    asum = 0.0
    count = 0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > 0:
        return asum / count
    else:
        return np.nan


@ndreduce([float32(float32), float64(float64)])
def nanstd(a):
    # for now, fix ddof=1. See https://github.com/numbagg/numbagg/issues/138 for
    # discussion of whether to add an option.
    ddof = 1
    asum = 0
    count = 0
    for ai in a.flat:
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
        return np.sqrt(asum / (count - ddof))
    else:
        return np.nan


@ndreduce([float32(float32), float64(float64)])
def nanvar(a):
    # for now, fix ddof=1. See https://github.com/numbagg/numbagg/issues/138 for
    # discussion of whether to add an option.
    ddof = 1
    asum = 0
    count = 0
    for ai in a.flat:
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
        return asum / (count - ddof)
    else:
        return np.nan


@ndreduce(
    [int64(int32), int64(int64), int64(float32), int64(float64)],
    supports_parallel=False,
)
def nanargmax(a):
    if not a.size:
        raise ValueError("All-NaN slice encountered")
    amax = -np.infty
    idx = -1
    for i, ai in enumerate(a.flat):
        if ai > amax or (idx == -1 and not np.isnan(ai)):
            amax = ai
            idx = i
    if idx == -1:
        raise ValueError("All-NaN slice encountered")
    return idx


@ndreduce(
    [int64(int32), int64(int64), int64(float32), int64(float64)],
    supports_parallel=False,
)
def nanargmin(a):
    if not a.size:
        raise ValueError("All-NaN slice encountered")
    amin = np.infty
    idx = -1
    for i, ai in enumerate(a.flat):
        if ai < amin or (idx == -1 and not np.isnan(ai)):
            amin = ai
            idx = i
    if idx == -1:
        raise ValueError("All-NaN slice encountered")
    return idx


@ndreduce(
    [int64(int32), int64(int64), float32(float32), float64(float64)],
    supports_parallel=False,
)
def nanmax(a):
    if not a.size:
        raise ValueError(
            "zero-size array to reduction operation fmax which has no identity"
        )
    amax = -np.infty
    all_missing = 1
    for ai in a.flat:
        if ai >= amax:
            amax = ai
            all_missing = 0
    if all_missing:
        amax = np.nan
    return amax


@ndreduce(
    [int64(int32), int64(int64), float32(float32), float64(float64)],
    supports_parallel=False,
)
def nanmin(a):
    if not a.size:
        raise ValueError(
            "zero-size array to reduction operation fmin which has no identity"
        )
    amin = np.infty
    all_missing = 1
    for ai in a.flat:
        if ai <= amin:
            amin = ai
            all_missing = 0
    if all_missing:
        amin = np.nan
    return amin


@guvectorize([(float64[:], float64[:], float64[:])], "(n),(m)->(m)")
def nanquantile_(arr, quantile, out):
    # valid (non NaN) observations
    valid_obs = np.sum(np.isfinite(arr))
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val

    # two columns for indexes — floor and ceiling
    indexes = np.zeros((len(quantile), 2), dtype=np.int32)
    # store ranks as floats
    ranks = np.zeros(len(quantile), dtype=np.float64)

    for i in range(len(quantile)):
        rank = (valid_obs - 1) * quantile[i]
        ranks[i] = rank
        indexes[i, 0] = int(np.floor(rank))
        indexes[i, 1] = int(np.ceil(rank))

    # partition sorts but only ensures indexes passed to kth are in the correct positions
    unique_indices = np.unique(indexes)
    sorted = np.partition(arr, kth=unique_indices)

    for i in range(len(quantile)):
        # linear interpolation (like numpy percentile) takes the fractional part of
        # desired position
        proportion = ranks[i] - indexes[i, 0]

        floor_val = sorted[indexes[i, 0]]
        ceil_val = sorted[indexes[i, 1]]

        result = floor_val + proportion * (ceil_val - floor_val)

        out[i] = result


def move_axes(arr: np.ndarray, axes: tuple[int, ...]):
    """
    Move & reshape a tuple of axes to an array's final axis.
    """
    moved_arr = np.moveaxis(arr, axes, range(arr.ndim - len(axes), arr.ndim))
    new_shape = moved_arr.shape[: -len(axes)] + (-1,)
    return moved_arr.reshape(new_shape)


def nanquantile(
    a: np.ndarray,
    quantiles: float | Iterable[float],
    axis: int | tuple[int, ...] | None = None,
    **kwargs,
):
    # Gufunc doesn't support a 0-len dimension for quantiles, so we need to make and
    # then remove a dummy axis.
    if not isinstance(quantiles, Iterable):
        squeeze = True
        quantiles = [quantiles]
    else:
        squeeze = False
    quantiles = np.asarray(quantiles)

    if axis is None:
        axis = tuple(range(a.ndim))
    elif not isinstance(axis, Iterable):
        axis = (axis,)

    a = move_axes(a, axis)

    # - 1st array is our input; we've moved the axes to the final axis.
    # - 2nd is the quantiles array, and is always only a single axis.
    # - 3rd array is the result array, and returns a final axis for quantiles.
    axes = [-1, -1, -1]

    result = nanquantile_(a, quantiles, axes=axes, **kwargs)

    # numpy returns quantiles as the first axis, so we move ours to that position too
    result = np.moveaxis(result, -1, 0)
    if squeeze:
        result = result.squeeze(axis=0)
    return result


@ndfill()
def bfill(a, limit, out):
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


@ndfill()
def ffill(a, limit, out):
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
