from __future__ import annotations

import numpy as np
from numba import bool_, float32, float64, int32, int64

from numbagg.decorators import ndfill, ndquantile, ndreduce


@ndreduce.wrap([bool_(int32), bool_(int64), bool_(float32), bool_(float64)])
def allnan(a):
    f = True
    for ai in a.flat:
        if not np.isnan(ai):
            f = False
            break
    return f


@ndreduce.wrap([bool_(int32), bool_(int64), bool_(float32), bool_(float64)])
def anynan(a):
    f = False
    for ai in a.flat:
        if np.isnan(ai):
            f = True
            break
    return f


@ndreduce.wrap([int64(int32), int64(int64), int64(float32), int64(float64)])
def nancount(a):
    non_missing = 0
    for ai in a.flat:
        if not np.isnan(ai):
            non_missing += 1
    return non_missing


@ndreduce.wrap([int32(int32), int64(int64), float32(float32), float64(float64)])
def nansum(a):
    asum = 0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
    return asum


@ndreduce.wrap([float32(float32), float64(float64)])
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


@ndreduce.wrap([float32(float32), float64(float64)])
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


@ndreduce.wrap([float32(float32), float64(float64)])
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


@ndreduce.wrap(
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


@ndreduce.wrap(
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


@ndreduce.wrap(
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


@ndreduce.wrap(
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


@ndquantile.wrap(([(float64[:], float64[:], float64[:])], "(n),(m)->(m)"))
def nanquantile(arr, quantile, out):
    # valid (non NaN) observations
    valid_obs = np.sum(np.isfinite(arr))

    if valid_obs == 0:
        out[:] = np.nan
        return

    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val

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


@ndfill.wrap()
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
