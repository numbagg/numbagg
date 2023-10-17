import numpy as np
from numba import bool_, float32, float64, guvectorize, int32, int64

from numbagg.decorators import ndreduce


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
    # for now, fix ddof=0
    ddof = 0
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
    # for now, fix ddof=0
    ddof = 0
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


@ndreduce([int64(int32), int64(int64), int64(float32), int64(float64)])
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


@ndreduce([int64(int32), int64(int64), int64(float32), int64(float64)])
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


@ndreduce([int64(int32), int64(int64), float32(float32), float64(float64)])
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


@ndreduce([int64(int32), int64(int64), float32(float32), float64(float64)])
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


@guvectorize([(float64[:], float64, float64[:])], "(n),()->()")
def nanquantile_single(arr, quantile, out):
    # valid (non NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr))
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val

    rank = (valid_obs - 1) * quantile
    floor_index = int(np.floor(rank))
    col_index = int(np.ceil(rank))
    sorted = np.partition(arr, kth=[floor_index, col_index])

    # linear interpolation (like numpy percentile) takes the fractional part of desired position
    floor_val = sorted[floor_index]
    ceil_val = sorted[col_index]

    proportion = rank - np.floor(rank)

    result = floor_val + proportion * (ceil_val - floor_val)

    out[0] = result


nanquantile = nanquantile_single


# WIP (which doesn't work) on multiple quantiles
# @guvectorize([(float64[:], float64[:, :], float64[:])], "(n),(n,m)->(m)")
@guvectorize([(float64[:], float64[:], float64[:])], "(n),()->()")
def nanquantile_multiple(arr, quantiles, out):
    # valid (non NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr))
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    # TODO: would `argsort` or `partition` be faster here?
    arr = np.sort(arr)

    for i in range(len(quantiles)):
        quant = quantiles[i]
        # desired position as well as floor and ceiling of it
        rank = (valid_obs - 1) * quant

        # linear interpolation (like numpy percentile) takes the fractional part of desired position
        floor_val = arr[int(np.floor(rank))]
        ceil_val = arr[int(np.ceil(rank))]

        proportion = rank - np.floor(rank)

        result = floor_val + proportion * (ceil_val - floor_val)

        # quant_arr = floor_val + ceil_val
        # quant_arr[fc_equal_k_mask] = _zvalue_from_index(
        #     arr=arr, ind=rank.astype(np.int32)
        # )[
        #     fc_equal_k_mask
        # ]  # if floor == ceiling take floor value

        # out[i] = result
        out[0] = result


count = nancount
