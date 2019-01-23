import numpy as np
from numba import float32, float64, int64, int32

from .decorators import ndmoving, ndmovingexp


@ndmovingexp([(float64[:], float64, float64[:]), (float32[:], float32, float32[:])])
def move_exp_nanmean(a, alpha, out):

    N = len(a)
    if N == 0:
        return

    old_wt_factor = 1.0 - alpha
    new_wt = 1.0
    ignore_na = False  # could add as option in the future

    weighted_avg = a[0]
    is_observation = not np.isnan(weighted_avg)
    nobs = int(is_observation)
    out[0] = weighted_avg
    old_wt = 1.0

    for i in range(1, N):
        cur = a[i]
        is_observation = not np.isnan(cur)
        nobs += int(is_observation)
        if not np.isnan(weighted_avg):
            if is_observation or (not ignore_na):
                old_wt *= old_wt_factor

                if is_observation:
                    # avoid numerical errors on constant series
                    if weighted_avg != cur:
                        weighted_avg = ((old_wt * weighted_avg) + (new_wt * cur)) / (
                            old_wt + new_wt
                        )
                    old_wt += new_wt
        elif is_observation:
            weighted_avg = cur

        out[i] = weighted_avg


@ndmoving(
    [(float64[:], int64, int64, float64[:]), (float32[:], int64, int64, float32[:])]
)
def move_mean(a, window, min_count, out):

    asum = 0.0
    count = 0

    for i in range(min_count - 1):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = np.nan

    for i in range(min_count - 1, window - 1):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = asum / count if count >= min_count else np.nan

    for i in range(window - 1, len(a)):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1

        aold = a[i - window]
        if not np.isnan(aold):
            asum -= aold
            count -= 1

        out[i] = asum / count if count >= min_count else np.nan
