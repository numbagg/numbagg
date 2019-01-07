import numpy as np
from numba import float64, int64

from .decorators import ndmoving


def ewm_window_validator(arr, window):
    if window < 0:
        raise ValueError("Com must be positive; currently {}".format(window))


@ndmoving([(float64[:], float64, float64[:])], window_validator=ewm_window_validator)
def ewm_nanmean(a, com, out):

    N = len(a)
    if N == 0:
        return

    alpha = 1.0 / (1.0 + com)
    old_wt_factor = 1.0 - alpha
    new_wt = 1.0

    weighted_avg = a[0]
    is_observation = weighted_avg == weighted_avg
    nobs = int(is_observation)
    out[0] = weighted_avg
    old_wt = 1.0

    for i in range(1, N):
        cur = a[i]
        is_observation = cur == cur
        nobs += int(is_observation)
        if weighted_avg == weighted_avg:
            if is_observation:
                old_wt *= old_wt_factor

                # avoid numerical errors on constant series
                if weighted_avg != cur:
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * cur)) / (
                        old_wt + new_wt
                    )
                old_wt += new_wt
        elif is_observation:
            weighted_avg = cur

        out[i] = weighted_avg


@ndmoving([(float64[:], int64, float64[:])])
def move_nanmean(a, window, out):

    asum = 0.0
    count = 0
    for i in range(window - 1):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
        out[i] = np.nan
    i = window - 1
    ai = a[i]
    if ai == ai:
        asum += ai
        count += 1
    if count > 0:
        out[i] = asum / count
    else:
        out[i] = np.nan
    for i in range(window, len(a)):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
        aold = a[i - window]
        if aold == aold:
            asum -= aold
            count -= 1
        if count > 0:
            out[i] = asum / count
        else:
            out[i] = np.nan
