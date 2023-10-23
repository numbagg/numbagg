import numpy as np
from numba import float32, float64, int64

from .decorators import ndmoving


@ndmoving(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_mean(a, window, min_count, out):
    asum = 0.0
    count = 0

    # We previously had an initial loop which filled NaNs before `min_count`, but it
    # didn't have a discernible effect on performance.

    for i in range(window):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = asum / count if count >= min_count else np.nan

    for i in range(window, len(a)):
        ai = a[i]
        aold = a[i - window]

        ai_valid = not np.isnan(ai)
        aold_valid = not np.isnan(aold)

        if ai_valid and aold_valid:
            asum += ai - aold
        elif ai_valid:
            asum += ai
            count += 1
        elif aold_valid:
            asum -= aold
            count -= 1

        out[i] = asum / count if count >= min_count else np.nan


@ndmoving(
    [
        (float32[:], float32[:]),
        (float64[:], float64[:]),
    ]
)
def ffill(a, out):
    current = np.nan
    for i in range(len(a)):
        if not np.isnan(a[i]):
            current = a[i]
        out[i] = current
