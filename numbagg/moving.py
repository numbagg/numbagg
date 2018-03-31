import numpy as np
from numba import float64, int64

from .decorators import ndmoving


@ndmoving([
    (float64[:], int64, float64[:]),
])
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
