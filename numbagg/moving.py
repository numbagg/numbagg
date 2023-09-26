import numpy as np
from numba import float32, float64, int64

from .decorators import ndmoving, ndmovingexp


@ndmovingexp(
    [
        (float32[:], float32, float32, float32[:]),
        (float64[:], float64, float64, float64[:]),
    ]
)
def move_exp_nanmean(a, alpha, min_weight, out):
    N = len(a)

    numer = denom = np.nan
    weight = 0
    decay = 1.0 - alpha

    for i in range(N):
        a_i = a[i]

        numer *= decay
        denom *= decay
        weight *= decay

        if not np.isnan(a_i):
            # If it's the first observation, toggle the values to non-nan.
            if np.isnan(numer):
                numer = denom = 0
            numer += a_i
            denom += 1
            weight += alpha

        if weight >= min_weight:
            out[i] = numer / denom
        else:
            out[i] = np.nan


@ndmovingexp(
    [
        (float32[:], float32, float32, float32[:]),
        (float64[:], float64, float64, float64[:]),
    ]
)
def move_exp_nansum(a, alpha, min_weight, out):
    """
    Calculates the exponentially decayed sum.

    Very similar to move_exp_nanmean, but calculates a decayed sum rather than the mean.
    """

    # As per https://github.com/numbagg/numbagg/issues/26#issuecomment-830437132, we
    # could try implementing this with a bool flag on the same function as
    # `move_exp_nanmean` and hope that numba optimizes it away. The code here is
    # basically a subset of that function.

    N = len(a)

    numer = np.nan
    weight = 0
    decay = 1.0 - alpha

    for i in range(N):
        a_i = a[i]

        numer *= decay
        weight *= decay

        if not np.isnan(a_i):
            # If it's the first observation, toggle the values to non-nan.
            if np.isnan(numer):
                numer = 0
            numer += a_i
            weight += alpha

        if weight >= min_weight:
            out[i] = numer
        else:
            out[i] = np.nan


@ndmovingexp(
    [
        (float32[:], float32, float32, float32[:]),
        (float64[:], float64, float64, float64[:]),
    ]
)
def move_exp_nanvar(a, alpha, min_weight, out):
    N = len(a)

    # sum_x: decayed sum of the sequence values.
    # sum_x2: decayed sum of the squared sequence values.
    # n: decayed count of non-missing values observed so far in the sequence.
    # n2: decayed sum of the (already-decayed) weights of non-missing values.
    sum_x2 = sum_x = sum_weight = sum_weight2 = weight = 0
    decay = 1.0 - alpha

    have_observed_value = False

    for i in range(N):
        a_i = a[i]

        if not np.isnan(a_i):
            have_observed_value = True

            sum_x2 += a_i**2
            sum_x += a_i
            sum_weight += 1
            sum_weight2 += 1
            weight += alpha

        if have_observed_value:
            # decay the values
            sum_x2 *= decay
            sum_x *= decay
            sum_weight *= decay
            # We decay this twice because we want the weight^2, so need to decay again
            # (We could explain this better; contributions welcome...)
            sum_weight2 *= decay**2
            weight *= decay

            var_biased = (sum_x2 / sum_weight) - ((sum_x / sum_weight) ** 2)

            # - Ultimately we want `sum(weights_norm**2)`, where `weights_norm` is
            #   `weights / sum(weights)`:
            #
            #   sum(weights_norm**2)
            #   = sum(weights**2 / sum(weights)**2)
            #   = sum(weights**2) / sum(weights)**2
            #   = sum_weight2 / sum_weight**2
            bias = 1 - sum_weight2 / (sum_weight**2)

            if bias > 0 and weight >= min_weight:
                out[i] = var_biased / bias
            else:
                out[i] = np.nan

        else:
            out[i] = np.nan


@ndmoving(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
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

    for i in range(min_count - 1, window):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = asum / count if count >= min_count else np.nan

    count_inv = 1 / count if count >= min_count else np.nan
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
            count_inv = 1 / count if count >= min_count else np.nan
        elif aold_valid:
            asum -= aold
            count -= 1
            count_inv = 1 / count if count >= min_count else np.nan

        out[i] = asum * count_inv
