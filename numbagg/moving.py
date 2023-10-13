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

    numer = denom = weight = 0.0
    inertia = 1.0 - alpha

    for i in range(N):
        a_i = a[i]

        numer *= inertia
        denom *= inertia
        weight *= inertia

        if not np.isnan(a_i):
            numer += a_i
            denom += 1.0
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

    numer = weight = 0.0
    inertia = 1.0 - alpha
    zero_count = True

    for i in range(N):
        a_i = a[i]

        numer *= inertia
        weight *= inertia

        if not np.isnan(a_i):
            zero_count = False
            numer += a_i
            weight += alpha

        if weight >= min_weight and not zero_count:
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
    sum_x_2 = sum_x = sum_weight = sum_weight_2 = weight = 0.0
    inertia = 1.0 - alpha

    for i in range(N):
        a_i = a[i]

        # decay the values
        sum_x_2 *= inertia
        sum_x *= inertia
        sum_weight *= inertia
        # We decay this twice because we want the weight^2, so need to decay again
        # (We could explain this better; contributions welcome...)
        sum_weight_2 *= inertia**2
        weight *= inertia

        if not np.isnan(a_i):
            sum_x_2 += a_i**2
            sum_x += a_i
            sum_weight += 1.0
            sum_weight_2 += 1.0
            weight += alpha

        var_biased = (sum_x_2 / sum_weight) - ((sum_x / sum_weight) ** 2)

        # - Ultimately we want `sum(weights_norm**2)`, where `weights_norm` is
        #   `weights / sum(weights)`:
        #
        #   sum(weights_norm**2)
        #   = sum(weights**2 / sum(weights)**2)
        #   = sum(weights**2) / sum(weights)**2
        #   = sum_weight_2 / sum_weight**2
        bias = 1 - sum_weight_2 / (sum_weight**2)

        if weight >= min_weight:
            out[i] = var_biased / bias
        else:
            out[i] = np.nan


@ndmovingexp(
    [
        (float32[:], float32, float32, float32[:]),
        (float64[:], float64, float64, float64[:]),
    ]
)
def move_exp_nanstd(a, alpha, min_weight, out):
    """
    Calculates the exponentially decayed standard deviation.

    Note that technically the unbiased weighted standard deviation is not exactly the
    same as the square root of the unbiased weighted variance, since the bias is
    concave. But it's close, and it's what pandas does.

    (If anyone knows the math well and wants to take a pass at improving it,
    contributions are welcome.)
    """
    # This is very similar to `move_exp_nanvar`, but square-roots in the final step. It
    # could be implemented as a wrapper around `move_exp_nanvar`, but it causes a couple
    # of small complications around warnings for `np.sqrt` on invalid values, and passing
    # the `axis` parameter, such that it was easier to just copy-pasta.

    N = len(a)

    # sum_x: decayed sum of the sequence values.
    # sum_x2: decayed sum of the squared sequence values.
    # n: decayed count of non-missing values observed so far in the sequence.
    # n2: decayed sum of the (already-decayed) weights of non-missing values.
    sum_x_2 = sum_x = sum_weight = sum_weight_2 = weight = 0.0
    inertia = 1.0 - alpha

    for i in range(N):
        a_i = a[i]

        # decay the values
        sum_x_2 *= inertia
        sum_x *= inertia
        sum_weight *= inertia
        # We decay this twice because we want the weight^2, so need to decay again
        # (We could explain this better; contributions welcome...)
        sum_weight_2 *= inertia**2
        weight *= inertia

        if not np.isnan(a_i):
            sum_x_2 += a_i**2
            sum_x += a_i
            sum_weight += 1.0
            sum_weight_2 += 1.0
            weight += alpha

        var_biased = (sum_x_2 / sum_weight) - ((sum_x / sum_weight) ** 2)

        # - Ultimately we want `sum(weights_norm**2)`, where `weights_norm` is
        #   `weights / sum(weights)`:
        #
        #   sum(weights_norm**2)
        #   = sum(weights**2 / sum(weights)**2)
        #   = sum(weights**2) / sum(weights)**2
        #   = sum_weight_2 / sum_weight**2
        bias = 1 - sum_weight_2 / (sum_weight**2)

        if weight >= min_weight:
            out[i] = np.sqrt(var_biased / bias)
        else:
            out[i] = np.nan


@ndmovingexp(
    [
        (float32[:], float32[:], float32, float32, float32[:]),
        (float64[:], float64[:], float64, float64, float64[:]),
    ]
)
def move_exp_nancov(a1, a2, alpha, min_weight, out):
    N = len(a1)

    # sum_x1: decayed sum of the sequence values for a1.
    # sum_x2: decayed sum of the sequence values for a2.
    # sum_x1x2: decayed sum of the product of sequence values for a1 and a2.
    # sum_weight: decayed count of non-missing values observed so far in the sequence.
    # sum_weight_2: decayed sum of the (already-decayed) weights of non-missing values.
    sum_x1 = sum_x2 = sum_x1x2 = sum_weight = sum_weight_2 = weight = 0.0
    inertia = 1.0 - alpha

    for i in range(N):
        a1_i = a1[i]
        a2_i = a2[i]

        # decay the values
        sum_x1 *= inertia
        sum_x2 *= inertia
        sum_x1x2 *= inertia
        sum_weight *= inertia
        sum_weight_2 *= inertia**2
        weight *= inertia

        if not (np.isnan(a1_i) or np.isnan(a2_i)):
            sum_x1 += a1_i
            sum_x2 += a2_i
            sum_x1x2 += a1_i * a2_i
            sum_weight += 1
            sum_weight_2 += 1
            weight += alpha

        cov_biased = ((sum_x1x2) - (sum_x1 * sum_x2 / sum_weight)) / sum_weight

        # Adjust for the bias. (explained in `move_exp_nanvar`)
        bias = 1 - sum_weight_2 / (sum_weight**2)

        if weight >= min_weight:
            out[i] = cov_biased / bias
        else:
            out[i] = np.nan


@ndmovingexp(
    [
        (float32[:], float32[:], float32, float32, float32[:]),
        (float64[:], float64[:], float64, float64, float64[:]),
    ]
)
def move_exp_nancorr(a1, a2, alpha, min_weight, out):
    N = len(a1)

    sum_x1 = sum_x2 = sum_x1x2 = sum_weight = sum_weight_2 = 0
    weight = sum_x1_2 = sum_x2_2 = 0
    inertia = 1.0 - alpha

    for i in range(N):
        a1_i = a1[i]
        a2_i = a2[i]

        sum_x1 *= inertia
        sum_x2 *= inertia
        sum_x1x2 *= inertia
        sum_weight *= inertia
        sum_weight_2 *= inertia**2
        weight *= inertia

        sum_x1_2 *= inertia
        sum_x2_2 *= inertia

        if not (np.isnan(a1_i) or np.isnan(a2_i)):
            sum_x1 += a1_i
            sum_x2 += a2_i
            sum_x1x2 += a1_i * a2_i
            sum_weight += 1
            sum_weight_2 += 1
            weight += alpha
            sum_x1_2 += a1_i**2
            sum_x2_2 += a2_i**2

        cov_biased = (sum_x1x2 - (sum_x1 * sum_x2 / sum_weight)) / sum_weight
        var_a1_biased = (sum_x1_2 - (sum_x1**2 / sum_weight)) / sum_weight
        var_a2_biased = (sum_x2_2 - (sum_x2**2 / sum_weight)) / sum_weight

        # Adjust for the bias. (explained in `move_exp_nanvar`)
        bias = 1 - sum_weight_2 / (sum_weight**2)

        if weight >= min_weight:
            var_a1_unbiased = var_a1_biased / bias
            var_a2_unbiased = var_a2_biased / bias
            cov_unbiased = cov_biased / bias

            denominator = np.sqrt(var_a1_unbiased * var_a2_unbiased)
            if denominator != 0:
                out[i] = cov_unbiased / denominator
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
