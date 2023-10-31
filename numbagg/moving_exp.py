import numpy as np
from numba import float32, float64

from .decorators import ndmovingexp


@ndmovingexp(
    [
        (float32[:], float32, float32, float32[:]),
        (float64[:], float64, float64, float64[:]),
    ]
)
def move_exp_nancount(a, alpha, min_weight, out):
    N = len(a)

    count = weight = 0.0
    decay = 1.0 - alpha

    for i in range(N):
        a_i = a[i]

        count *= decay
        weight *= decay

        if not np.isnan(a_i):
            count += 1
            weight += alpha

        if weight >= min_weight:
            out[i] = count
        else:
            out[i] = np.nan


@ndmovingexp(
    [
        (float32[:], float32, float32, float32[:]),
        (float64[:], float64, float64, float64[:]),
    ]
)
def move_exp_nanmean(a, alpha, min_weight, out):
    N = len(a)

    number = denom = weight = 0.0
    decay = 1.0 - alpha

    for i in range(N):
        a_i = a[i]

        number *= decay
        denom *= decay
        weight *= decay

        if not np.isnan(a_i):
            number += a_i
            denom += 1.0
            weight += alpha

        if weight >= min_weight:
            out[i] = number / denom
        else:
            out[i] = np.nan


@ndmovingexp(
    [
        (float32[:], float32, float32, float32[:]),
        (float64[:], float64, float64, float64[:]),
    ]
)
def move_exp_nansum(a, alpha, min_weight, out):
    N = len(a)

    number = weight = 0.0
    decay = 1.0 - alpha
    zero_count = True

    for i in range(N):
        a_i = a[i]

        number *= decay
        weight *= decay

        if not np.isnan(a_i):
            zero_count = False
            number += a_i
            weight += alpha

        if weight >= min_weight and not zero_count:
            out[i] = number
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
    decay = 1.0 - alpha

    for i in range(N):
        a_i = a[i]

        # decay the values
        sum_x_2 *= decay
        sum_x *= decay
        sum_weight *= decay
        # We decay this twice because we want the weight^2, so need to decay again
        # (We could explain this better; contributions welcome...)
        sum_weight_2 *= decay**2
        weight *= decay

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

        if weight >= min_weight and bias > 0:
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
    decay = 1.0 - alpha

    for i in range(N):
        a_i = a[i]

        # decay the values
        sum_x_2 *= decay
        sum_x *= decay
        sum_weight *= decay
        # We decay this twice because we want the weight^2, so need to decay again
        # (We could explain this better; contributions welcome...)
        sum_weight_2 *= decay**2
        weight *= decay

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

        if weight >= min_weight and bias > 0:
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
    decay = 1.0 - alpha

    for i in range(N):
        a1_i = a1[i]
        a2_i = a2[i]

        # decay the values
        sum_x1 *= decay
        sum_x2 *= decay
        sum_x1x2 *= decay
        sum_weight *= decay
        sum_weight_2 *= decay**2
        weight *= decay

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

        if weight >= min_weight and bias > 0:
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
    decay = 1.0 - alpha

    for i in range(N):
        a1_i = a1[i]
        a2_i = a2[i]

        sum_x1 *= decay
        sum_x2 *= decay
        sum_x1x2 *= decay
        sum_weight *= decay
        sum_weight_2 *= decay**2
        weight *= decay

        sum_x1_2 *= decay
        sum_x2_2 *= decay

        if not (np.isnan(a1_i) or np.isnan(a2_i)):
            sum_x1 += a1_i
            sum_x2 += a2_i
            sum_x1x2 += a1_i * a2_i
            sum_weight += 1
            sum_weight_2 += 1
            weight += alpha
            sum_x1_2 += a1_i**2
            sum_x2_2 += a2_i**2

        # The bias cancels out, so we don't need to adjust for it

        cov = (sum_x1x2 - (sum_x1 * sum_x2 / sum_weight)) / sum_weight
        var_a1 = (sum_x1_2 - (sum_x1**2 / sum_weight)) / sum_weight
        var_a2 = (sum_x2_2 - (sum_x2**2 / sum_weight)) / sum_weight

        if weight >= min_weight:
            denominator = np.sqrt(var_a1 * var_a2)
            if denominator != 0:
                out[i] = cov / denominator
            else:
                out[i] = np.nan
        else:
            out[i] = np.nan
