import numpy as np
from numba import float32, float64, int64

from .decorators import ndmoving


@ndmoving.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_mean(a, window, min_count, out):
    asum = 0.0
    count = 0
    min_count = max(min_count, 1)

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


@ndmoving.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_sum(a, window, min_count, out):
    asum = 0.0
    count = 0

    # We don't generally split these up into two loops, but in `move_sum` & `move_mean`,
    # they're sufficiently different that it's worthwhile.

    for i in range(window):
        ai = a[i]
        if not np.isnan(ai):
            asum += ai
            count += 1
        out[i] = asum if count >= min_count else np.nan

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

        out[i] = asum if count >= min_count else np.nan


# TODO: pandas doesn't use a `min_count`, which maybe makes sense, but also makes it inconsistent?
# @ndmoving.wrap(
#     [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
# )
# def move_count(a, window, min_count, out):

#     count = 0

#     for i in range(window):
#         if not np.isnan(a[i]):
#             count += 1
#         out[i] = count if count >= min_count else np.nan

#     for i in range(window, len(a)):
#         if not np.isnan(a[i]):
#             count += 1
#         if not np.isnan(a[i - window]):
#             count -= 1
#         out[i] = count if count >= min_count else np.nan


@ndmoving.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_std(a, window, min_count, out):
    asum = 0.0
    asum_sq = 0.0
    count = 0
    min_count = max(min_count, 2)

    for i in range(len(a)):
        ai = a[i]

        if i >= window:
            aold = a[i - window]
            if not np.isnan(aold):
                asum -= aold
                asum_sq -= aold * aold
                count -= 1

        if not np.isnan(ai):
            asum += ai
            asum_sq += ai * ai
            count += 1

        if count >= min_count:
            variance = (asum_sq - asum**2 / count) / (count - 1)
            out[i] = np.sqrt(variance)
        else:
            out[i] = np.nan


@ndmoving.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_var(a, window, min_count, out):
    asum = 0.0
    asum_sq = 0.0
    count = 0
    min_count = max(min_count, 2)

    for i in range(len(a)):
        ai = a[i]

        if i >= window:
            aold = a[i - window]
            if not np.isnan(aold):
                asum -= aold
                asum_sq -= aold * aold
                count -= 1

        if not np.isnan(ai):
            asum += ai
            asum_sq += ai * ai
            count += 1

        if count >= min_count:
            out[i] = (asum_sq - asum**2 / count) / (count - 1)
        else:
            out[i] = np.nan


@ndmoving.wrap(
    [
        (float32[:], float32[:], int64, int64, float32[:]),
        (float64[:], float64[:], int64, int64, float64[:]),
    ]
)
def move_cov(a, b, window, min_count, out):
    asum = 0.0
    bsum = 0.0
    prodsum = (
        0.0  # This will store the sum of products of corresponding values in a and b
    )
    count = 0
    min_count = max(min_count, 2)

    for i in range(len(a)):
        ai = a[i]
        bi = b[i]

        if i >= window:
            aold = a[i - window]
            bold = b[i - window]
            if not (np.isnan(aold) or np.isnan(bold)):
                asum -= aold
                bsum -= bold
                prodsum -= aold * bold
                count -= 1

        if not (np.isnan(ai) or np.isnan(bi)):
            asum += ai
            bsum += bi
            prodsum += ai * bi
            count += 1
        if count >= min_count:
            out[i] = (prodsum - asum * bsum / count) / (count - 1)
        else:
            out[i] = np.nan


@ndmoving.wrap(
    [
        (float32[:], float32[:], int64, int64, float32[:]),
        (float64[:], float64[:], int64, int64, float64[:]),
    ]
)
def move_corr(a, b, window, min_count, out):
    asum = 0.0
    bsum = 0.0
    prodsum = 0.0
    asum_sq = 0.0
    bsum_sq = 0.0
    count = 0

    min_count = max(min_count, 1)

    for i in range(len(a)):
        ai = a[i]
        bi = b[i]

        if i >= window:
            aold = a[i - window]
            bold = b[i - window]
            if not (np.isnan(aold) or np.isnan(bold)):
                asum -= aold
                bsum -= bold
                prodsum -= aold * bold
                asum_sq -= aold * aold
                bsum_sq -= bold * bold
                count -= 1

        if not (np.isnan(ai) or np.isnan(bi)):
            asum += ai
            bsum += bi
            prodsum += ai * bi
            asum_sq += ai * ai
            bsum_sq += bi * bi
            count += 1
        if count >= min_count:
            count_reciprocal = 1.0 / count
            avg_a = asum * count_reciprocal
            avg_b = bsum * count_reciprocal
            var_a = asum_sq * count_reciprocal - avg_a**2
            var_b = bsum_sq * count_reciprocal - avg_b**2
            cov_ab = prodsum * count_reciprocal - avg_a * avg_b
            var_a_var_b = var_a * var_b
            if var_a_var_b > 0:
                out[i] = cov_ab / np.sqrt(var_a_var_b)
            else:
                out[i] = np.nan

        else:
            out[i] = np.nan
