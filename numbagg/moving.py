from typing import TypeVar

import numpy as np
from numba import float32, float64, int64, njit

from .decorators import _ENABLE_CACHE, ndmove
from .utils import FloatArray

T = TypeVar("T", bound=FloatArray)

# Why `move_var`, `move_std`, `move_cov` & `move_corr` offset their inputs and
# accumulate in float64:
#
# They all accumulate a sum of squares (or of products) and then subtract the
# square of the sum. Both terms scale with the square of the values themselves, so
# when the values are large relative to their spread the difference is almost
# entirely cancellation and the result keeps hardly any significant digits. Two
# separate losses were showing up:
#
# - The products were computed in the input's dtype, so a float32 series held only
#   ~7 digits of `ai * ai`. `move_var` on a constant float32 series at 1e8 returned
#   3.4e8 rather than 0, and `move_corr` on float32 values around 1e6 returned
#   correlations more than 30 outside [-1, 1].
# - Even in float64 the offset dominates: `move_cov` on values around 1e8 was off
#   by ~10 in absolute terms.
#
# Subtracting a fixed value from each series fixes the second — variance,
# covariance and correlation are all invariant to it, so it changes nothing in
# exact arithmetic — and computing in float64 fixes the first.
#
# Which constant to subtract is a real choice, because the accumulators run for the
# whole series: every term carries `(value - offset)**2`, so the offset sets the
# rounding floor everywhere, not just where it's a good fit. The mean of the whole
# series is the most accurate choice on most inputs, but it's ruled out here — it
# depends on values that haven't been seen yet, which would break the guarantee
# `test_appending_doesnt_change_earlier_results` pins, that appending to a series
# doesn't change results that were already emitted. The first observation keeps
# that guarantee but rests on `a[0]` being representative; when it isn't, the whole
# series pays for it. So we average the first `window` observations: still a
# function of data that any window-`window` result already depends on, but a single
# outlier is diluted by the window rather than setting the floor on its own.


# Cached alongside the gufuncs that call it: numba can't cache a compiled function
# whose callees aren't cacheable themselves.
@njit(cache=_ENABLE_CACHE)
def _offset(a, window) -> float:
    """The mean of the non-NaN values among the first `window` values of `a`."""
    total = 0.0
    count = 0
    for i in range(min(window, len(a))):
        ai = np.float64(a[i])
        if not np.isnan(ai):
            total += ai
            count += 1
    if count > 0:
        return total / count

    # The first window is entirely NaN, so fall back to the first non-NaN value
    # anywhere — better than 0.0 for a series that starts with a gap. Still safe
    # for the appending guarantee: if this scan finds different values for `a` and
    # a prefix of `a`, the prefix has no non-NaN values at all and every result it
    # can emit is NaN either way.
    for i in range(len(a)):
        ai = np.float64(a[i])
        if not np.isnan(ai):
            return ai
    return 0.0


@ndmove.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_mean(a: T, window: int, min_count: int, out: T) -> None:
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

        ai_valid: bool = not np.isnan(ai)
        aold_valid: bool = not np.isnan(aold)

        # We previously had a single operation where both variables are valid, but it
        # caused some numerical instability for float32 values. For example the
        # `test_numerical_issues_float32_move_mean_1` test fails. While it had a 10%
        # performance impact relative to the previous if / elif, the current mode with
        # just two `if` branches is about 10% faster than the previous mode; maybe it
        # can execute both branches in parallel?

        # if ai_valid and aold_valid:
        #     asum += ai - aold
        # elif ...

        if aold_valid:
            asum -= aold
            count -= 1
        if ai_valid:
            asum += ai
            count += 1

        out[i] = asum / count if count >= min_count else np.nan


@ndmove.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_sum(a: T, window: int, min_count: int, out: T) -> None:
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

        ai_valid: bool = not np.isnan(ai)
        aold_valid: bool = not np.isnan(aold)

        # Similar to the comment in `move_mean`, we previously had a single operation if
        # both were valid. That causes numerical instability for float32 values with a
        # window of 1.
        #
        # But possibly — particularly with a sum — the old and new values are likely to
        # be closer to each other than to the accumulator, so the numerical instability
        # is worse with this approach. When testing — for example with
        # `test_numerical_issues_float32_move_sum_100`, both approaches seem to fail
        # when increasing the multiplier at approximately the same rate.

        if ai_valid:
            asum += ai
            count += 1
        if aold_valid:
            asum -= aold
            count -= 1

        out[i] = asum if count >= min_count else np.nan


# TODO: pandas doesn't use a `min_count`, which maybe makes sense, but also makes it inconsistent?
# @ndmove.wrap(
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


@ndmove.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_std(a: T, window: int, min_count: int, out: T) -> None:
    asum = 0.0
    asum_sq = 0.0
    count = 0
    min_count = max(min_count, 2)

    # See the note above `_offset` for why we offset and use float64 here.
    offset = _offset(a, window)

    for i in range(len(a)):
        ai = np.float64(a[i]) - offset

        if i >= window:
            aold = np.float64(a[i - window]) - offset
            if not np.isnan(aold):
                asum -= aold
                asum_sq -= aold * aold
                count -= 1

        if not np.isnan(ai):
            asum += ai
            asum_sq += ai * ai
            count += 1

        if count >= min_count:
            # Clamp at zero: `asum**2 / count` can still exceed `asum_sq` by a
            # rounding error, and the negative variance that produces would come
            # back out of `sqrt` as NaN.
            variance = max((asum_sq - asum**2 / count) / (count - 1), 0.0)
            out[i] = np.sqrt(variance)
        else:
            out[i] = np.nan


@ndmove.wrap(
    [(float32[:], int64, int64, float32[:]), (float64[:], int64, int64, float64[:])]
)
def move_var(a: T, window: int, min_count: int, out: T) -> None:
    asum = 0.0
    asum_sq = 0.0
    count = 0
    min_count = max(min_count, 2)

    offset = _offset(a, window)

    for i in range(len(a)):
        ai = np.float64(a[i]) - offset

        if i >= window:
            aold = np.float64(a[i - window]) - offset
            if not np.isnan(aold):
                asum -= aold
                asum_sq -= aold * aold
                count -= 1

        if not np.isnan(ai):
            asum += ai
            asum_sq += ai * ai
            count += 1

        if count >= min_count:
            # Clamp at zero, as in `move_std` — a variance is never negative.
            out[i] = max((asum_sq - asum**2 / count) / (count - 1), 0.0)
        else:
            out[i] = np.nan


@ndmove.wrap(
    [
        (float32[:], float32[:], int64, int64, float32[:]),
        (float64[:], float64[:], int64, int64, float64[:]),
    ]
)
def move_cov(a: T, b: T, window: int, min_count: int, out: T) -> None:
    asum = 0.0
    bsum = 0.0
    prodsum = (
        0.0  # This will store the sum of products of corresponding values in a and b
    )
    count = 0
    min_count = max(min_count, 2)

    a_offset = _offset(a, window)
    b_offset = _offset(b, window)

    for i in range(len(a)):
        ai = np.float64(a[i]) - a_offset
        bi = np.float64(b[i]) - b_offset

        if i >= window:
            aold = np.float64(a[i - window]) - a_offset
            bold = np.float64(b[i - window]) - b_offset
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


@ndmove.wrap(
    [
        (float32[:], float32[:], int64, int64, float32[:]),
        (float64[:], float64[:], int64, int64, float64[:]),
    ]
)
def move_corr(a: T, b: T, window: int, min_count: int, out: T) -> None:
    asum = 0.0
    bsum = 0.0
    prodsum = 0.0
    asum_sq = 0.0
    bsum_sq = 0.0
    count = 0

    min_count = max(min_count, 1)

    a_offset = _offset(a, window)
    b_offset = _offset(b, window)

    for i in range(len(a)):
        ai = np.float64(a[i]) - a_offset
        bi = np.float64(b[i]) - b_offset

        if i >= window:
            aold = np.float64(a[i - window]) - a_offset
            bold = np.float64(b[i - window]) - b_offset
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
            # Clamp each variance at zero, as `move_var` does — otherwise two
            # variances that round slightly negative multiply to a positive
            # product, pass the guard below, and give a meaningless denominator.
            # A degenerate window then reaches the `else` branch and emits NaN,
            # which is the right answer for a correlation with a constant series.
            var_a = max(asum_sq * count_reciprocal - avg_a**2, 0.0)
            var_b = max(bsum_sq * count_reciprocal - avg_b**2, 0.0)
            cov_ab = prodsum * count_reciprocal - avg_a * avg_b
            var_a_var_b = var_a * var_b
            if var_a_var_b > 0:
                out[i] = cov_ab / np.sqrt(var_a_var_b)
            else:
                out[i] = np.nan

        else:
            out[i] = np.nan


# Re-export matrix functions for backward compatibility
from .moving_matrix import move_corrmatrix, move_covmatrix

__all__ = [
    "move_mean",
    "move_sum",
    "move_std",
    "move_var",
    "move_cov",
    "move_corr",
    "move_corrmatrix",
    "move_covmatrix",
]
