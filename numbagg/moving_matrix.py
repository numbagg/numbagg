"""Moving window matrix functions using the decorator pattern."""

import numpy as np
from numba import float32, float64, int64, njit

from .decorators import _ENABLE_CACHE, ndmoveexpmatrix, ndmovematrix

__all__ = [
    "move_corrmatrix",
    "move_covmatrix",
    "move_exp_nancorrmatrix",
    "move_exp_nancovmatrix",
]

# Why these four functions offset their inputs and accumulate in float64:
#
# Each accumulates a sum of products (and, for the correlations, a sum of squares)
# and then subtracts the product of the means. Both terms scale with the square of
# the values themselves, so when the values sit far from zero the difference is
# almost entirely cancellation and keeps hardly any significant digits. Two losses
# stack:
#
# - The accumulators took the input's dtype, so a float32 series held only ~7
#   digits of `x * y`. On a constant float32 series at 1e8, `move_covmatrix`
#   returned a variance of -2e8 rather than 0, and `move_exp_nancovmatrix` -2e9.
# - Even in float64 the offset dominates the spread: on standard-normal data
#   shifted by 1e8, the covariance diagonals — variances, so never negative — came
#   back around -15, `move_corrmatrix` reached 2.4 (a correlation, so outside
#   [-1, 1] by more than its own range), and most of `move_corrmatrix`'s output
#   collapsed to NaN because the variances rounded non-positive and hit the
#   `var_i > 0` guard.
#
# Correlation and covariance are both invariant to a per-variable offset, so
# subtracting one changes nothing in exact arithmetic; float64 accumulators fix the
# first loss and the offset fixes the second. #758 applies the same remedy to the
# non-matrix moving functions and #759 to the static `nancovmatrix` /
# `nancorrmatrix`; the three share this rationale and the offset choice below.
#
# Which constant to subtract is a real choice, because the accumulators run for the
# whole series and every term carries `(value - offset)**2` — the offset sets the
# rounding floor everywhere, not only where it fits well. The mean of the whole
# series is the most accurate on most inputs but is ruled out: it depends on values
# not yet seen, so `f(a)[:n]` would depend on `a[n:]` and appending to a series
# would change results already emitted.
#
# The windowed functions instead average the first `min(window, min_count)`
# observations, which dilutes an outlier while reading only rows the accumulators
# have already reached. Both bounds are load-bearing. `min_count` is what keeps the
# offset from looking ahead: output at time `t` is emitted as soon as a pair has
# `min_count` observations, so with `min_count < window` averaging the whole first
# window would make the value at `t` a function of `a[t + 1 : window]` — and when
# that tail holds a much larger value than the partial window's own, the offset is
# orders of magnitude off and cancellation gets worse than with no offset at all.
# A non-NaN output at `t` already requires `t >= min_count - 1`, so those rows are
# always in hand. `window` bounds it in turn because `min_count` may exceed the
# window only nominally; `window <= n_obs` is enforced, so a prefix short enough to
# change the average can't be evaluated at all.
#
# The exponential functions have no window to average over, so they take each
# variable's first non-NaN observation.


# Cached alongside the gufuncs that call them: numba can't cache a compiled
# function whose callees aren't cacheable themselves.
@njit(cache=_ENABLE_CACHE)
def _first_observation(a, k):
    """First non-NaN observation of variable `k` in `a`, or 0.0 if it has none."""
    for t in range(a.shape[0]):
        v = np.float64(a[t, k])
        if not np.isnan(v):
            return v
    return 0.0


@njit(cache=_ENABLE_CACHE)
def _leading_rows_shift(a, n_rows):
    """Per-variable mean of the non-NaN values among the first `n_rows` rows of `a`."""
    n_obs, n_vars = a.shape
    shift = np.zeros(n_vars, dtype=np.float64)
    for k in range(n_vars):
        total = 0.0
        count = 0
        for t in range(min(n_rows, n_obs)):
            v = np.float64(a[t, k])
            if not np.isnan(v):
                total += v
                count += 1
        if count > 0:
            shift[k] = total / count
        else:
            # Those rows are entirely NaN for this variable, so fall back to its
            # first non-NaN value anywhere — better than 0.0 for a variable that
            # starts with a gap, and still look-ahead-free: any output involving
            # this variable needs a non-NaN value of it at or before that time
            # step, so the first one anywhere is the first one already seen.
            shift[k] = _first_observation(a, k)
    return shift


@njit(cache=_ENABLE_CACHE)
def _first_observation_shift(a):
    """Per-variable first non-NaN observation of `a`, or 0.0 for an all-NaN variable."""
    n_vars = a.shape[1]
    shift = np.zeros(n_vars, dtype=np.float64)
    for k in range(n_vars):
        shift[k] = _first_observation(a, k)
    return shift


@ndmovematrix.wrap(
    signature=(
        [
            (float32[:, :], int64, int64, float32[:, :, :]),
            (float64[:, :], int64, int64, float64[:, :, :]),
        ],
        "(m,n),(),()->(m,n,n)",
    )
)
def move_corrmatrix(a, window, min_count, out):
    """
    Moving window correlation matrix gufunc.

    Dimension conventions (fixed for consistency):
    - Input: (n_obs, n_vars) - observations as rows, variables as columns
    - Output: (n_obs, n_vars, n_vars) - correlation matrix at each time step
    - Broadcasting: Supports arbitrary leading dimensions via NumPy's gufunc system

    For each time step, computes correlation matrix using the rolling window ending at that time.
    Unlike NumPy's corrcoef (2D only), this broadcasts over higher dimensions:

    Examples:
    - 2D: (100, 3) -> (100, 3, 3) - matrix for each of 100 time steps
    - 3D: (batch=5, obs=100, vars=3) -> (5, 100, 3, 3)
    - 4D: (2, 5, 100, 3) -> (2, 5, 100, 3, 3)
    """
    n_obs = a.shape[0]
    n_vars = a.shape[1]
    min_count = max(min_count, 1)

    # Each variable is offset by the mean of its first `min(window, min_count)`
    # observations, and the accumulators run in float64. Correlation is invariant
    # to a per-variable offset, so this changes nothing in exact arithmetic, but it
    # keeps the significant digits: the sums below scale with the square of the
    # values, so on data sitting far from zero the final subtraction is nearly all
    # cancellation. See the module comment for why this particular offset.
    shift = _leading_rows_shift(a, min(window, min_count))

    # Initialize pairwise statistics - each (i,j) pair tracks its own statistics
    # to ensure all moments are computed over the same set of observations
    sums_i = np.zeros((n_vars, n_vars), dtype=np.float64)
    sums_j = np.zeros((n_vars, n_vars), dtype=np.float64)
    sums_sq_i = np.zeros((n_vars, n_vars), dtype=np.float64)
    sums_sq_j = np.zeros((n_vars, n_vars), dtype=np.float64)
    prods = np.zeros((n_vars, n_vars), dtype=np.float64)
    pair_counts = np.zeros((n_vars, n_vars), dtype=np.int64)

    for t in range(n_obs):
        # Remove old values when window slides
        if t >= window:
            for i in range(n_vars):
                old_val_i = np.float64(a[t - window, i]) - shift[i]
                if np.isnan(old_val_i):
                    continue
                for j in range(n_vars):
                    old_val_j = np.float64(a[t - window, j]) - shift[j]
                    if np.isnan(old_val_j):
                        continue
                    # Only update pairwise statistics for observations where BOTH are valid
                    sums_i[i, j] -= old_val_i
                    sums_j[i, j] -= old_val_j
                    sums_sq_i[i, j] -= old_val_i * old_val_i
                    sums_sq_j[i, j] -= old_val_j * old_val_j
                    prods[i, j] -= old_val_i * old_val_j
                    pair_counts[i, j] -= 1

        # Add new values
        for i in range(n_vars):
            new_val_i = np.float64(a[t, i]) - shift[i]
            if np.isnan(new_val_i):
                continue
            for j in range(n_vars):
                new_val_j = np.float64(a[t, j]) - shift[j]
                if np.isnan(new_val_j):
                    continue
                # Only update pairwise statistics for observations where BOTH are valid
                sums_i[i, j] += new_val_i
                sums_j[i, j] += new_val_j
                sums_sq_i[i, j] += new_val_i * new_val_i
                sums_sq_j[i, j] += new_val_j * new_val_j
                prods[i, j] += new_val_i * new_val_j
                pair_counts[i, j] += 1

        # Compute correlation matrix for current window
        for i in range(n_vars):
            for j in range(n_vars):
                n = pair_counts[i, j]
                # Need at least 2 observations for correlation (to compute variance)
                if n >= max(min_count, 2):
                    mean_i = sums_i[i, j] / n
                    mean_j = sums_j[i, j] / n

                    # Compute variances using pairwise statistics
                    var_i = sums_sq_i[i, j] / n - mean_i * mean_i
                    var_j = sums_sq_j[i, j] / n - mean_j * mean_j

                    # A variance is never negative; rounding can still take these
                    # below zero on a near-constant window. Clamping means such a
                    # window yields NaN below rather than a correlation built on a
                    # negative variance — two of which multiply to a positive
                    # product and would pass a guard on the product alone.
                    var_i = max(var_i, 0.0)
                    var_j = max(var_j, 0.0)

                    # Compute covariance using pairwise statistics
                    cov = prods[i, j] / n - mean_i * mean_j

                    # Compute correlation
                    if var_i > 0 and var_j > 0:
                        corr = cov / np.sqrt(var_i * var_j)
                        # A window of exactly two observations is perfectly
                        # correlated, and rounding can put the quotient a few
                        # ulps outside [-1, 1] — `test_correlation_bounds`
                        # catches it without this. `np.corrcoef`, the documented
                        # counterpart of these matrix functions, clips for the
                        # same reason; this runs after the variance clamp above,
                        # so a degenerate window still yields NaN rather than a
                        # clipped value. Deliberately unlike `move_corr`, which
                        # #758 leaves unclipped so that a badly-conditioned
                        # result stays visible as an out-of-range number.
                        out[t, i, j] = min(max(corr, -1.0), 1.0)
                    else:
                        out[t, i, j] = np.nan
                else:
                    out[t, i, j] = np.nan


@ndmovematrix.wrap(
    signature=(
        [
            (float32[:, :], int64, int64, float32[:, :, :]),
            (float64[:, :], int64, int64, float64[:, :, :]),
        ],
        "(m,n),(),()->(m,n,n)",
    )
)
def move_covmatrix(a, window, min_count, out):
    """
    Moving window covariance matrix gufunc.

    Dimension conventions (fixed for consistency):
    - Input: (n_obs, n_vars) - observations as rows, variables as columns
    - Output: (n_obs, n_vars, n_vars) - covariance matrix at each time step
    - Broadcasting: Supports arbitrary leading dimensions via NumPy's gufunc system

    For each time step, computes covariance matrix using the rolling window ending at that time.
    Unlike NumPy's cov (2D only), this broadcasts over higher dimensions:

    Examples:
    - 2D: (100, 3) -> (100, 3, 3) - matrix for each of 100 time steps
    - 3D: (batch=5, obs=100, vars=3) -> (5, 100, 3, 3)
    - 4D: (2, 5, 100, 3) -> (2, 5, 100, 3, 3)
    """
    n_obs = a.shape[0]
    n_vars = a.shape[1]
    min_count = max(min_count, 1)

    # See `move_corrmatrix` — covariance is invariant to a per-variable offset too.
    shift = _leading_rows_shift(a, min(window, min_count))

    # Initialize pairwise statistics - each (i,j) pair tracks its own statistics
    # to ensure all moments are computed over the same set of observations
    sums_i = np.zeros((n_vars, n_vars), dtype=np.float64)
    sums_j = np.zeros((n_vars, n_vars), dtype=np.float64)
    prods = np.zeros((n_vars, n_vars), dtype=np.float64)
    pair_counts = np.zeros((n_vars, n_vars), dtype=np.int64)

    for t in range(n_obs):
        # Remove old values when window slides
        if t >= window:
            for i in range(n_vars):
                old_val_i = np.float64(a[t - window, i]) - shift[i]
                if np.isnan(old_val_i):
                    continue
                for j in range(n_vars):
                    old_val_j = np.float64(a[t - window, j]) - shift[j]
                    if np.isnan(old_val_j):
                        continue
                    # Only update pairwise statistics for observations where BOTH are valid
                    sums_i[i, j] -= old_val_i
                    sums_j[i, j] -= old_val_j
                    prods[i, j] -= old_val_i * old_val_j
                    pair_counts[i, j] -= 1

        # Add new values
        for i in range(n_vars):
            new_val_i = np.float64(a[t, i]) - shift[i]
            if np.isnan(new_val_i):
                continue
            for j in range(n_vars):
                new_val_j = np.float64(a[t, j]) - shift[j]
                if np.isnan(new_val_j):
                    continue
                # Only update pairwise statistics for observations where BOTH are valid
                sums_i[i, j] += new_val_i
                sums_j[i, j] += new_val_j
                prods[i, j] += new_val_i * new_val_j
                pair_counts[i, j] += 1

        # Compute covariance matrix for current window
        for i in range(n_vars):
            for j in range(n_vars):
                n = pair_counts[i, j]
                if n >= min_count:
                    if n > 1:
                        # Unbiased covariance with ddof=1 using pairwise statistics
                        mean_i = sums_i[i, j] / n
                        mean_j = sums_j[i, j] / n
                        out[t, i, j] = (prods[i, j] / n - mean_i * mean_j) * n / (n - 1)
                    else:
                        # n == 1, covariance is undefined (requires at least 2 points)
                        out[t, i, j] = np.nan
                else:
                    out[t, i, j] = np.nan

        # The diagonal is a variance, so it can't be negative; off-diagonal
        # covariances legitimately can be. Kept out of the loop above so that
        # loop stays free of an `i == j` branch.
        for i in range(n_vars):
            if out[t, i, i] < 0.0:
                out[t, i, i] = 0.0


@ndmoveexpmatrix.wrap(
    signature=(
        [
            (float32[:, :], float32[:], float32, float32[:, :, :]),
            (float64[:, :], float64[:], float64, float64[:, :, :]),
        ],
        "(m,n),(m),()->(m,n,n)",
    )
)
def move_exp_nancorrmatrix(a, alpha, min_weight, out):
    """
    Exponential moving window correlation matrix gufunc.

    Dimension conventions (fixed for consistency):
    - Input: (n_obs, n_vars) - observations as rows, variables as columns
    - Output: (n_obs, n_vars, n_vars) - correlation matrix at each time step
    - Broadcasting: Supports arbitrary leading dimensions via NumPy's gufunc system
    - Alpha parameter: Supports scalar or array broadcasting

    For each time step, computes correlation matrix using exponentially weighted observations
    up to that time. Recent observations have higher weight based on the alpha parameter.
    Unlike NumPy's corrcoef (2D only), this broadcasts over higher dimensions:

    Examples:
    - 2D: (100, 3) -> (100, 3, 3) - matrix for each of 100 time steps
    - 3D: (batch=5, obs=100, vars=3) -> (5, 100, 3, 3)
    - 4D: (2, 5, 100, 3) -> (2, 5, 100, 3, 3)
    """
    n_obs = a.shape[0]
    n_vars = a.shape[1]

    # Same cancellation problem as `move_corrmatrix`, same remedy. There's no
    # window to average over here, so the offset is each variable's first
    # non-NaN observation.
    shift = _first_observation_shift(a)

    # Initialize pairwise statistics - each (i,j) pair tracks its own statistics
    # This is necessary for consistency with non-matrix exponential functions
    sums_i = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # sum of variable i for pair (i,j)
    sums_j = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # sum of variable j for pair (i,j)
    sums_sq_i = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # sum of squares of variable i for pair (i,j)
    sums_sq_j = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # sum of squares of variable j for pair (i,j)
    prods = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # sum of products for pair (i,j)
    pair_weights = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # accumulated alpha weights
    pair_sum_weights = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # count of valid pairs
    pair_sum_weights_sq = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # sum of squared weights

    for t in range(n_obs):
        alpha_t = alpha[t]
        decay = 1.0 - alpha_t

        # Apply exponential decay to all pairwise statistics
        sums_i *= decay
        sums_j *= decay
        sums_sq_i *= decay
        sums_sq_j *= decay
        prods *= decay
        pair_weights *= decay
        pair_sum_weights *= decay
        pair_sum_weights_sq *= decay**2

        # Add new values - track pairwise statistics for consistency
        for i in range(n_vars):
            new_val_i = np.float64(a[t, i]) - shift[i]
            if np.isnan(new_val_i):
                continue

            for j in range(n_vars):
                new_val_j = np.float64(a[t, j]) - shift[j]
                if np.isnan(new_val_j):
                    continue

                # Only update pairwise statistics if BOTH values are non-NaN (consistent with non-matrix functions)
                sums_i[i, j] += new_val_i
                sums_j[i, j] += new_val_j
                sums_sq_i[i, j] += new_val_i * new_val_i
                sums_sq_j[i, j] += new_val_j * new_val_j
                prods[i, j] += new_val_i * new_val_j
                pair_weights[i, j] += alpha_t
                pair_sum_weights[i, j] += 1.0
                pair_sum_weights_sq[i, j] += 1.0

        # Compute correlation matrix for current time step
        for i in range(n_vars):
            for j in range(n_vars):
                # Use pairwise statistics for each (i,j) combination
                bias = (
                    1 - pair_sum_weights_sq[i, j] / (pair_sum_weights[i, j] ** 2)
                    if pair_sum_weights[i, j] > 0
                    else 0.0
                )

                if pair_weights[i, j] >= min_weight and bias > 0:
                    # Compute correlation using pairwise statistics
                    n = pair_sum_weights[i, j]
                    mean_i = sums_i[i, j] / n
                    mean_j = sums_j[i, j] / n

                    # Compute variances (biased)
                    var_i_biased = (sums_sq_i[i, j] / n) - (mean_i * mean_i)
                    var_j_biased = (sums_sq_j[i, j] / n) - (mean_j * mean_j)

                    # Compute covariance (biased)
                    cov_biased = (prods[i, j] / n) - (mean_i * mean_j)

                    # Apply bias correction
                    var_i = var_i_biased / bias
                    var_j = var_j_biased / bias
                    cov = cov_biased / bias

                    # See `move_corrmatrix` — a variance can't be negative, and
                    # guarding the product alone lets two negative ones through.
                    var_i = max(var_i, 0.0)
                    var_j = max(var_j, 0.0)

                    # Compute correlation
                    if var_i > 0 and var_j > 0:
                        corr = cov / np.sqrt(var_i * var_j)
                        # A window of exactly two observations is perfectly
                        # correlated, and rounding can put the quotient a few
                        # ulps outside [-1, 1] — `test_correlation_bounds`
                        # catches it without this. `np.corrcoef`, the documented
                        # counterpart of these matrix functions, clips for the
                        # same reason; this runs after the variance clamp above,
                        # so a degenerate window still yields NaN rather than a
                        # clipped value. Deliberately unlike `move_corr`, which
                        # #758 leaves unclipped so that a badly-conditioned
                        # result stays visible as an out-of-range number.
                        out[t, i, j] = min(max(corr, -1.0), 1.0)
                    else:
                        out[t, i, j] = np.nan
                else:
                    out[t, i, j] = np.nan


@ndmoveexpmatrix.wrap(
    signature=(
        [
            (float32[:, :], float32[:], float32, float32[:, :, :]),
            (float64[:, :], float64[:], float64, float64[:, :, :]),
        ],
        "(m,n),(m),()->(m,n,n)",
    )
)
def move_exp_nancovmatrix(a, alpha, min_weight, out):
    """
    Exponential moving window covariance matrix gufunc.

    Dimension conventions (fixed for consistency):
    - Input: (n_obs, n_vars) - observations as rows, variables as columns
    - Output: (n_obs, n_vars, n_vars) - covariance matrix at each time step
    - Broadcasting: Supports arbitrary leading dimensions via NumPy's gufunc system
    - Alpha parameter: Supports scalar or array broadcasting

    For each time step, computes covariance matrix using exponentially weighted observations
    up to that time. Recent observations have higher weight based on the alpha parameter.
    Unlike NumPy's cov (2D only), this broadcasts over higher dimensions:

    Examples:
    - 2D: (100, 3) -> (100, 3, 3) - matrix for each of 100 time steps
    - 3D: (batch=5, obs=100, vars=3) -> (5, 100, 3, 3)
    - 4D: (2, 5, 100, 3) -> (2, 5, 100, 3, 3)
    """
    n_obs = a.shape[0]
    n_vars = a.shape[1]

    # See `move_exp_nancorrmatrix`.
    shift = _first_observation_shift(a)

    # Initialize pairwise statistics - each (i,j) pair tracks its own statistics
    # This is necessary for consistency with non-matrix exponential functions
    sums_i = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # sum of variable i for pair (i,j)
    sums_j = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # sum of variable j for pair (i,j)
    prods = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # sum of products for pair (i,j)
    pair_weights = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # accumulated alpha weights
    pair_sum_weights = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # count of valid pairs
    pair_sum_weights_sq = np.zeros(
        (n_vars, n_vars), dtype=np.float64
    )  # sum of squared weights

    for t in range(n_obs):
        alpha_t = alpha[t]
        decay = 1.0 - alpha_t

        # Apply exponential decay to all pairwise statistics
        sums_i *= decay
        sums_j *= decay
        prods *= decay
        pair_weights *= decay
        pair_sum_weights *= decay
        pair_sum_weights_sq *= decay**2

        # Add new values - track pairwise statistics for consistency
        for i in range(n_vars):
            new_val_i = np.float64(a[t, i]) - shift[i]
            if np.isnan(new_val_i):
                continue

            for j in range(n_vars):
                new_val_j = np.float64(a[t, j]) - shift[j]
                if np.isnan(new_val_j):
                    continue

                # Only update pairwise statistics if BOTH values are non-NaN (consistent with non-matrix functions)
                sums_i[i, j] += new_val_i
                sums_j[i, j] += new_val_j
                prods[i, j] += new_val_i * new_val_j
                pair_weights[i, j] += alpha_t
                pair_sum_weights[i, j] += 1.0
                pair_sum_weights_sq[i, j] += 1.0

        # Compute covariance matrix for current time step
        for i in range(n_vars):
            for j in range(n_vars):
                # Check if we have sufficient weight for a meaningful covariance calculation
                bias = (
                    1 - pair_sum_weights_sq[i, j] / (pair_sum_weights[i, j] ** 2)
                    if pair_sum_weights[i, j] > 0
                    else 0.0
                )

                if pair_weights[i, j] >= min_weight and bias > 0:
                    # Compute covariance using pairwise statistics
                    n = pair_sum_weights[i, j]
                    mean_i = sums_i[i, j] / n
                    mean_j = sums_j[i, j] / n

                    # Compute biased covariance
                    cov_biased = (prods[i, j] / n) - mean_i * mean_j

                    # Apply bias correction
                    out[t, i, j] = cov_biased / bias
                else:
                    out[t, i, j] = np.nan

        # See `move_covmatrix` — the diagonal is a variance, so never negative.
        for i in range(n_vars):
            if out[t, i, i] < 0.0:
                out[t, i, i] = 0.0
