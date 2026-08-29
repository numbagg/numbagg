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


@njit(cache=_ENABLE_CACHE)
def _initial_shift(a):
    """Return each variable's first finite value as a translation constant."""
    n_obs, n_vars = a.shape
    shift = np.zeros(n_vars, dtype=np.float64)
    for k in range(n_vars):
        for t in range(n_obs):
            value = np.float64(a[t, k])
            if not np.isnan(value):
                shift[k] = value
                break
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
                old_val_i = np.float64(a[t - window, i])
                if np.isnan(old_val_i):
                    continue
                for j in range(i, n_vars):
                    old_val_j = np.float64(a[t - window, j])
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
            new_val_i = np.float64(a[t, i])
            if np.isnan(new_val_i):
                continue
            for j in range(i, n_vars):
                new_val_j = np.float64(a[t, j])
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
            for j in range(i, n_vars):
                n = pair_counts[i, j]
                # Need at least 2 observations for correlation (to compute variance)
                if n >= max(min_count, 2):
                    mean_i = sums_i[i, j] / n
                    mean_j = sums_j[i, j] / n

                    # Compute variances using pairwise statistics
                    var_i = sums_sq_i[i, j] / n - mean_i * mean_i
                    var_j = sums_sq_j[i, j] / n - mean_j * mean_j

                    # Compute covariance using pairwise statistics
                    cov = prods[i, j] / n - mean_i * mean_j

                    # Compute correlation
                    if var_i > 0 and var_j > 0:
                        denominator = np.sqrt(var_i * var_j)
                        if denominator == 0.0 or not np.isfinite(denominator):
                            denominator = np.sqrt(var_i) * np.sqrt(var_j)
                        corr = cov / denominator
                        if corr > 1.0:
                            corr = 1.0
                        elif corr < -1.0:
                            corr = -1.0
                        out[t, i, j] = corr
                        out[t, j, i] = corr
                    else:
                        out[t, i, j] = np.nan
                        out[t, j, i] = np.nan
                else:
                    out[t, i, j] = np.nan
                    out[t, j, i] = np.nan


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

    # Initialize pairwise statistics - each (i,j) pair tracks its own statistics
    # to ensure all moments are computed over the same set of observations
    sums_i = np.zeros((n_vars, n_vars), dtype=np.float64)
    sums_j = np.zeros((n_vars, n_vars), dtype=np.float64)
    prods = np.zeros((n_vars, n_vars), dtype=np.float64)
    pair_counts = np.zeros((n_vars, n_vars), dtype=np.int64)
    old_obs = np.empty(n_vars, dtype=np.float64)
    new_obs = np.empty(n_vars, dtype=np.float64)

    for t in range(n_obs):
        # Remove old values when window slides
        if t >= window:
            for i in range(n_vars):
                old_obs[i] = np.float64(a[t - window, i])
            for i in range(n_vars):
                old_val_i = old_obs[i]
                if np.isnan(old_val_i):
                    continue
                for j in range(i, n_vars):
                    old_val_j = old_obs[j]
                    if np.isnan(old_val_j):
                        continue
                    # Only update pairwise statistics for observations where BOTH are valid
                    sums_i[i, j] -= old_val_i
                    sums_j[i, j] -= old_val_j
                    prods[i, j] -= old_val_i * old_val_j
                    pair_counts[i, j] -= 1

        # Add new values
        for i in range(n_vars):
            new_obs[i] = np.float64(a[t, i])
        for i in range(n_vars):
            new_val_i = new_obs[i]
            if np.isnan(new_val_i):
                continue
            for j in range(i, n_vars):
                new_val_j = new_obs[j]
                if np.isnan(new_val_j):
                    continue
                # Only update pairwise statistics for observations where BOTH are valid
                sums_i[i, j] += new_val_i
                sums_j[i, j] += new_val_j
                prods[i, j] += new_val_i * new_val_j
                pair_counts[i, j] += 1

        # Compute covariance matrix for current window
        for i in range(n_vars):
            for j in range(i, n_vars):
                n = pair_counts[i, j]
                if n >= min_count:
                    if n > 1:
                        # Unbiased covariance with ddof=1 using pairwise statistics
                        mean_i = sums_i[i, j] / n
                        mean_j = sums_j[i, j] / n
                        cov = (prods[i, j] / n - mean_i * mean_j) * n / (n - 1)
                        out[t, i, j] = cov
                        out[t, j, i] = cov
                    else:
                        # n == 1, covariance is undefined (requires at least 2 points)
                        out[t, i, j] = np.nan
                        out[t, j, i] = np.nan
                else:
                    out[t, i, j] = np.nan
                    out[t, j, i] = np.nan
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

    # Pairwise moving means and central moments preserve accuracy as old values
    # decay away. Float64 state avoids compounding float32 rounding over time.
    means_i = np.zeros((n_vars, n_vars), dtype=np.float64)
    means_j = np.zeros((n_vars, n_vars), dtype=np.float64)
    moments_i = np.zeros((n_vars, n_vars), dtype=np.float64)
    moments_j = np.zeros((n_vars, n_vars), dtype=np.float64)
    co_moments = np.zeros((n_vars, n_vars), dtype=np.float64)
    pair_weights = np.zeros((n_vars, n_vars), dtype=np.float64)
    sum_weights = np.zeros((n_vars, n_vars), dtype=np.float64)
    weight_products = np.zeros((n_vars, n_vars), dtype=np.float64)
    # Correlation is translation invariant. Subtracting an observed value keeps
    # Welford's evolving means small when the input has a large common offset.
    shift = _initial_shift(a)
    obs = np.empty(n_vars, dtype=np.float64)

    for t in range(n_obs):
        alpha_t = np.float64(alpha[t])
        decay = 1.0 - alpha_t

        moments_i *= decay
        moments_j *= decay
        co_moments *= decay
        pair_weights *= decay
        sum_weights *= decay
        weight_products *= decay**2

        # Add new values - track pairwise statistics for consistency
        for i in range(n_vars):
            obs[i] = np.float64(a[t, i]) - shift[i]
        for i in range(n_vars):
            new_val_i = obs[i]
            if np.isnan(new_val_i):
                continue

            for j in range(i, n_vars):
                new_val_j = obs[j]
                if np.isnan(new_val_j):
                    continue

                old_weight = sum_weights[i, j]
                new_weight = old_weight + 1.0
                delta_i = new_val_i - means_i[i, j]
                delta_j = new_val_j - means_j[i, j]
                adjustment = old_weight / new_weight

                means_i[i, j] += delta_i / new_weight
                means_j[i, j] += delta_j / new_weight
                moments_i[i, j] += adjustment * delta_i * delta_i
                moments_j[i, j] += adjustment * delta_j * delta_j
                co_moments[i, j] += adjustment * delta_i * delta_j
                pair_weights[i, j] += alpha_t
                weight_products[i, j] += 2.0 * old_weight
                sum_weights[i, j] = new_weight

        # Compute correlation matrix for current time step
        for i in range(n_vars):
            for j in range(i, n_vars):
                if pair_weights[i, j] >= min_weight and weight_products[i, j] > 0:
                    denominator = np.sqrt(moments_i[i, j] * moments_j[i, j])
                    if denominator == 0.0 or not np.isfinite(denominator):
                        denominator = np.sqrt(moments_i[i, j]) * np.sqrt(
                            moments_j[i, j]
                        )
                    if denominator > 0:
                        corr = co_moments[i, j] / denominator
                        if corr > 1.0:
                            corr = 1.0
                        elif corr < -1.0:
                            corr = -1.0
                        out[t, i, j] = corr
                        out[t, j, i] = corr
                    else:
                        out[t, i, j] = np.nan
                        out[t, j, i] = np.nan
                else:
                    out[t, i, j] = np.nan
                    out[t, j, i] = np.nan


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

    means_i = np.zeros((n_vars, n_vars), dtype=np.float64)
    means_j = np.zeros((n_vars, n_vars), dtype=np.float64)
    co_moments = np.zeros((n_vars, n_vars), dtype=np.float64)
    pair_weights = np.zeros((n_vars, n_vars), dtype=np.float64)
    sum_weights = np.zeros((n_vars, n_vars), dtype=np.float64)
    weight_products = np.zeros((n_vars, n_vars), dtype=np.float64)
    # Covariance is translation invariant. Subtracting an observed value keeps
    # Welford's evolving means small when the input has a large common offset.
    shift = _initial_shift(a)
    obs = np.empty(n_vars, dtype=np.float64)

    for t in range(n_obs):
        alpha_t = np.float64(alpha[t])
        decay = 1.0 - alpha_t

        co_moments *= decay
        pair_weights *= decay
        sum_weights *= decay
        weight_products *= decay**2

        # Add new values - track pairwise statistics for consistency
        for i in range(n_vars):
            obs[i] = np.float64(a[t, i]) - shift[i]
        for i in range(n_vars):
            new_val_i = obs[i]
            if np.isnan(new_val_i):
                continue

            for j in range(i, n_vars):
                new_val_j = obs[j]
                if np.isnan(new_val_j):
                    continue

                old_weight = sum_weights[i, j]
                new_weight = old_weight + 1.0
                delta_i = new_val_i - means_i[i, j]
                delta_j = new_val_j - means_j[i, j]
                adjustment = old_weight / new_weight

                means_i[i, j] += delta_i / new_weight
                means_j[i, j] += delta_j / new_weight
                co_moments[i, j] += adjustment * delta_i * delta_j
                pair_weights[i, j] += alpha_t
                weight_products[i, j] += 2.0 * old_weight
                sum_weights[i, j] = new_weight

        # Compute covariance matrix for current time step
        for i in range(n_vars):
            for j in range(i, n_vars):
                total_weight = sum_weights[i, j]
                if pair_weights[i, j] >= min_weight and weight_products[i, j] > 0:
                    cov = co_moments[i, j] * total_weight / weight_products[i, j]
                    out[t, i, j] = cov
                    out[t, j, i] = cov
                else:
                    out[t, i, j] = np.nan
                    out[t, j, i] = np.nan
