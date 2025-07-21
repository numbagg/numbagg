"""Moving window matrix functions using the decorator pattern."""

import numpy as np
from numba import float32, float64, int64

from .decorators import ndmoveexpmatrix, ndmovematrix

__all__ = [
    "move_corrmatrix",
    "move_covmatrix",
    "move_exp_nancorrmatrix",
    "move_exp_nancovmatrix",
]


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

    # Initialize running statistics
    sums = np.zeros(n_vars, dtype=a.dtype)
    sums_sq = np.zeros(n_vars, dtype=a.dtype)
    counts = np.zeros(n_vars, dtype=np.int64)

    # Initialize pairwise statistics
    prods = np.zeros((n_vars, n_vars), dtype=a.dtype)
    pair_counts = np.zeros((n_vars, n_vars), dtype=np.int64)

    for t in range(n_obs):
        # Remove old values when window slides
        if t >= window:
            for i in range(n_vars):
                old_val = a[t - window, i]
                if not np.isnan(old_val):
                    sums[i] -= old_val
                    sums_sq[i] -= old_val * old_val
                    counts[i] -= 1

                    # Update pairwise products
                    for j in range(n_vars):
                        old_val_j = a[t - window, j]
                        if not np.isnan(old_val_j):
                            prods[i, j] -= old_val * old_val_j
                            pair_counts[i, j] -= 1

        # Add new values
        for i in range(n_vars):
            new_val = a[t, i]
            if not np.isnan(new_val):
                sums[i] += new_val
                sums_sq[i] += new_val * new_val
                counts[i] += 1

                # Update pairwise products
                for j in range(n_vars):
                    new_val_j = a[t, j]
                    if not np.isnan(new_val_j):
                        prods[i, j] += new_val * new_val_j
                        pair_counts[i, j] += 1

        # Compute correlation matrix for current window
        for i in range(n_vars):
            for j in range(n_vars):
                # Compute correlation
                n = pair_counts[i, j]
                # Need at least 2 observations for correlation (to compute variance)
                if n >= max(min_count, 2) and counts[i] >= 2 and counts[j] >= 2:
                    mean_i = sums[i] / counts[i]
                    mean_j = sums[j] / counts[j]

                    # Compute variances
                    var_i = sums_sq[i] / counts[i] - mean_i * mean_i
                    var_j = sums_sq[j] / counts[j] - mean_j * mean_j

                    # Compute covariance
                    cov = prods[i, j] / n - (sums[i] / counts[i]) * (
                        sums[j] / counts[j]
                    )

                    # Compute correlation
                    if var_i > 0 and var_j > 0:
                        corr = cov / np.sqrt(var_i * var_j)
                        out[t, i, j] = corr
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

    # Initialize running statistics
    sums = np.zeros(n_vars, dtype=a.dtype)
    counts = np.zeros(n_vars, dtype=np.int64)

    # Initialize pairwise statistics
    prods = np.zeros((n_vars, n_vars), dtype=a.dtype)
    pair_counts = np.zeros((n_vars, n_vars), dtype=np.int64)

    for t in range(n_obs):
        # Remove old values when window slides
        if t >= window:
            for i in range(n_vars):
                old_val = a[t - window, i]
                if not np.isnan(old_val):
                    sums[i] -= old_val
                    counts[i] -= 1

                    # Update pairwise products
                    for j in range(n_vars):
                        old_val_j = a[t - window, j]
                        if not np.isnan(old_val_j):
                            prods[i, j] -= old_val * old_val_j
                            pair_counts[i, j] -= 1

        # Add new values
        for i in range(n_vars):
            new_val = a[t, i]
            if not np.isnan(new_val):
                sums[i] += new_val
                counts[i] += 1

                # Update pairwise products
                for j in range(n_vars):
                    new_val_j = a[t, j]
                    if not np.isnan(new_val_j):
                        prods[i, j] += new_val * new_val_j
                        pair_counts[i, j] += 1

        # Compute covariance matrix for current window
        for i in range(n_vars):
            for j in range(n_vars):
                n = pair_counts[i, j]
                if n >= min_count:
                    if n > 1:
                        # Unbiased covariance with ddof=1
                        cov = (prods[i, j] - sums[i] * sums[j] / n) / (n - 1)
                        out[t, i, j] = cov
                    else:
                        # n == 1, covariance is undefined (requires at least 2 points)
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

    # Initialize pairwise statistics - each (i,j) pair tracks its own statistics
    # This is necessary for consistency with non-matrix exponential functions
    sums_i = np.zeros(
        (n_vars, n_vars), dtype=a.dtype
    )  # sum of variable i for pair (i,j)
    sums_j = np.zeros(
        (n_vars, n_vars), dtype=a.dtype
    )  # sum of variable j for pair (i,j)
    sums_sq_i = np.zeros(
        (n_vars, n_vars), dtype=a.dtype
    )  # sum of squares of variable i for pair (i,j)
    sums_sq_j = np.zeros(
        (n_vars, n_vars), dtype=a.dtype
    )  # sum of squares of variable j for pair (i,j)
    prods = np.zeros((n_vars, n_vars), dtype=a.dtype)  # sum of products for pair (i,j)
    pair_weights = np.zeros(
        (n_vars, n_vars), dtype=a.dtype
    )  # accumulated alpha weights
    pair_sum_weights = np.zeros((n_vars, n_vars), dtype=a.dtype)  # count of valid pairs
    pair_sum_weights_sq = np.zeros(
        (n_vars, n_vars), dtype=a.dtype
    )  # sum of squared weights

    for t in range(n_obs):
        alpha_t = alpha[t]
        decay = 1.0 - alpha_t

        # Apply exponential decay to all pairwise statistics
        for i in range(n_vars):
            for j in range(n_vars):
                sums_i[i, j] *= decay
                sums_j[i, j] *= decay
                sums_sq_i[i, j] *= decay
                sums_sq_j[i, j] *= decay
                prods[i, j] *= decay
                pair_weights[i, j] *= decay
                pair_sum_weights[i, j] *= decay
                pair_sum_weights_sq[i, j] *= decay**2

        # Add new values - track pairwise statistics for consistency
        for i in range(n_vars):
            for j in range(n_vars):
                new_val_i = a[t, i]
                new_val_j = a[t, j]

                # Only update if BOTH values are non-NaN (consistent with non-matrix functions)
                if not (np.isnan(new_val_i) or np.isnan(new_val_j)):
                    # Update pairwise statistics
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

                    # Compute correlation
                    if var_i > 0 and var_j > 0:
                        corr = cov / np.sqrt(var_i * var_j)
                        out[t, i, j] = corr
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

    # Initialize pairwise statistics - each (i,j) pair tracks its own statistics
    # This is necessary for consistency with non-matrix exponential functions
    sums_i = np.zeros(
        (n_vars, n_vars), dtype=a.dtype
    )  # sum of variable i for pair (i,j)
    sums_j = np.zeros(
        (n_vars, n_vars), dtype=a.dtype
    )  # sum of variable j for pair (i,j)
    prods = np.zeros((n_vars, n_vars), dtype=a.dtype)  # sum of products for pair (i,j)
    pair_weights = np.zeros(
        (n_vars, n_vars), dtype=a.dtype
    )  # accumulated alpha weights
    pair_sum_weights = np.zeros((n_vars, n_vars), dtype=a.dtype)  # count of valid pairs
    pair_sum_weights_sq = np.zeros(
        (n_vars, n_vars), dtype=a.dtype
    )  # sum of squared weights

    for t in range(n_obs):
        alpha_t = alpha[t]
        decay = 1.0 - alpha_t

        # Apply exponential decay to all pairwise statistics
        for i in range(n_vars):
            for j in range(n_vars):
                sums_i[i, j] *= decay
                sums_j[i, j] *= decay
                prods[i, j] *= decay
                pair_weights[i, j] *= decay
                pair_sum_weights[i, j] *= decay
                pair_sum_weights_sq[i, j] *= decay**2

        # Add new values - track pairwise statistics for consistency
        for i in range(n_vars):
            for j in range(n_vars):
                new_val_i = a[t, i]
                new_val_j = a[t, j]

                # Only update if BOTH values are non-NaN (consistent with non-matrix functions)
                if not (np.isnan(new_val_i) or np.isnan(new_val_j)):
                    # Update pairwise statistics
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
