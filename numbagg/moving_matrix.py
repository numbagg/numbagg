"""Moving window matrix functions using the decorator pattern."""

import numpy as np
from numba import float32, float64, int64

from .decorators import ndmovematrix

__all__ = ["move_nancorrmatrix", "move_nancovmatrix"]


@ndmovematrix.wrap(
    signature=(
        [
            (float32[:, :], int64, int64, float32[:, :, :]),
            (float64[:, :], int64, int64, float64[:, :, :]),
        ],
        "(n,m),(),()->(m,n,n)",
    )
)
def move_nancorrmatrix(a, window, min_count, out):
    """
    Moving window correlation matrix gufunc.

    For 2D input, correlates variables (rows) across observations (columns in the window).
    """
    n_vars = a.shape[0]
    n_obs = a.shape[1]
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
                old_val = a[i, t - window]
                if not np.isnan(old_val):
                    sums[i] -= old_val
                    sums_sq[i] -= old_val * old_val
                    counts[i] -= 1

                    # Update pairwise products
                    for j in range(n_vars):
                        old_val_j = a[j, t - window]
                        if not np.isnan(old_val_j):
                            prods[i, j] -= old_val * old_val_j
                            pair_counts[i, j] -= 1

        # Add new values
        for i in range(n_vars):
            new_val = a[i, t]
            if not np.isnan(new_val):
                sums[i] += new_val
                sums_sq[i] += new_val * new_val
                counts[i] += 1

                # Update pairwise products
                for j in range(n_vars):
                    new_val_j = a[j, t]
                    if not np.isnan(new_val_j):
                        prods[i, j] += new_val * new_val_j
                        pair_counts[i, j] += 1

        # Compute correlation matrix for current window
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    # Diagonal is 1 when we have enough data for correlation
                    # Correlation requires at least 2 observations to compute variance
                    if counts[i] >= max(min_count, 2):
                        out[t, i, j] = 1.0
                    else:
                        out[t, i, j] = np.nan
                else:
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
                            # Clamp to [-1, 1] for numerical stability
                            out[t, i, j] = max(-1.0, min(1.0, corr))
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
        "(n,m),(),()->(m,n,n)",
    )
)
def move_nancovmatrix(a, window, min_count, out):
    """
    Moving window covariance matrix gufunc.

    For 2D input, computes covariance between variables (rows) across observations (columns in the window).
    """
    n_vars = a.shape[0]
    n_obs = a.shape[1]
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
                old_val = a[i, t - window]
                if not np.isnan(old_val):
                    sums[i] -= old_val
                    counts[i] -= 1

                    # Update pairwise products
                    for j in range(n_vars):
                        old_val_j = a[j, t - window]
                        if not np.isnan(old_val_j):
                            prods[i, j] -= old_val * old_val_j
                            pair_counts[i, j] -= 1

        # Add new values
        for i in range(n_vars):
            new_val = a[i, t]
            if not np.isnan(new_val):
                sums[i] += new_val
                counts[i] += 1

                # Update pairwise products
                for j in range(n_vars):
                    new_val_j = a[j, t]
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
