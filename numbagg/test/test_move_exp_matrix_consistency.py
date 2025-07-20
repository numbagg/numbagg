"""Test consistency between move_exp matrix functions and their non-matrix counterparts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from numbagg import (
    move_exp_nancorr,
    move_exp_nancorrmatrix,
    move_exp_nancov,
    move_exp_nancovmatrix,
)


class TestMoveExpMatrixConsistency:
    """Test that matrix functions match non-matrix functions for pairwise cases."""

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
    def test_covariance_consistency(self, alpha):
        """Test that move_exp_nancovmatrix matches move_exp_nancov for pairs."""
        np.random.seed(42)

        # Create two time series
        n_obs = 50
        a1 = np.random.randn(n_obs)
        a2 = np.random.randn(n_obs) * 2 + 1

        # Compute using non-matrix function
        cov_nonmatrix = move_exp_nancov(a1, a2, alpha=alpha)

        # Compute using matrix function
        data_matrix = np.array([a1, a2])
        cov_matrix_result = move_exp_nancovmatrix(data_matrix, alpha=alpha)

        # Extract the off-diagonal element (covariance between a1 and a2)
        cov_from_matrix = cov_matrix_result[:, 0, 1]

        # They should match
        assert_allclose(cov_nonmatrix, cov_from_matrix, rtol=1e-10)

        # Also check symmetry - (0,1) should equal (1,0)
        assert_allclose(
            cov_matrix_result[:, 0, 1], cov_matrix_result[:, 1, 0], rtol=1e-10
        )

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
    def test_correlation_consistency(self, alpha):
        """Test that move_exp_nancorrmatrix matches move_exp_nancorr for pairs."""
        np.random.seed(123)

        # Create two time series
        n_obs = 50
        a1 = np.random.randn(n_obs)
        a2 = np.random.randn(n_obs) * 1.5 + 0.5

        # Compute using non-matrix function
        corr_nonmatrix = move_exp_nancorr(a1, a2, alpha=alpha)

        # Compute using matrix function
        data_matrix = np.array([a1, a2])
        corr_matrix_result = move_exp_nancorrmatrix(data_matrix, alpha=alpha)

        # Extract the off-diagonal element (correlation between a1 and a2)
        corr_from_matrix = corr_matrix_result[:, 0, 1]

        # They should match
        assert_allclose(corr_nonmatrix, corr_from_matrix, rtol=1e-10)

        # Also check symmetry - (0,1) should equal (1,0)
        assert_allclose(
            corr_matrix_result[:, 0, 1], corr_matrix_result[:, 1, 0], rtol=1e-10
        )

    def test_covariance_with_nans(self):
        """Test consistency with NaN values."""
        np.random.seed(456)

        # Create two time series with some NaN values
        n_obs = 30
        a1 = np.random.randn(n_obs)
        a2 = np.random.randn(n_obs) * 2

        # Add some NaN values
        a1[5:8] = np.nan
        a2[15:17] = np.nan

        alpha = 0.3

        # Compute using non-matrix function
        cov_nonmatrix = move_exp_nancov(a1, a2, alpha=alpha)

        # Compute using matrix function
        data_matrix = np.array([a1, a2])
        cov_matrix_result = move_exp_nancovmatrix(data_matrix, alpha=alpha)
        cov_from_matrix = cov_matrix_result[:, 0, 1]

        # They should match
        assert_allclose(cov_nonmatrix, cov_from_matrix, rtol=1e-10)

    def test_correlation_with_nans(self):
        """Test correlation consistency with NaN values."""
        np.random.seed(789)

        # Create two time series with some NaN values
        n_obs = 30
        a1 = np.random.randn(n_obs)
        a2 = np.random.randn(n_obs) * 1.2 + 0.3

        # Add some NaN values
        a1[3:6] = np.nan
        a2[12:15] = np.nan

        alpha = 0.4

        # Compute using non-matrix function
        corr_nonmatrix = move_exp_nancorr(a1, a2, alpha=alpha)

        # Compute using matrix function
        data_matrix = np.array([a1, a2])
        corr_matrix_result = move_exp_nancorrmatrix(data_matrix, alpha=alpha)
        corr_from_matrix = corr_matrix_result[:, 0, 1]

        # They should match
        assert_allclose(corr_nonmatrix, corr_from_matrix, rtol=1e-10)

    def test_array_alpha_consistency(self):
        """Test consistency when alpha is an array."""
        np.random.seed(999)

        # Create two time series
        n_obs = 20
        a1 = np.random.randn(n_obs)
        a2 = np.random.randn(n_obs) * 0.8 + 0.2

        # Create varying alpha
        alpha_array = np.linspace(0.1, 0.9, n_obs)

        # Compute using non-matrix functions
        cov_nonmatrix = move_exp_nancov(a1, a2, alpha=alpha_array)
        corr_nonmatrix = move_exp_nancorr(a1, a2, alpha=alpha_array)

        # Compute using matrix functions
        data_matrix = np.array([a1, a2])
        cov_matrix_result = move_exp_nancovmatrix(data_matrix, alpha=alpha_array)
        corr_matrix_result = move_exp_nancorrmatrix(data_matrix, alpha=alpha_array)

        # Extract off-diagonal elements
        cov_from_matrix = cov_matrix_result[:, 0, 1]
        corr_from_matrix = corr_matrix_result[:, 0, 1]

        # They should match
        assert_allclose(cov_nonmatrix, cov_from_matrix, rtol=1e-10)
        assert_allclose(corr_nonmatrix, corr_from_matrix, rtol=1e-10)

    def test_min_weight_consistency(self):
        """Test consistency with different min_weight values."""
        np.random.seed(111)

        # Create two time series
        n_obs = 25
        a1 = np.random.randn(n_obs)
        a2 = np.random.randn(n_obs) * 1.3

        alpha = 0.2  # Low alpha to test min_weight effects
        min_weight = 0.5

        # Compute using non-matrix functions
        cov_nonmatrix = move_exp_nancov(a1, a2, alpha=alpha, min_weight=min_weight)
        corr_nonmatrix = move_exp_nancorr(a1, a2, alpha=alpha, min_weight=min_weight)

        # Compute using matrix functions
        data_matrix = np.array([a1, a2])
        cov_matrix_result = move_exp_nancovmatrix(
            data_matrix, alpha=alpha, min_weight=min_weight
        )
        corr_matrix_result = move_exp_nancorrmatrix(
            data_matrix, alpha=alpha, min_weight=min_weight
        )

        # Extract off-diagonal elements
        cov_from_matrix = cov_matrix_result[:, 0, 1]
        corr_from_matrix = corr_matrix_result[:, 0, 1]

        # They should match
        assert_allclose(cov_nonmatrix, cov_from_matrix, rtol=1e-10)
        assert_allclose(corr_nonmatrix, corr_from_matrix, rtol=1e-10)

    def test_perfect_correlation_consistency(self):
        """Test with perfectly correlated data."""
        np.random.seed(222)

        # Create perfectly correlated data
        n_obs = 15
        a1 = np.random.randn(n_obs)
        a2 = 2 * a1 + 1  # Perfect linear relationship

        alpha = 0.6

        # Compute using non-matrix function
        corr_nonmatrix = move_exp_nancorr(a1, a2, alpha=alpha)

        # Compute using matrix function
        data_matrix = np.array([a1, a2])
        corr_matrix_result = move_exp_nancorrmatrix(data_matrix, alpha=alpha)
        corr_from_matrix = corr_matrix_result[:, 0, 1]

        # They should match and approach 1.0
        assert_allclose(corr_nonmatrix, corr_from_matrix, rtol=1e-10)

        # For later time points with perfect correlation, should be close to 1.0
        final_corr = corr_from_matrix[-1]
        assert abs(final_corr - 1.0) < 1e-10

    def test_single_series_diagonal_consistency(self):
        """Test that diagonal elements match what we'd expect for a single series."""
        np.random.seed(333)

        # Create a single time series
        n_obs = 20
        a1 = np.random.randn(n_obs)

        alpha = 0.4

        # For correlation, diagonal should always be 1.0 (after enough data)
        data_matrix = np.array([a1])
        corr_matrix_result = move_exp_nancorrmatrix(data_matrix, alpha=alpha)

        # Diagonal elements should be 1.0 where finite
        diagonal_values = corr_matrix_result[:, 0, 0]
        finite_mask = np.isfinite(diagonal_values)
        assert_allclose(diagonal_values[finite_mask], 1.0, rtol=1e-10)

    def test_three_series_consistency(self):
        """Test consistency for a 3x3 matrix case."""
        np.random.seed(444)

        # Create three time series
        n_obs = 30
        a1 = np.random.randn(n_obs)
        a2 = np.random.randn(n_obs) * 1.5 + 0.5
        a3 = np.random.randn(n_obs) * 0.8 - 0.2

        alpha = 0.35

        # Test all pairwise combinations
        pairs = [(a1, a2, 0, 1), (a1, a3, 0, 2), (a2, a3, 1, 2)]

        # Compute matrix result once
        data_matrix = np.array([a1, a2, a3])
        cov_matrix_result = move_exp_nancovmatrix(data_matrix, alpha=alpha)
        corr_matrix_result = move_exp_nancorrmatrix(data_matrix, alpha=alpha)

        for series1, series2, i, j in pairs:
            # Compute pairwise results
            cov_nonmatrix = move_exp_nancov(series1, series2, alpha=alpha)
            corr_nonmatrix = move_exp_nancorr(series1, series2, alpha=alpha)

            # Extract from matrix results
            cov_from_matrix = cov_matrix_result[:, i, j]
            corr_from_matrix = corr_matrix_result[:, i, j]

            # They should match
            assert_allclose(
                cov_nonmatrix,
                cov_from_matrix,
                rtol=1e-10,
                err_msg=f"Covariance mismatch for series {i},{j}",
            )
            assert_allclose(
                corr_nonmatrix,
                corr_from_matrix,
                rtol=1e-10,
                err_msg=f"Correlation mismatch for series {i},{j}",
            )

            # Also check symmetry
            assert_allclose(
                cov_matrix_result[:, i, j], cov_matrix_result[:, j, i], rtol=1e-10
            )
            assert_allclose(
                corr_matrix_result[:, i, j], corr_matrix_result[:, j, i], rtol=1e-10
            )
