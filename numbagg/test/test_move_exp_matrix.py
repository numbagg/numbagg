"""Tests for exponential moving matrix functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from numbagg import move_exp_nancorrmatrix, move_exp_nancovmatrix


class TestMoveExpMatrixFunctions:
    """Test exponential moving matrix functions."""

    @pytest.mark.parametrize(
        "func,expected_diag",
        [
            (move_exp_nancorrmatrix, 1.0),  # Correlation has 1s on diagonal
            (move_exp_nancovmatrix, None),  # Covariance has variance on diagonal
        ],
    )
    def test_simple_matrix(self, func, expected_diag):
        """Test simple 2x2 matrix calculation with exponential decay."""
        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float64)
        alpha = 0.5
        result = func(data, alpha=alpha)

        # Check shape - should be (time, vars, vars)
        assert result.shape == (4, 2, 2)

        # Check diagonal at the end
        final_result = result[-1]
        if expected_diag is not None:
            assert_allclose(
                np.diag(final_result), [expected_diag, expected_diag], rtol=1e-10
            )
        else:
            # For covariance, just check diagonal is non-negative
            assert np.all(np.diag(final_result) >= 0)

        # Check symmetry at each time step
        for t in range(result.shape[0]):
            assert_allclose(result[t], result[t].T, rtol=1e-10)

        # For perfect linear relationship, correlation should be 1
        if func == move_exp_nancorrmatrix:
            # Check that off-diagonal elements approach 1 as we get more data
            assert_allclose(final_result, [[1.0, 1.0], [1.0, 1.0]], rtol=1e-10)

    @pytest.mark.parametrize("func", [move_exp_nancorrmatrix, move_exp_nancovmatrix])
    def test_with_nans(self, func):
        """Test with NaN values."""
        data = np.array(
            [[1, 2, np.nan, 4], [2, 4, 6, np.nan], [np.nan, 1, 2, 3]], dtype=np.float64
        )
        alpha = 0.3
        result = func(data, alpha=alpha)

        # Check shape
        assert result.shape == (4, 3, 3)

        # Should handle NaN gracefully - check that we get some finite values
        assert np.any(np.isfinite(result))

    @pytest.mark.parametrize("func", [move_exp_nancorrmatrix, move_exp_nancovmatrix])
    def test_single_variable(self, func):
        """Test with single variable (1x1 matrix)."""
        data = np.array([[1, 2, 3, 4]], dtype=np.float64)
        alpha = 0.4
        result = func(data, alpha=alpha)

        # Check shape
        assert result.shape == (4, 1, 1)

        # Diagonal should be 1 for correlation (once we have enough data), positive for covariance
        if func == move_exp_nancorrmatrix:
            # First time step might be NaN, but later ones should be 1.0
            assert (
                np.isnan(result[0, 0, 0]) or result[0, 0, 0] == 1.0
            )  # First might be NaN
            assert_allclose(
                result[1:, 0, 0], [1.0, 1.0, 1.0], rtol=1e-10
            )  # Later should be 1
        else:
            # For covariance, check finite values are non-negative
            finite_mask = np.isfinite(result[:, 0, 0])
            assert np.all(result[finite_mask, 0, 0] >= 0)

    @pytest.mark.parametrize("func", [move_exp_nancorrmatrix, move_exp_nancovmatrix])
    def test_min_weight(self, func):
        """Test min_weight parameter."""
        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float64)
        alpha = 0.1  # Low alpha means slow buildup of weight

        # High min_weight should produce more NaNs initially
        result_high = func(data, alpha=alpha, min_weight=0.8)
        result_low = func(data, alpha=alpha, min_weight=0.1)

        # Check that high min_weight produces more NaNs initially
        nan_count_high = np.sum(np.isnan(result_high[0]))
        nan_count_low = np.sum(np.isnan(result_low[0]))
        assert nan_count_high >= nan_count_low

    @pytest.mark.parametrize("func", [move_exp_nancorrmatrix, move_exp_nancovmatrix])
    def test_different_alphas(self, func):
        """Test behavior with different alpha values."""
        data = np.array([[1, 2, 3, 4, 5], [1, 4, 9, 16, 25]], dtype=np.float64)

        # High alpha (fast decay) vs low alpha (slow decay)
        result_high = func(data, alpha=0.9)
        result_low = func(data, alpha=0.1)

        # Both should have same shape
        assert result_high.shape == result_low.shape == (5, 2, 2)

        # Results should be different
        assert not np.allclose(result_high[-1], result_low[-1], rtol=1e-3)

    def test_correlation_bounds(self):
        """Test that correlation values are properly bounded between -1 and 1."""
        # Create data with some negative correlation
        np.random.seed(42)
        data = np.random.randn(3, 100)
        data[1] = -data[0] + 0.1 * np.random.randn(100)  # Strong negative correlation

        result = move_exp_nancorrmatrix(data, alpha=0.5)

        # All correlation values should be between -1 and 1
        finite_mask = np.isfinite(result)
        assert np.all(result[finite_mask] >= -1.0)
        assert np.all(result[finite_mask] <= 1.0)

    def test_exponential_decay_property(self):
        """Test that older observations have less influence (exponential decay)."""
        # Create a dataset where values change over time
        np.random.seed(42)  # For reproducibility
        early_data = np.random.randn(2, 20)
        late_data = np.random.randn(2, 20) + 10  # Different mean
        data = np.concatenate([early_data, late_data], axis=1)

        # With high alpha, recent values should dominate more than low alpha
        result_high = move_exp_nancovmatrix(data, alpha=0.9)
        result_low = move_exp_nancovmatrix(data, alpha=0.1)

        # The final covariance matrices should be different
        # (high alpha should weight recent data more)
        final_cov_high = result_high[-1]
        final_cov_low = result_low[-1]

        # Results should be different due to different weighting
        assert not np.allclose(final_cov_high, final_cov_low, rtol=1e-3)

    @pytest.mark.parametrize("func", [move_exp_nancorrmatrix, move_exp_nancovmatrix])
    def test_array_alpha(self, func):
        """Test with alpha as an array rather than scalar."""
        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float64)
        alpha_array = np.array([0.1, 0.5, 0.9, 0.3])

        result = func(data, alpha=alpha_array)

        # Should work and produce expected shape
        assert result.shape == (4, 2, 2)

        # Should be different from constant alpha
        result_constant = func(data, alpha=0.5)
        assert not np.allclose(result, result_constant)

    def test_all_nan_input(self):
        """Test behavior with all-NaN input."""
        data = np.full((2, 4), np.nan, dtype=np.float64)

        result_corr = move_exp_nancorrmatrix(data, alpha=0.5)
        result_cov = move_exp_nancovmatrix(data, alpha=0.5)

        # All results should be NaN
        assert np.all(np.isnan(result_corr))
        assert np.all(np.isnan(result_cov))

    def test_single_valid_observation(self):
        """Test with only one valid observation per variable."""
        data = np.array(
            [[1.0, np.nan, np.nan, np.nan], [np.nan, 2.0, np.nan, np.nan]],
            dtype=np.float64,
        )

        result_corr = move_exp_nancorrmatrix(data, alpha=0.5)
        result_cov = move_exp_nancovmatrix(data, alpha=0.5)

        # Should have shape (4, 2, 2)
        assert result_corr.shape == (4, 2, 2)
        assert result_cov.shape == (4, 2, 2)

        # Most values should be NaN since we need at least 2 observations for correlation/covariance
        assert np.sum(np.isnan(result_corr)) > np.sum(np.isfinite(result_corr))
        assert np.sum(np.isnan(result_cov)) > np.sum(np.isfinite(result_cov))
