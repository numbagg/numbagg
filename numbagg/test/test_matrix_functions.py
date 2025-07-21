"""Tests for all matrix functions (correlation and covariance matrices)."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from numbagg import (
    move_exp_nancorr,
    move_exp_nancorrmatrix,
    move_exp_nancov,
    move_exp_nancovmatrix,
    move_nancorrmatrix,
    move_nancovmatrix,
    nancorrmatrix,
    nancovmatrix,
)


class TestCorrelationCovarianceMatrices:
    """Test correlation and covariance matrix functions (nancorrmatrix, nancovmatrix)."""

    @pytest.mark.parametrize(
        "func,expected_diag",
        [
            (nancorrmatrix, 1.0),  # Correlation has 1s on diagonal
            (nancovmatrix, None),  # Covariance has variance on diagonal
        ],
    )
    def test_simple_matrix(self, func, expected_diag):
        """Test simple 2x2 matrix calculation."""
        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float64)
        result = func(data)

        # Check shape
        assert result.shape == (2, 2)

        # Check diagonal
        if expected_diag is not None:
            assert_allclose(np.diag(result), [expected_diag, expected_diag])
        else:
            # For covariance, just check diagonal is non-negative
            assert np.all(np.diag(result) >= 0)

        # Check symmetry
        assert_allclose(result, result.T)

        # For perfect linear relationship, correlation should be 1
        if func == nancorrmatrix:
            assert_allclose(result, [[1.0, 1.0], [1.0, 1.0]])

    @pytest.mark.parametrize("func", [nancorrmatrix, nancovmatrix])
    def test_with_nans(self, func):
        """Test with NaN values."""
        data = np.array(
            [[1, 2, np.nan, 4], [2, 4, 6, np.nan], [np.nan, 1, 2, 3]], dtype=np.float64
        )
        result = func(data)

        # Check shape and symmetry
        assert result.shape == (3, 3)
        assert_allclose(result, result.T, equal_nan=True)

        # For correlation, check diagonal is 1 where not NaN
        if func == nancorrmatrix:
            assert_allclose(np.diag(result), [1.0, 1.0, 1.0])

    @pytest.mark.parametrize("func", [nancorrmatrix, nancovmatrix])
    def test_1d_array_raises_error(self, func):
        """Test that 1D arrays raise an appropriate error."""
        data_1d = np.array([1, 2, 3, 4, 5], dtype=np.float64)

        with pytest.raises(ValueError, match="requires at least a 2D array"):
            func(data_1d)

    @pytest.mark.parametrize("func", [nancorrmatrix, nancovmatrix])
    def test_comparison_with_numpy(self, func):
        """Compare with numpy's implementation for data without NaNs."""
        np.random.seed(42)
        data = np.random.randn(5, 100)

        result = func(data)
        if func == nancorrmatrix:
            expected = np.corrcoef(data)
        else:
            expected = np.cov(data)

        assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("func", [nancorrmatrix, nancovmatrix])
    def test_constant_variables(self, func):
        """Test with constant (zero variance) variables."""
        data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 3, 4]], dtype=np.float64)
        result = func(data)

        if func == nancorrmatrix:
            # Correlation with constant variables should be NaN
            assert np.isnan(result[0, 1])  # Two constants
            assert np.isnan(result[0, 2])  # Constant with non-constant
            assert result[2, 2] == 1.0  # Variable with itself
        else:
            # Covariance of constants should be 0
            assert result[0, 0] == 0.0
            assert result[1, 1] == 0.0
            assert result[0, 1] == 0.0

    @pytest.mark.parametrize("func", [nancorrmatrix, nancovmatrix])
    def test_broadcasting_higher_dims(self, func):
        """Test that gufunc broadcasting works correctly for higher dimensional arrays."""
        np.random.seed(42)

        # 3D array: (2, 4, 10) -> broadcast dims (2,) + core dims (4, 10)
        data_3d = np.random.randn(2, 4, 10)
        result_3d = func(data_3d)
        assert result_3d.shape == (2, 4, 4)

        # 4D array: (2, 3, 4, 10) -> broadcast dims (2, 3) + core dims (4, 10)
        data_4d = np.random.randn(2, 3, 4, 10)
        result_4d = func(data_4d)
        assert result_4d.shape == (2, 3, 4, 4)

        # Check each broadcast element is valid
        for i in range(2):
            for j in range(3):
                matrix = result_4d[i, j]
                # Check symmetry
                assert_allclose(matrix, matrix.T, rtol=1e-10)

                if func == nancorrmatrix:
                    # Check diagonal is 1
                    assert_allclose(np.diag(matrix), np.ones(4), rtol=1e-10)
                    # Check bounds
                    assert np.all((matrix >= -1) & (matrix <= 1))
                else:
                    # Check diagonal (variance) is non-negative
                    assert np.all(np.diag(matrix) >= 0)

        # Verify correctness - each slice should match individual computation
        for i in range(2):
            single_result = func(data_3d[i])
            assert_allclose(result_3d[i], single_result, rtol=1e-10)

    def test_correlation_covariance_relationship(self):
        """Test relationship between correlation and covariance."""
        np.random.seed(42)
        data = np.random.randn(4, 50)

        cov_matrix = nancovmatrix(data)
        corr_matrix = nancorrmatrix(data)

        # Correlation = Covariance / (std_i * std_j)
        stds = np.sqrt(np.diag(cov_matrix))
        expected_corr = cov_matrix / np.outer(stds, stds)

        assert_allclose(corr_matrix, expected_corr, rtol=1e-10)

    def test_no_axis_parameter_accepted(self):
        """Test that axis parameter is no longer accepted."""
        data = np.random.randn(3, 100)

        # These should all raise TypeError since axis parameter removed
        with pytest.raises(TypeError):
            nancorrmatrix(data, axis=0)

        with pytest.raises(TypeError):
            nancorrmatrix(data, axis=-1)

        with pytest.raises(TypeError):
            nancovmatrix(data, axis=1)

    def test_fixed_dimensional_conventions(self):
        """Test fixed dimensional conventions: (..., vars, obs) -> (..., vars, vars)."""
        np.random.seed(42)

        # Basic test: (vars, obs) -> (vars, vars)
        data = np.random.randn(3, 100)  # (vars, obs)
        corr_result = nancorrmatrix(data)
        cov_result = nancovmatrix(data)

        assert corr_result.shape == (3, 3)
        assert cov_result.shape == (3, 3)

        # Broadcasting test: (batch, vars, obs) -> (batch, vars, vars)
        data_3d = np.random.randn(2, 3, 100)  # (batch, vars, obs)
        corr_3d = nancorrmatrix(data_3d)
        cov_3d = nancovmatrix(data_3d)

        assert corr_3d.shape == (2, 3, 3)
        assert cov_3d.shape == (2, 3, 3)


class TestMovingMatrices:
    """Test moving window matrix functions (move_nancorrmatrix, move_nancovmatrix)."""

    @pytest.mark.parametrize(
        "move_func,static_func,window",
        [
            (move_nancorrmatrix, nancorrmatrix, 3),
            (move_nancovmatrix, nancovmatrix, 3),
        ],
    )
    def test_rolling_simple(self, move_func, static_func, window):
        """Test rolling functions with simple data."""
        # Moving functions expect (obs, vars) format
        data = np.array(
            [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12]], dtype=np.float64
        )
        result = move_func(data, window=window, min_count=2)

        # Shape should be (n_obs, n_vars, n_vars)
        assert result.shape == (6, 2, 2)

        # Check symmetry for each time point
        for t in range(6):
            assert_allclose(result[t], result[t].T, equal_nan=True)

        # For perfect linear relationship, correlation should be 1
        if move_func == move_nancorrmatrix:
            for i in range(1, 6):  # From second window onwards (min_count=2)
                assert_allclose(result[i], [[1.0, 1.0], [1.0, 1.0]], rtol=1e-10)

    @pytest.mark.parametrize(
        "move_func,window",
        [(move_nancorrmatrix, 5), (move_nancovmatrix, 5)],
    )
    def test_rolling_comparison_with_pandas(self, move_func, window):
        """Compare rolling functions with pandas."""
        np.random.seed(42)
        n_vars = 4
        n_obs = 20
        # Moving functions expect (obs, vars) format
        data = np.random.randn(n_obs, n_vars)

        # NumBagg result
        numbagg_result = move_func(data, window=window, min_count=window)

        # Pandas result - data is already in (obs, vars) format that pandas expects
        df = pd.DataFrame(data)
        if move_func == move_nancorrmatrix:
            pandas_result = df.rolling(window, min_periods=window).corr()
        else:
            pandas_result = df.rolling(window, min_periods=window).cov()

        # Compare each window
        for t in range(window - 1, n_obs):
            # Extract pandas matrix for this timepoint
            pandas_matrix = pandas_result.loc[t].values
            # Compare
            assert_allclose(numbagg_result[t], pandas_matrix, rtol=1e-10)

    @pytest.mark.parametrize("move_func", [move_nancorrmatrix, move_nancovmatrix])
    def test_rolling_1d_array_raises_error(self, move_func):
        """Test that 1D arrays raise an appropriate error for rolling functions."""
        data_1d = np.array([1, 2, 3, 4, 5], dtype=np.float64)

        with pytest.raises(ValueError, match="requires at least a 2D array"):
            move_func(data_1d, window=3)

    @pytest.mark.parametrize(
        "move_func,expected_diag",
        [
            (move_nancorrmatrix, 1.0),
            (move_nancovmatrix, None),
        ],
    )
    def test_rolling_zero_variance_windows(self, move_func, expected_diag):
        """Test rolling windows with zero variance."""
        # Moving functions expect (obs, vars) format
        data = np.array(
            [[1, 2], [1, 2], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float64
        )
        result = move_func(data, window=3, min_count=2)

        # First full window has constant values
        if move_func == move_nancorrmatrix:
            # Correlation undefined for zero variance
            assert np.isnan(result[2, 0, 1])
        else:
            # Covariance should be 0
            assert result[2, 0, 0] == 0.0
            assert result[2, 1, 1] == 0.0
            assert result[2, 0, 1] == 0.0

        # Later windows have variance
        assert not np.all(np.isnan(result[5]))

    @pytest.mark.parametrize("move_func", [move_nancorrmatrix, move_nancovmatrix])
    def test_rolling_broadcasting_higher_dims(self, move_func):
        """Test that rolling functions broadcast correctly over higher dimensions."""
        np.random.seed(42)
        window = 5

        # 3D array: (2, 20, 4) -> output (2, 20, 4, 4) - moving functions expect (..., obs, vars)
        data_3d = np.random.randn(2, 20, 4)
        result_3d = move_func(data_3d, window=window)
        assert result_3d.shape == (2, 20, 4, 4)

        # 4D array: (2, 3, 20, 4) -> output (2, 3, 20, 4, 4) - moving functions expect (..., obs, vars)
        data_4d = np.random.randn(2, 3, 20, 4)
        result_4d = move_func(data_4d, window=window)
        assert result_4d.shape == (2, 3, 20, 4, 4)

        # Verify correctness - each slice should match individual computation
        for i in range(2):
            single_result = move_func(data_3d[i], window=window)
            assert_allclose(result_3d[i], single_result, rtol=1e-10)

    def test_consistency_vs_basic_matrix_full_window(self):
        """Test consistency between moving and basic matrix functions when using full window."""
        np.random.seed(42)

        # Create data in both conventions
        data_moving = np.random.randn(50, 4)  # (obs, vars) for moving
        data_basic = data_moving.T  # (vars, obs) for basic

        # Basic correlation using all data
        corr_basic = nancorrmatrix(data_basic)

        # Moving correlation using full window
        corr_moving = move_nancorrmatrix(data_moving, window=50)

        # Last timestep should match basic result
        assert_allclose(corr_moving[-1], corr_basic, rtol=1e-14)

    def test_fixed_dimensional_conventions(self):
        """Test fixed dimensional conventions: (..., obs, vars) -> (..., obs, vars, vars)."""
        np.random.seed(42)

        # Basic test: (obs, vars) -> (obs, vars, vars)
        data_moving = np.random.randn(100, 3)  # (obs, vars)
        corr_moving = move_nancorrmatrix(data_moving, window=10)
        cov_moving = move_nancovmatrix(data_moving, window=10)

        assert corr_moving.shape == (100, 3, 3)
        assert cov_moving.shape == (100, 3, 3)

        # Broadcasting test: (batch, obs, vars) -> (batch, obs, vars, vars)
        data_moving_3d = np.random.randn(2, 100, 3)  # (batch, obs, vars)
        corr_moving_3d = move_nancorrmatrix(data_moving_3d, window=10)

        assert corr_moving_3d.shape == (2, 100, 3, 3)


class TestExponentialMatrices:
    """Test exponential moving matrix functions (move_exp_nancorrmatrix, move_exp_nancovmatrix)."""

    @pytest.mark.parametrize(
        "func,expected_diag",
        [
            (move_exp_nancorrmatrix, 1.0),  # Correlation has 1s on diagonal
            (move_exp_nancovmatrix, None),  # Covariance has variance on diagonal
        ],
    )
    def test_simple_matrix(self, func, expected_diag):
        """Test simple 2x2 matrix calculation with exponential decay."""
        # Exponential moving functions expect (obs, vars) format
        data = np.array([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=np.float64)
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
        # Exponential moving functions expect (obs, vars) format
        data = np.array(
            [[1, 2, np.nan], [2, 4, 1], [np.nan, 6, 2], [4, np.nan, 3]],
            dtype=np.float64,
        )
        alpha = 0.3
        result = func(data, alpha=alpha)

        # Check shape
        assert result.shape == (4, 3, 3)

        # Should handle NaN gracefully - check that we get some finite values
        assert np.any(np.isfinite(result))

    @pytest.mark.parametrize("func", [move_exp_nancorrmatrix, move_exp_nancovmatrix])
    def test_min_weight(self, func):
        """Test min_weight parameter."""
        # Exponential moving functions expect (obs, vars) format
        data = np.array([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=np.float64)
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
        # Exponential moving functions expect (obs, vars) format
        data = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25]], dtype=np.float64)

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
        # Exponential moving functions expect (obs, vars) format
        np.random.seed(42)
        data = np.random.randn(100, 3)
        data[:, 1] = -data[:, 0] + 0.1 * np.random.randn(
            100
        )  # Strong negative correlation

        result = move_exp_nancorrmatrix(data, alpha=0.5)

        # All correlation values should be between -1 and 1
        finite_mask = np.isfinite(result)
        assert np.all(result[finite_mask] >= -1.0)
        assert np.all(result[finite_mask] <= 1.0)

    @pytest.mark.parametrize("func", [move_exp_nancorrmatrix, move_exp_nancovmatrix])
    def test_array_alpha(self, func):
        """Test with alpha as an array rather than scalar."""
        # Exponential moving functions expect (obs, vars) format
        data = np.array([[1, 2], [2, 4], [3, 6], [4, 8]], dtype=np.float64)
        alpha_array = np.array([0.1, 0.5, 0.9, 0.3])

        result = func(data, alpha=alpha_array)

        # Should work and produce expected shape
        assert result.shape == (4, 2, 2)

        # Should be different from constant alpha
        result_constant = func(data, alpha=0.5)
        assert not np.allclose(result, result_constant)

    @pytest.mark.parametrize("func", [move_exp_nancorrmatrix, move_exp_nancovmatrix])
    def test_broadcasting_higher_dims(self, func):
        """Test that exponential matrix functions broadcast correctly for higher dimensional arrays."""
        np.random.seed(42)

        # 3D array: (2, 20, 4) -> broadcast dims (2,) + core dims (20, 4) = (obs, vars)
        data_3d = np.random.randn(2, 20, 4)
        result_3d = func(data_3d, alpha=0.3)
        assert result_3d.shape == (2, 20, 4, 4)

        # 4D array: (2, 3, 15, 4) -> broadcast dims (2, 3) + core dims (15, 4) = (obs, vars)
        data_4d = np.random.randn(2, 3, 15, 4)
        result_4d = func(data_4d, alpha=0.3)
        assert result_4d.shape == (2, 3, 15, 4, 4)

        # Verify correctness - each slice should match individual computation
        for i in range(2):
            single_result = func(data_3d[i], alpha=0.3)
            assert_allclose(result_3d[i], single_result, rtol=1e-10, equal_nan=True)

    def test_positive_semidefinite_covariance(self):
        """Test that covariance matrices are positive semi-definite."""
        np.random.seed(42)
        # Exponential moving functions expect (obs, vars) format
        data = np.random.randn(50, 4)

        result = move_exp_nancovmatrix(data, alpha=0.3)

        # Check that all finite covariance matrices are positive semi-definite
        for t in range(result.shape[0]):
            cov_matrix = result[t]
            if not np.any(np.isnan(cov_matrix)):
                # Compute eigenvalues
                eigenvals = np.linalg.eigvals(cov_matrix)
                # All eigenvalues should be non-negative (allowing small numerical errors)
                assert np.all(eigenvals >= -1e-10), (
                    f"Negative eigenvalue found at time {t}: {eigenvals.min()}"
                )

    def test_correlation_matrix_properties(self):
        """Test mathematical properties of correlation matrices."""
        np.random.seed(123)
        # Exponential moving functions expect (obs, vars) format
        data = np.random.randn(30, 3)

        result = move_exp_nancorrmatrix(data, alpha=0.4)

        for t in range(result.shape[0]):
            corr_matrix = result[t]
            if not np.any(np.isnan(corr_matrix)):
                # 1. Diagonal should be 1.0
                assert_allclose(np.diag(corr_matrix), 1.0, rtol=1e-12)

                # 2. Matrix should be symmetric
                assert_allclose(corr_matrix, corr_matrix.T, rtol=1e-12)

                # 3. All values should be very close to [-1, 1] (allowing for floating-point precision)
                assert np.all(corr_matrix >= -1.0 - 1e-10)
                assert np.all(corr_matrix <= 1.0 + 1e-10)

                # 4. Should be positive semi-definite
                eigenvals = np.linalg.eigvals(corr_matrix)
                assert np.all(eigenvals >= -1e-10), (
                    f"Correlation matrix not PSD at time {t}"
                )

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
    def test_consistency_with_pairwise_functions(self, alpha):
        """Test that matrix functions match non-matrix functions for pairwise cases."""
        np.random.seed(42)

        # Create two time series
        n_obs = 50
        a1 = np.random.randn(n_obs)
        a2 = np.random.randn(n_obs) * 2 + 1

        # Compute using non-matrix functions
        cov_nonmatrix = move_exp_nancov(a1, a2, alpha=alpha)
        corr_nonmatrix = move_exp_nancorr(a1, a2, alpha=alpha)

        # Compute using matrix functions
        # Exponential moving functions expect (obs, vars) format
        data_matrix = np.column_stack([a1, a2])
        cov_matrix_result = move_exp_nancovmatrix(data_matrix, alpha=alpha)
        corr_matrix_result = move_exp_nancorrmatrix(data_matrix, alpha=alpha)

        # Extract the off-diagonal element (covariance/correlation between a1 and a2)
        cov_from_matrix = cov_matrix_result[:, 0, 1]
        corr_from_matrix = corr_matrix_result[:, 0, 1]

        # They should match
        assert_allclose(cov_nonmatrix, cov_from_matrix, rtol=1e-10)
        assert_allclose(corr_nonmatrix, corr_from_matrix, rtol=1e-10)

        # Also check symmetry - (0,1) should equal (1,0)
        assert_allclose(
            cov_matrix_result[:, 0, 1], cov_matrix_result[:, 1, 0], rtol=1e-10
        )
        assert_allclose(
            corr_matrix_result[:, 0, 1], corr_matrix_result[:, 1, 0], rtol=1e-10
        )

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
        # Exponential moving functions expect (obs, vars) format
        data_matrix = np.column_stack([a1, a2, a3])
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

    def test_fixed_dimensional_conventions(self):
        """Test fixed dimensional conventions: (..., obs, vars) -> (..., obs, vars, vars)."""
        np.random.seed(42)

        # Basic test: (obs, vars) -> (obs, vars, vars)
        data = np.random.randn(20, 3)  # (obs, vars)
        corr_result = move_exp_nancorrmatrix(data, alpha=0.4)
        cov_result = move_exp_nancovmatrix(data, alpha=0.4)

        assert corr_result.shape == (20, 3, 3)
        assert cov_result.shape == (20, 3, 3)

        # Broadcasting test: (batch, obs, vars) -> (batch, obs, vars, vars)
        data_3d = np.random.randn(2, 20, 3)  # (batch, obs, vars)
        corr_3d = move_exp_nancorrmatrix(data_3d, alpha=0.4)

        assert corr_3d.shape == (2, 20, 3, 3)


class TestMatrixDtypePreservation:
    """Test dtype preservation across all matrix function types."""

    @pytest.mark.parametrize(
        "func",
        [nancorrmatrix, nancovmatrix, move_nancorrmatrix, move_nancovmatrix],
    )
    def test_dtype_preservation(self, func):
        """Test that dtypes are preserved."""
        # Set up appropriate data and window for rolling vs basic
        is_rolling = func.__name__.startswith("move_")

        # Test float32
        if is_rolling:
            # Moving functions expect (obs, vars)
            data32 = np.random.randn(10, 3).astype(np.float32)
            result32 = func(data32, window=5, min_count=3)
        else:
            # Basic functions expect (vars, obs)
            data32 = np.random.randn(3, 10).astype(np.float32)
            result32 = func(data32)
        assert result32.dtype == np.float32

        # Test float64
        if is_rolling:
            # Moving functions expect (obs, vars)
            data64 = np.random.randn(10, 3).astype(np.float64)
            result64 = func(data64, window=5, min_count=3)
        else:
            # Basic functions expect (vars, obs)
            data64 = np.random.randn(3, 10).astype(np.float64)
            result64 = func(data64)
        assert result64.dtype == np.float64
