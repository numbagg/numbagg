"""Consolidated tests for all matrix functions (corr/cov, static/rolling)."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from numbagg import move_nancorrmatrix, move_nancovmatrix, nancorrmatrix, nancovmatrix


class TestMatrixFunctions:
    """Test all matrix functions with parametrized tests."""

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

    @pytest.mark.parametrize(
        "move_func,static_func,window",
        [
            (move_nancorrmatrix, nancorrmatrix, 3),
            (move_nancovmatrix, nancovmatrix, 3),
        ],
    )
    def test_rolling_simple(self, move_func, static_func, window):
        """Test rolling functions with simple data."""
        data = np.array([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]], dtype=np.float64)
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
        data = np.random.randn(n_vars, n_obs)

        # NumBagg result
        numbagg_result = move_func(data, window=window, min_count=window)

        # Pandas result - need to transpose for pandas (wants observations as rows)
        df = pd.DataFrame(data.T)
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
        "func",
        [nancorrmatrix, nancovmatrix, move_nancorrmatrix, move_nancovmatrix],
    )
    def test_dtype_preservation(self, func):
        """Test that dtypes are preserved."""
        # Set up appropriate data and window for rolling vs static
        is_rolling = func.__name__.startswith("move_")

        # Test float32
        data32 = np.random.randn(3, 10).astype(np.float32)
        if is_rolling:
            result32 = func(data32, window=5, min_count=3)
        else:
            result32 = func(data32)
        assert result32.dtype == np.float32

        # Test float64
        data64 = np.random.randn(3, 10).astype(np.float64)
        if is_rolling:
            result64 = func(data64, window=5, min_count=3)
        else:
            result64 = func(data64)
        assert result64.dtype == np.float64

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

    @pytest.mark.parametrize(
        "move_func,expected_diag",
        [
            (move_nancorrmatrix, 1.0),
            (move_nancovmatrix, None),
        ],
    )
    def test_rolling_zero_variance_windows(self, move_func, expected_diag):
        """Test rolling windows with zero variance."""
        data = np.array([[1, 1, 1, 2, 3, 4], [2, 2, 2, 3, 4, 5]], dtype=np.float64)
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

    @pytest.mark.parametrize("move_func", [move_nancorrmatrix, move_nancovmatrix])
    def test_rolling_broadcasting_higher_dims(self, move_func):
        """Test that rolling functions broadcast correctly over higher dimensions."""
        np.random.seed(42)
        window = 5

        # 3D array: (2, 4, 20) -> output (2, 20, 4, 4)
        data_3d = np.random.randn(2, 4, 20)
        result_3d = move_func(data_3d, window=window)
        assert result_3d.shape == (2, 20, 4, 4)

        # 4D array: (2, 3, 4, 20) -> output (2, 3, 20, 4, 4)
        data_4d = np.random.randn(2, 3, 4, 20)
        result_4d = move_func(data_4d, window=window)
        assert result_4d.shape == (2, 3, 20, 4, 4)

        # Verify correctness - each slice should match individual computation
        for i in range(2):
            single_result = move_func(data_3d[i], window=window)
            assert_allclose(result_3d[i], single_result, rtol=1e-10)
