import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numbagg import nancorrmatrix


class TestNanCorrMatrix:
    def test_simple_correlation_matrix(self):
        # Simple 2x2 correlation matrix
        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float64)
        result = nancorrmatrix(data)

        # Perfect correlation since second row is 2x first row
        expected = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert_allclose(result, expected, rtol=1e-10)

    def test_anticorrelation(self):
        # Test negative correlation
        data = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float64)
        result = nancorrmatrix(data)

        # Perfect negative correlation
        expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
        assert_allclose(result, expected, rtol=1e-10)

    def test_with_nans(self):
        # Test with NaN values
        data = np.array(
            [[1, 2, np.nan, 4], [2, 4, 6, np.nan], [np.nan, 1, 2, 3]], dtype=np.float64
        )
        result = nancorrmatrix(data)

        # Check diagonal is 1
        assert_allclose(np.diag(result), [1.0, 1.0, 1.0])

        # Check symmetry
        assert_allclose(result, result.T)

        # Correlation values should be between -1 and 1
        assert np.all((result >= -1) & (result <= 1) | np.isnan(result))

    def test_all_nan_variable(self):
        # Test with a variable that is all NaN
        data = np.array(
            [[1, 2, 3, 4], [np.nan, np.nan, np.nan, np.nan], [2, 3, 4, 5]],
            dtype=np.float64,
        )
        result = nancorrmatrix(data)

        # Second row and column should be NaN
        assert np.all(np.isnan(result[1, :]))
        assert np.all(np.isnan(result[:, 1]))

        # Other correlations should still work
        assert result[0, 0] == 1.0
        assert result[2, 2] == 1.0
        assert not np.isnan(result[0, 2])

    def test_single_observation(self):
        # Test with only one observation per variable
        data = np.array([[1], [2], [3]], dtype=np.float64)
        result = nancorrmatrix(data)

        # Should be NaN except diagonal
        expected = np.full((3, 3), np.nan)
        np.fill_diagonal(expected, 1.0)
        assert_array_equal(result, expected)

    def test_zero_variance(self):
        # Test with zero variance (constant) variables
        data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 3, 4]], dtype=np.float64)
        result = nancorrmatrix(data)

        # Correlation with constant variables should be NaN
        assert result[2, 2] == 1.0  # Variable with itself
        assert np.isnan(result[0, 1])  # Two constants
        assert np.isnan(result[0, 2])  # Constant with non-constant
        assert np.isnan(result[1, 2])  # Constant with non-constant

    def test_dtype_preservation(self):
        # Test float32
        data32 = np.random.randn(5, 20).astype(np.float32)
        result32 = nancorrmatrix(data32)
        assert result32.dtype == np.float32

        # Test float64
        data64 = np.random.randn(5, 20).astype(np.float64)
        result64 = nancorrmatrix(data64)
        assert result64.dtype == np.float64

    def test_axis_parameter(self):
        # Test with different axes
        data = np.random.randn(3, 4, 5)

        # Default should correlate along last axis
        result_default = nancorrmatrix(data)
        assert result_default.shape == (3, 4, 4)

        # Test with axis=0
        result_0 = nancorrmatrix(data, axis=0)
        assert result_0.shape == (4, 5, 5)

        # Test with axis=1
        result_1 = nancorrmatrix(data, axis=1)
        assert result_1.shape == (3, 5, 5)

    def test_comparison_with_numpy(self):
        # Compare with numpy's corrcoef for data without NaNs
        np.random.seed(42)
        data = np.random.randn(5, 100)

        result = nancorrmatrix(data)
        expected = np.corrcoef(data)

        assert_allclose(result, expected, rtol=1e-10)

    def test_large_matrix(self):
        # Test performance with larger matrix
        np.random.seed(42)
        data = np.random.randn(50, 1000)

        # Add some NaNs
        mask = np.random.rand(50, 1000) < 0.1
        data[mask] = np.nan

        result = nancorrmatrix(data)

        # Check basic properties
        assert result.shape == (50, 50)
        assert_allclose(np.diag(result), np.ones(50), rtol=1e-10)
        assert_allclose(result, result.T, rtol=1e-10)
        assert np.all((result >= -1) & (result <= 1) | np.isnan(result))
