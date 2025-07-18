import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numbagg import nancovmatrix


class TestNanCovMatrix:
    def test_simple_covariance_matrix(self):
        # Simple 2x2 covariance matrix
        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float64)
        result = nancovmatrix(data)

        # Calculate expected covariance
        expected = np.cov(data)
        assert_allclose(result, expected, rtol=1e-10)

    def test_zero_mean_variables(self):
        # Test with zero-mean variables
        data = np.array([[-1, 0, 1], [-2, 0, 2]], dtype=np.float64)
        result = nancovmatrix(data)

        expected = np.cov(data)
        assert_allclose(result, expected, rtol=1e-10)

    def test_with_nans(self):
        # Test with NaN values
        data = np.array(
            [[1, 2, np.nan, 4], [2, 4, 6, np.nan], [np.nan, 1, 2, 3]], dtype=np.float64
        )
        result = nancovmatrix(data)

        # Check diagonal is variance
        assert np.all(np.diag(result) >= 0)

        # Check symmetry
        assert_allclose(result, result.T)

    def test_all_nan_variable(self):
        # Test with a variable that is all NaN
        data = np.array(
            [[1, 2, 3, 4], [np.nan, np.nan, np.nan, np.nan], [2, 3, 4, 5]],
            dtype=np.float64,
        )
        result = nancovmatrix(data)

        # Second row and column should be NaN
        assert np.all(np.isnan(result[1, :]))
        assert np.all(np.isnan(result[:, 1]))

        # Other covariances should still work
        assert not np.isnan(result[0, 0])
        assert not np.isnan(result[2, 2])
        assert not np.isnan(result[0, 2])

    def test_single_observation(self):
        # Test with only one observation per variable
        data = np.array([[1], [2], [3]], dtype=np.float64)
        result = nancovmatrix(data)

        # Should be all NaN since variance is undefined with n=1
        expected = np.full((3, 3), np.nan)
        assert_array_equal(result, expected)

    def test_constant_variable(self):
        # Test with constant (zero variance) variables
        data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, 2, 3, 4]], dtype=np.float64)
        result = nancovmatrix(data)

        # Diagonal elements for constant variables should be 0
        assert result[0, 0] == 0.0
        assert result[1, 1] == 0.0
        assert result[2, 2] > 0  # Non-constant variable

        # Covariance between constants should be 0
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0

    def test_dtype_preservation(self):
        # Test float32
        data32 = np.random.randn(5, 20).astype(np.float32)
        result32 = nancovmatrix(data32)
        assert result32.dtype == np.float32

        # Test float64
        data64 = np.random.randn(5, 20).astype(np.float64)
        result64 = nancovmatrix(data64)
        assert result64.dtype == np.float64

    def test_axis_parameter(self):
        # Test with different axes
        data = np.random.randn(3, 4, 5)

        # Default should compute covariance along last axis
        result_default = nancovmatrix(data)
        assert result_default.shape == (3, 4, 4)

        # Test with axis=0
        result_0 = nancovmatrix(data, axis=0)
        assert result_0.shape == (4, 5, 5)

        # Test with axis=1
        result_1 = nancovmatrix(data, axis=1)
        assert result_1.shape == (3, 5, 5)

    def test_comparison_with_numpy(self):
        # Compare with numpy's cov for data without NaNs
        np.random.seed(42)
        data = np.random.randn(5, 100)

        result = nancovmatrix(data)
        expected = np.cov(data)

        assert_allclose(result, expected, rtol=1e-10)

    def test_relationship_to_correlation(self):
        # Test relationship between covariance and correlation
        from numbagg import nancorrmatrix

        np.random.seed(42)
        data = np.random.randn(4, 50)

        cov_matrix = nancovmatrix(data)
        corr_matrix = nancorrmatrix(data)

        # Correlation = Covariance / (std_i * std_j)
        stds = np.sqrt(np.diag(cov_matrix))
        expected_corr = cov_matrix / np.outer(stds, stds)

        assert_allclose(corr_matrix, expected_corr, rtol=1e-10)

    def test_large_matrix(self):
        # Test performance with larger matrix
        np.random.seed(42)
        data = np.random.randn(50, 1000)

        # Add some NaNs
        mask = np.random.rand(50, 1000) < 0.1
        data[mask] = np.nan

        result = nancovmatrix(data)

        # Check basic properties
        assert result.shape == (50, 50)
        assert_allclose(result, result.T, rtol=1e-10)

    def test_broadcasting_higher_dims(self):
        # Test that gufunc broadcasting works correctly for higher dimensional arrays
        np.random.seed(42)

        # 4D array: (2, 3, 4, 10) -> broadcast dims (2, 3) + core dims (4, 10)
        data_4d = np.random.randn(2, 3, 4, 10)
        result_4d = nancovmatrix(data_4d)
        assert result_4d.shape == (2, 3, 4, 4)

        # Check each broadcast element is a valid covariance matrix
        for i in range(2):
            for j in range(3):
                cov_matrix = result_4d[i, j]
                # Check symmetry
                assert_allclose(cov_matrix, cov_matrix.T, rtol=1e-10)
                # Diagonal should be variance (non-negative)
                assert np.all(np.diag(cov_matrix) >= 0)

        # 5D array: (2, 2, 2, 5, 20) -> broadcast dims (2, 2, 2) + core dims (5, 20)
        data_5d = np.random.randn(2, 2, 2, 5, 20)
        result_5d = nancovmatrix(data_5d)
        assert result_5d.shape == (2, 2, 2, 5, 5)

        # Verify a specific covariance matches manual calculation
        # Compare first broadcast element
        manual_cov = np.cov(data_5d[0, 0, 0])
        assert_allclose(result_5d[0, 0, 0], manual_cov, rtol=1e-10)
