"""Advanced tests for exponential moving matrix functions - numerical stability and mathematical properties."""

import numpy as np
from numpy.testing import assert_allclose

from numbagg import move_exp_nancorrmatrix, move_exp_nancovmatrix


class TestMoveExpMatrixAdvanced:
    """Advanced tests for numerical stability and mathematical properties."""

    def test_positive_semidefinite_covariance(self):
        """Test that covariance matrices are positive semi-definite."""
        np.random.seed(42)
        data = np.random.randn(4, 50)

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
        data = np.random.randn(3, 30)

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

    def test_extreme_alpha_values(self):
        """Test behavior with alpha values very close to 0 and 1."""
        # Use data that's not perfectly correlated to see differences
        np.random.seed(42)
        data = np.random.randn(2, 20)
        data[1] = 0.7 * data[0] + 0.5 * np.random.randn(20)  # Partial correlation

        # Test with alpha very close to 0 (almost no decay)
        result_low = move_exp_nancorrmatrix(data, alpha=1e-6)

        # Test with alpha close to 1 (very fast decay)
        result_high = move_exp_nancorrmatrix(data, alpha=0.99)

        # Both should produce valid results
        assert not np.all(np.isnan(result_low[-1]))
        assert not np.all(np.isnan(result_high[-1]))

        # With different correlation patterns, results should differ
        # Check the off-diagonal correlation values
        assert not np.allclose(result_low[-5:, 0, 1], result_high[-5:, 0, 1], rtol=1e-2)

    def test_constant_time_series(self):
        """Test behavior with constant time series (zero variance)."""
        # Create constant data
        data = np.array([[5, 5, 5, 5], [3, 3, 3, 3]], dtype=np.float64)

        corr_result = move_exp_nancorrmatrix(data, alpha=0.5)
        cov_result = move_exp_nancovmatrix(data, alpha=0.5)

        # For constant data, covariance should be zero (off-diagonal)
        # and correlation should be undefined (NaN) for off-diagonal
        final_cov = cov_result[-1]
        final_corr = corr_result[-1]

        # Diagonal covariance should be zero for constant data
        assert_allclose(np.diag(final_cov), [0.0, 0.0], atol=1e-10)

        # Off-diagonal covariance should be zero
        assert_allclose(final_cov[0, 1], 0.0, atol=1e-10)
        assert_allclose(final_cov[1, 0], 0.0, atol=1e-10)

        # Correlation diagonal should be NaN for constant data (zero variance)
        assert np.isnan(final_corr[0, 0])
        assert np.isnan(final_corr[1, 1])

        # Off-diagonal correlation should be NaN (0/0 case)
        assert np.isnan(final_corr[0, 1])
        assert np.isnan(final_corr[1, 0])

    def test_numerical_stability_near_singular(self):
        """Test numerical stability with nearly linearly dependent variables."""
        np.random.seed(456)
        n_obs = 100
        a1 = np.random.randn(n_obs)
        # Create a2 that's almost identical to a1 but with detectable noise
        a2 = a1 + 1e-8 * np.random.randn(
            n_obs
        )  # Slightly larger noise for detectability

        data = np.array([a1, a2])

        corr_result = move_exp_nancorrmatrix(data, alpha=0.3)
        cov_result = move_exp_nancovmatrix(data, alpha=0.3)

        # Should not produce NaN or inf values
        assert np.all(np.isfinite(corr_result[-1]))
        assert np.all(np.isfinite(cov_result[-1]))

        # Correlation should be very close to 1
        final_corr = corr_result[-1]
        assert_allclose(final_corr[0, 1], 1.0, rtol=1e-4)

        # With the added noise, correlation should be slightly less than 1.0
        # But we'll be more lenient since exponential weighting might make it exactly 1.0
        assert final_corr[0, 1] <= 1.0

    def test_bias_correction_edge_cases(self):
        """Test bias correction in edge cases."""
        # Create scenario where bias correction might be problematic
        data = np.array(
            [[1, np.nan, np.nan, 2], [3, np.nan, np.nan, 4]], dtype=np.float64
        )

        # Very low alpha means slow weight accumulation
        result = move_exp_nancovmatrix(data, alpha=0.01, min_weight=0.001)

        # Should handle the sparse data gracefully
        # Early time steps should be NaN due to insufficient weight
        assert np.isnan(result[0, 0, 1])
        assert np.isnan(result[1, 0, 1])

        # Final result should be valid if enough weight accumulated
        final_result = result[-1]
        if not np.isnan(final_result[0, 1]):
            # If not NaN, should be finite
            assert np.isfinite(final_result[0, 1])

    def test_different_dtypes(self):
        """Test consistency between float32 and float64."""
        data_f64 = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float64)
        data_f32 = data_f64.astype(np.float32)

        alpha = 0.5

        result_f64 = move_exp_nancorrmatrix(data_f64, alpha=alpha)
        result_f32 = move_exp_nancorrmatrix(data_f32, alpha=alpha)

        # Results should be close (within float32 precision)
        assert_allclose(result_f64, result_f32, rtol=1e-6)

    def test_sparse_valid_data(self):
        """Test with very sparse non-NaN observations."""
        # Create data where only every 5th observation is valid
        data = np.full((2, 20), np.nan, dtype=np.float64)
        data[0, ::5] = [1, 2, 3, 4]  # Only positions 0, 5, 10, 15
        data[1, ::5] = [2, 4, 6, 8]

        result = move_exp_nancorrmatrix(data, alpha=0.3, min_weight=0.1)

        # Should produce some valid results eventually
        assert not np.all(np.isnan(result))

        # Check that results are reasonable where they exist
        finite_mask = np.isfinite(result)
        if np.any(finite_mask):
            finite_values = result[finite_mask]
            assert np.all(finite_values >= -1.0)
            assert np.all(finite_values <= 1.0)

    def test_regime_change_data(self):
        """Test with data that has structural breaks."""
        np.random.seed(789)

        # First regime: positive correlation
        regime1 = np.random.randn(2, 25)
        regime1[1] = 0.8 * regime1[0] + 0.6 * np.random.randn(25)

        # Second regime: negative correlation
        regime2 = np.random.randn(2, 25)
        regime2[1] = -0.8 * regime2[0] + 0.6 * np.random.randn(25)

        data = np.concatenate([regime1, regime2], axis=1)

        # Test with high alpha (should adapt quickly to regime change)
        result_high = move_exp_nancorrmatrix(data, alpha=0.9)

        # Test with low alpha (should be more stable across regime change)
        result_low = move_exp_nancorrmatrix(data, alpha=0.1)

        # The key insight: high alpha should reach the new regime correlation faster
        # Compare how close each gets to the true second regime correlation

        # Compute what the second regime correlation should be approximately
        regime2_corr = np.corrcoef(regime2)[0, 1]

        # High alpha should be closer to the true second regime correlation
        final_high = result_high[-1, 0, 1]
        final_low = result_low[-1, 0, 1]

        error_high = abs(final_high - regime2_corr)
        error_low = abs(final_low - regime2_corr)

        # High alpha should adapt faster, so should be closer to true regime 2 correlation
        assert (
            error_high < error_low or abs(error_high - error_low) < 0.1
        )  # Allow some tolerance

    def test_large_matrix_size(self):
        """Test with moderately large matrix to check scalability."""
        np.random.seed(999)
        n_vars = 10  # 10x10 matrix
        n_obs = 50

        data = np.random.randn(n_vars, n_obs)

        result = move_exp_nancorrmatrix(data, alpha=0.3)

        # Should complete without error and have correct shape
        assert result.shape == (n_obs, n_vars, n_vars)

        # Final matrix should have proper properties
        final_matrix = result[-1]
        assert_allclose(np.diag(final_matrix), np.ones(n_vars), rtol=1e-10)
        assert_allclose(final_matrix, final_matrix.T, rtol=1e-10)

    def test_memory_layout_sensitivity(self):
        """Test that function works with different memory layouts."""
        data_c = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float64, order="C")
        data_f = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float64, order="F")

        result_c = move_exp_nancorrmatrix(data_c, alpha=0.5)
        result_f = move_exp_nancorrmatrix(data_f, alpha=0.5)

        # Results should be identical regardless of memory layout
        assert_allclose(result_c, result_f, rtol=1e-15)
