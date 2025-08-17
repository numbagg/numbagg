"""Benchmark showing axis ordering optimization for same memory access patterns.

This benchmark demonstrates that arrays with the same fundamental memory access
pattern have similar performance after optimization, regardless of:
1. Whether they're C or F ordered
2. How the axes are specified (0,1) vs (1,0)

Key insight: C(300,500) and F(500,300) have the same memory access pattern!
- C(300,500): 300 blocks of 500 contiguous elements
- F(500,300): 500 blocks of 300 contiguous elements (same pattern!)

Run this benchmark:
    uv run pytest numbagg/test/test_axis_ordering_benchmark.py -m slow --benchmark-enable --benchmark-only --benchmark-group-by=group --benchmark-columns=mean -q
"""

import numpy as np
import pytest

from .. import nansum

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def arrays_2d():
    """Create 2D arrays with equivalent memory access patterns."""
    np.random.seed(42)

    # Create base data
    data_300x500 = np.random.rand(300, 500)
    mask = np.random.rand(300, 500) > 0.9
    data_300x500[mask] = np.nan

    # Group 1: Same memory access pattern
    # C(300,500) has same pattern as F(500,300)
    c_300x500 = np.ascontiguousarray(data_300x500)  # C-ordered (300, 500)
    f_500x300 = np.asfortranarray(data_300x500.T)  # F-ordered (500, 300) - transposed!

    # Group 2: Same memory access pattern
    # C(500,300) has same pattern as F(300,500)
    c_500x300 = np.ascontiguousarray(data_300x500.T)  # C-ordered (500, 300)
    f_300x500 = np.asfortranarray(data_300x500)  # F-ordered (300, 500)

    return {
        "C_300x500": c_300x500,
        "F_500x300": f_500x300,
        "C_500x300": c_500x300,
        "F_300x500": f_300x500,
    }


@pytest.fixture(
    params=[
        # Group 1: C(300x500) and F(500x300) - same memory access pattern
        ("C_300x500", (0, 1), "pattern1_C300x500_F500x300"),
        ("C_300x500", (1, 0), "pattern1_C300x500_F500x300"),
        ("F_500x300", (0, 1), "pattern1_C300x500_F500x300"),
        ("F_500x300", (1, 0), "pattern1_C300x500_F500x300"),
        # Group 2: C(500x300) and F(300x500) - same memory access pattern
        ("C_500x300", (0, 1), "pattern2_C500x300_F300x500"),
        ("C_500x300", (1, 0), "pattern2_C500x300_F300x500"),
        ("F_300x500", (0, 1), "pattern2_C500x300_F300x500"),
        ("F_300x500", (1, 0), "pattern2_C500x300_F300x500"),
    ]
)
def config_2d(request):
    """Configuration: (array_key, axes, group_name)."""
    return request.param


@pytest.mark.parametrize("func", [nansum])
@pytest.mark.benchmark(warmup=True, warmup_iterations=1)
def test_same_memory_pattern_performance(benchmark, func, arrays_2d, config_2d):
    """Test that same memory access patterns have similar performance."""
    array_key, axes, group = config_2d
    arr = arrays_2d[array_key]

    # Group by memory access pattern
    benchmark.group = group

    # Name shows specific configuration
    benchmark.name = f"{array_key}_axes{axes}"

    # Run benchmark
    result = benchmark(func, arr, axis=axes)

    # Verify all produce scalar (reducing all dimensions)
    assert np.isscalar(result) or result.shape == ()


def test_verify_same_results():
    """Verify that all configurations produce the same numerical result."""
    np.random.seed(42)

    # Create test data
    data = np.random.rand(30, 50)
    data[np.random.rand(30, 50) > 0.9] = np.nan

    # Create arrays with different layouts
    c_30x50 = np.ascontiguousarray(data)
    f_50x30 = np.asfortranarray(data.T)
    c_50x30 = np.ascontiguousarray(data.T)
    f_30x50 = np.asfortranarray(data)

    # All should produce the same result
    result1 = nansum(c_30x50, axis=(0, 1))
    result2 = nansum(c_30x50, axis=(1, 0))
    result3 = nansum(f_50x30, axis=(0, 1))
    result4 = nansum(f_50x30, axis=(1, 0))
    result5 = nansum(c_50x30, axis=(0, 1))
    result6 = nansum(c_50x30, axis=(1, 0))
    result7 = nansum(f_30x50, axis=(0, 1))
    result8 = nansum(f_30x50, axis=(1, 0))

    # Check all equal
    np.testing.assert_allclose(result1, result2)
    np.testing.assert_allclose(result1, result3)
    np.testing.assert_allclose(result1, result4)
    np.testing.assert_allclose(result1, result5)
    np.testing.assert_allclose(result1, result6)
    np.testing.assert_allclose(result1, result7)
    np.testing.assert_allclose(result1, result8)

    print(f"✓ All configurations produce same result: {result1:.6f}")
