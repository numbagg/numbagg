"""Benchmark showing axis ordering optimization for same memory access patterns.

This benchmark demonstrates that arrays with the same fundamental memory access
pattern have similar performance after optimization, regardless of:
1. Whether they're C or F ordered
2. How the axes are specified (0,1) vs (1,0)

Key insight: C(1500,2500) and F(2500,1500) have the same memory access pattern!
- C(1500,2500): 1500 blocks of 2500 contiguous elements
- F(2500,1500): 2500 blocks of 1500 contiguous elements (same pattern!)

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

    # Create base data - use larger arrays to see memory pattern differences
    # 1500x2500 = 3.75M elements, shows differences while still being fast
    data_1500x2500 = np.random.rand(1500, 2500)
    mask = np.random.rand(1500, 2500) > 0.9
    data_1500x2500[mask] = np.nan

    # Group 1: Same memory access pattern
    # C(1500,2500) has same pattern as F(2500,1500)
    c_1500x2500 = np.ascontiguousarray(data_1500x2500)  # C-ordered (1500, 2500)
    f_2500x1500 = np.asfortranarray(
        data_1500x2500.T
    )  # F-ordered (2500, 1500) - transposed!

    # Group 2: Same memory access pattern
    # C(2500,1500) has same pattern as F(1500,2500)
    c_2500x1500 = np.ascontiguousarray(data_1500x2500.T)  # C-ordered (2500, 1500)
    f_1500x2500 = np.asfortranarray(data_1500x2500)  # F-ordered (1500, 2500)

    return {
        "C_1500x2500": c_1500x2500,
        "F_2500x1500": f_2500x1500,
        "C_2500x1500": c_2500x1500,
        "F_1500x2500": f_1500x2500,
    }


@pytest.fixture(
    params=[
        # Group 1: C(1500x2500) and F(2500x1500) - same memory access pattern
        ("C_1500x2500", (0, 1), "pattern1_C1500x2500_F2500x1500"),
        ("C_1500x2500", (1, 0), "pattern1_C1500x2500_F2500x1500"),
        ("F_2500x1500", (0, 1), "pattern1_C1500x2500_F2500x1500"),
        ("F_2500x1500", (1, 0), "pattern1_C1500x2500_F2500x1500"),
        # Group 2: C(2500x1500) and F(1500x2500) - same memory access pattern
        ("C_2500x1500", (0, 1), "pattern2_C2500x1500_F1500x2500"),
        ("C_2500x1500", (1, 0), "pattern2_C2500x1500_F1500x2500"),
        ("F_1500x2500", (0, 1), "pattern2_C2500x1500_F1500x2500"),
        ("F_1500x2500", (1, 0), "pattern2_C2500x1500_F1500x2500"),
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

    # Create test data (smaller for quick test)
    data = np.random.rand(150, 250)
    data[np.random.rand(150, 250) > 0.9] = np.nan

    # Create arrays with different layouts
    c_150x250 = np.ascontiguousarray(data)
    f_250x150 = np.asfortranarray(data.T)
    c_250x150 = np.ascontiguousarray(data.T)
    f_150x250 = np.asfortranarray(data)

    # All should produce the same result
    result1 = nansum(c_150x250, axis=(0, 1))
    result2 = nansum(c_150x250, axis=(1, 0))
    result3 = nansum(f_250x150, axis=(0, 1))
    result4 = nansum(f_250x150, axis=(1, 0))
    result5 = nansum(c_250x150, axis=(0, 1))
    result6 = nansum(c_250x150, axis=(1, 0))
    result7 = nansum(f_150x250, axis=(0, 1))
    result8 = nansum(f_150x250, axis=(1, 0))

    # Check all equal
    np.testing.assert_allclose(result1, result2)
    np.testing.assert_allclose(result1, result3)
    np.testing.assert_allclose(result1, result4)
    np.testing.assert_allclose(result1, result5)
    np.testing.assert_allclose(result1, result6)
    np.testing.assert_allclose(result1, result7)
    np.testing.assert_allclose(result1, result8)

    print(f"✓ All configurations produce same result: {result1:.6f}")
