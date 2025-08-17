"""Benchmarks for axis ordering performance with C and F contiguous arrays.

Run this benchmark with:

    uv run pytest numbagg/test/test_axis_ordering_benchmark.py -m slow --benchmark-enable --benchmark-only --benchmark-group-by=group --benchmark-columns=mean -q
"""

import numpy as np
import pytest

from .. import nansum

pytestmark = pytest.mark.slow


@pytest.fixture(
    params=[
        # All same size (200x200x200) - only varying C/F and axis ordering
        # Single axis reductions
        ((200, 200, 200), 0, "C"),
        ((200, 200, 200), 0, "F"),
        ((200, 200, 200), 1, "C"),
        ((200, 200, 200), 1, "F"),
        ((200, 200, 200), 2, "C"),
        ((200, 200, 200), 2, "F"),
        # Multi-axis reductions - sorted order
        ((200, 200, 200), (0, 1), "C"),
        ((200, 200, 200), (0, 1), "F"),
        ((200, 200, 200), (0, 2), "C"),
        ((200, 200, 200), (0, 2), "F"),
        ((200, 200, 200), (1, 2), "C"),
        ((200, 200, 200), (1, 2), "F"),
        # Multi-axis reductions - unsorted order (to show effect of sorting)
        ((200, 200, 200), (1, 0), "C"),
        ((200, 200, 200), (1, 0), "F"),
        ((200, 200, 200), (2, 0), "C"),
        ((200, 200, 200), (2, 0), "F"),
        ((200, 200, 200), (2, 1), "C"),
        ((200, 200, 200), (2, 1), "F"),
    ]
)
def axis_config(request):
    """Fixture providing (shape, axis, order) configurations."""
    return request.param


@pytest.fixture
def axis_array(axis_config):
    """Create array with specified shape and memory ordering."""
    shape, axis, order = axis_config
    np.random.seed(42)
    arr = np.random.rand(*shape)
    # Add some NaNs
    mask = np.random.rand(*shape) > 0.9
    arr[mask] = np.nan
    # Ensure correct memory ordering
    if order == "F":
        arr = np.asfortranarray(arr)
    else:
        arr = np.ascontiguousarray(arr)
    return arr, axis


@pytest.mark.parametrize("func", [nansum])
@pytest.mark.benchmark(warmup=True, warmup_iterations=1)
def test_axis_ordering_performance(benchmark, func, axis_array):
    """Benchmark aggregation functions with different axis orderings."""
    arr, axis = axis_array

    # Get memory ordering info for the benchmark group name
    order = "F" if arr.flags["F_CONTIGUOUS"] else "C"

    # Create groups based on the actual reduction being performed
    # (what gets sorted internally will be the same)
    axis_str = str(axis).replace(" ", "")  # Remove spaces for cleaner output

    # Group by the number of axes being reduced (computational complexity)
    if isinstance(axis, int):
        benchmark.group = "reduce 1 axis (single)"
        benchmark.name = f"{order}-order_axis{axis}"
    else:
        # Group by number of axes being reduced
        num_axes = len(axis)
        benchmark.group = f"reduce {num_axes} axes (multi)"
        # Include the original axis order in the name
        benchmark.name = f"{order}-order_axis{axis_str}"

    benchmark(func, arr, axis=axis)


@pytest.mark.skip(reason="Benchmark fixture can only be used once per test")
@pytest.mark.parametrize(
    "shape,axes_list",
    [
        # Test different orderings of the same axes
        ((100, 100, 100), [(0, 1), (1, 0)]),
        ((100, 100, 100), [(0, 2), (2, 0)]),
        ((100, 100, 100), [(1, 2), (2, 1)]),
        ((100, 100, 100), [(0, 1, 2), (2, 1, 0), (1, 0, 2)]),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.benchmark(warmup=True, warmup_iterations=1)
def test_axis_order_consistency(benchmark, shape, axes_list, order):
    """Test that different orderings of the same axes have similar performance."""
    np.random.seed(42)
    arr = np.random.rand(*shape)

    # Add some NaNs
    mask = np.random.rand(*shape) > 0.9
    arr[mask] = np.nan

    # Ensure correct memory ordering
    if order == "F":
        arr = np.asfortranarray(arr)
    else:
        arr = np.ascontiguousarray(arr)

    # Run benchmark for each axis ordering
    results = []
    for axes in axes_list:
        result = benchmark.pedantic(
            nansum,
            args=(arr,),
            kwargs={"axis": axes},
            rounds=10,
            iterations=1,
        )
        results.append(result)

    benchmark.group = f"nansum|{shape}|axes={axes_list}|{order}"
