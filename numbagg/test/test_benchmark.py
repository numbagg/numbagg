import numpy as np
import pytest

from .. import bfill, ffill


@pytest.fixture(
    params=[
        (1_000,),
        pytest.param((10_000_000,), marks=pytest.mark.nightly),
        pytest.param((100, 100_000), marks=pytest.mark.slow),
        pytest.param((100, 1000, 1000), marks=pytest.mark.nightly),
        pytest.param((10, 10, 10, 10, 1000), marks=pytest.mark.nightly),
    ],
    scope="module",
)
def shape(request):
    return request.param


@pytest.mark.parametrize(
    "library", ["numbagg", "pandas", "bottleneck", "numpy"], indirect=True
)
@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=1,
)
def test_benchmark_main(benchmark, func, func_callable, shape):
    """
    Main func that benchmarks how fast functions are.
    """
    if func.__name__.startswith("group") and np.prod(shape) > 10_000_000:
        pytest.skip(
            "We're currently skipping the huge arrays with `group` functions, as they're quite slow"
        )
    if func.__name__ in ["allnan", "anynan"]:
        pytest.skip(
            "These functions need a different approach to benchmarking; so we're currently excluding them"
        )
    # Skip matrix functions for very large arrays as they create n×n outputs
    if func.__name__ in [
        "nancorrmatrix",
        "nancovmatrix",
        "move_nancorrmatrix",
        "move_nancovmatrix",
    ]:
        # For matrix functions, the output size is proportional to n^2 where n is the second-to-last dimension
        if len(shape) >= 2:
            n_vars = shape[-2]
            if n_vars > 100 or np.prod(shape) > 1_000_000:
                pytest.skip(
                    f"Skipping matrix function benchmark for large array (would create {n_vars}×{n_vars} matrix)"
                )
        # For moving matrix functions, also consider the time dimension
        if func.__name__.startswith("move_") and len(shape) >= 2:
            n_vars = shape[-2]
            n_time = shape[-1]
            # Output is n_time × n_vars × n_vars, so be conservative
            if n_vars > 50 or n_time * n_vars * n_vars > 10_000_000:
                pytest.skip(
                    f"Skipping moving matrix function benchmark for large array (would create {n_time}×{n_vars}×{n_vars} output)"
                )
    benchmark.group = f"{func}|{shape}"
    benchmark(func_callable)


@pytest.mark.parametrize("func", [ffill, bfill], indirect=True)
@pytest.mark.parametrize("shape", [(10, 10, 10, 10, 1000)], indirect=True)
def test_benchmark_f_bfill(benchmark, func_callable):
    """
    Was seeing some weird results for ffill and bfill — bfill was sometimes much faster
    than ffill. We can check this if we see this again.
    """
    benchmark.pedantic(
        func_callable,
        warmup_rounds=1,
        rounds=100,
        iterations=10,
    )


# Because this clears the cache, it really slows down running the tests. So we only run
# it selectively.
@pytest.mark.nightly
@pytest.mark.parametrize("shape", [(1, 20)], indirect=True)
def test_benchmark_compile(benchmark, clear_numba_cache, func_callable):
    benchmark.pedantic(func_callable, warmup_rounds=0, rounds=1, iterations=1)
