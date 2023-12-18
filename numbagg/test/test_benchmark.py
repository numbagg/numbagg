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
    # Can increase this if we're comparing code changes
    max_time=2,
    warmup=True,
    warmup_iterations=1,
    # We care more about accuracy than deviations, so we set this high
    calibration_precision=10,
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


@pytest.fixture
def clear_numba_cache(func):
    func.gufunc.cache_clear()

    yield


# Because this clears the cache, it really slows down running the tests. So we only run
# it selectively.
@pytest.mark.nightly
@pytest.mark.parametrize("shape", [(1, 20)], indirect=True)
def test_benchmark_compile(benchmark, clear_numba_cache, func_callable):
    benchmark.pedantic(func_callable, warmup_rounds=0, rounds=1, iterations=1)
