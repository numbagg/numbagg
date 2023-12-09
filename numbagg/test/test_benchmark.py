import importlib

import numpy as np
import pytest

from .. import bfill, ffill


@pytest.fixture(
    params=[
        (1, 1_000),
        (10, 1_000_000),
        (1, 10_000_000),
        (100, 1000, 1000),
        (10, 10, 10, 10, 1000),
    ],
    scope="module",
)
def shape(request):
    return request.param


@pytest.mark.parametrize("library", ["numbagg", "pandas", "bottleneck"], indirect=True)
def test_benchmark_all(benchmark, func, func_callable, shape):
    benchmark.group = f"{func}|{shape}"
    benchmark.pedantic(
        func_callable,
        warmup_rounds=1,
        rounds=3,
        iterations=int(max(10_000_000 // np.prod(shape), 1)),
    )


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
    import numbagg

    func.gufunc.cache_clear()
    importlib.reload(numbagg)

    yield


@pytest.mark.parametrize("shape", [(1, 20)], indirect=True)
def test_benchmark_compile(benchmark, clear_numba_cache, func_callable):
    def foo():
        try:
            func_callable()
        except Exception:
            pass

    benchmark.pedantic(foo, warmup_rounds=0, rounds=1, iterations=1)
