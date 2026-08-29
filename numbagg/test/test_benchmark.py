import numpy as np
import pytest

from .. import (
    bfill,
    ffill,
    move_corr,
    move_corrmatrix,
    move_cov,
    move_covmatrix,
    move_exp_nancorrmatrix,
    move_exp_nancovmatrix,
    move_std,
    move_var,
    nancorrmatrix,
    nancovmatrix,
)


@pytest.fixture(
    params=[
        (1_000,),
        pytest.param((10_000_000,), marks=pytest.mark.nightly),
        pytest.param((100, 100_000), marks=pytest.mark.slow),
        pytest.param((100, 1000, 1000), marks=pytest.mark.nightly),
        pytest.param((10, 10, 10, 10, 1000), marks=pytest.mark.nightly),
        # Additional matrix-friendly shapes for benchmarking
        pytest.param((20, 1000), marks=pytest.mark.slow),  # 20×20 matrix, larger size
        pytest.param(
            (3, 5000), marks=pytest.mark.slow
        ),  # 3×3 matrix, many observations
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
        "move_corrmatrix",
        "move_covmatrix",
        "move_exp_nancorrmatrix",
        "move_exp_nancovmatrix",
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


@pytest.fixture(
    params=[
        "dense-float64",
        "dense-float32",
        "nan-heavy-float64",
        "offset-float64",
        "expired-outlier-float64",
        "repeated-outliers-float32",
        "level-change-float64",
        "slow-drift-float64",
        "near-constant-float64",
    ],
    scope="module",
)
def rolling_moment_workload(request):
    """Inputs that separate ordinary throughput from stability-path costs."""
    rng = np.random.default_rng(0)
    dtype = (
        np.float32
        if request.param in ("dense-float32", "repeated-outliers-float32")
        else np.float64
    )
    a = rng.standard_normal(100_000).astype(dtype)
    b = (0.4 * a + rng.standard_normal(len(a))).astype(dtype)

    if request.param == "nan-heavy-float64":
        a[::2] = np.nan
        b[1::4] = np.nan
    elif request.param == "offset-float64":
        a += 1e8
        b += 1e8
    elif request.param == "expired-outlier-float64":
        # Removing this first value forces the shifted recovery state to reanchor.
        a[0] = 1e12
        b[0] = -1e12
    elif request.param == "repeated-outliers-float32":
        a[::2_000] = 1e6
        b[::2_000] = -1e6
    elif request.param == "level-change-float64":
        a[len(a) // 2 :] += 1e8
        b[len(b) // 2 :] += 1e8
    elif request.param == "slow-drift-float64":
        drift = np.linspace(0.0, 1e8, len(a))
        a += drift
        b += drift
    elif request.param == "near-constant-float64":
        a = 1e8 + np.resize(np.arange(5) * 1e-7, len(a))
        b = 1e8 + np.resize(np.arange(5)[::-1] * 1e-7, len(b))

    return request.param, a, b


@pytest.mark.parametrize(
    "func", [move_var, move_std, move_cov, move_corr], ids=lambda x: x.__name__
)
@pytest.mark.benchmark(warmup=True, warmup_iterations=1)
def test_benchmark_rolling_moment_stability(benchmark, func, rolling_moment_workload):
    """Track ordinary and difficult rolling-moment costs independently."""
    workload, a, b = rolling_moment_workload
    args = (a,) if func in (move_var, move_std) else (a, b)
    benchmark.group = f"rolling-moments|{workload}"
    benchmark(func, *args, window=100, min_count=20)


# Because this clears the cache, it really slows down running the tests. So we only run
# it selectively.
@pytest.mark.parametrize(
    "func",
    [
        nancorrmatrix,
        nancovmatrix,
        move_corrmatrix,
        move_covmatrix,
        move_exp_nancorrmatrix,
        move_exp_nancovmatrix,
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 100000), marks=pytest.mark.slow),  # 5×5 matrix, target ~10ms
        pytest.param((50, 500), marks=pytest.mark.slow),  # 50×50 matrix, target >2ms
        pytest.param(
            (3000, 5, 1000), marks=pytest.mark.slow
        ),  # 3000 independent 5×5 matrices, 1000 obs each, target ~50ms
    ],
    indirect=True,
)
@pytest.mark.parametrize("library", ["numbagg", "pandas", "numpy"], indirect=True)
@pytest.mark.benchmark(warmup=True, warmup_iterations=1)
def test_benchmark_matrix(benchmark, func, func_callable, shape):
    """
    Benchmark matrix functions on matrix-friendly shapes.
    """
    benchmark.group = f"{func}|{shape}"
    benchmark(func_callable)


@pytest.mark.nightly
@pytest.mark.parametrize("shape", [(1, 20)], indirect=True)
def test_benchmark_compile(benchmark, clear_numba_cache, func_callable):
    benchmark.pedantic(func_callable, warmup_rounds=0, rounds=1, iterations=1)
