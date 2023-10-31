import numpy as np
import pytest

from .. import (
    bfill,
    ffill,
    move_corr,
    move_cov,
    move_exp_nancorr,
    move_exp_nancount,
    move_exp_nancov,
    move_exp_nanmean,
    move_exp_nanstd,
    move_exp_nansum,
    move_exp_nanvar,
    move_mean,
    move_std,
    move_sum,
    move_var,
)


@pytest.fixture(
    params=[
        bfill,
        ffill,
        move_corr,
        move_cov,
        move_exp_nancorr,
        move_exp_nancount,
        move_exp_nancov,
        move_exp_nanmean,
        move_exp_nanstd,
        move_exp_nansum,
        move_exp_nanvar,
        move_mean,
        move_std,
        move_sum,
        move_var,
    ],
    scope="module",
)
def func(request):
    return request.param


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


# One disadvantage of having this in a fixture is that pytest keeps it around for the
# whole test session. So we at least use module scoping so it will keep a single one for
# all tests.
@pytest.fixture(scope="module")
def array(shape):
    array = np.random.RandomState(0).rand(*shape)
    return np.where(array > 0.1, array, np.nan)


def test_benchmark(benchmark, func, func_callable, shape):
    benchmark.group = f"{func}|{shape}"
    benchmark.pedantic(
        func_callable,
        warmup_rounds=1,
        rounds=3,
        iterations=int(max(10_000_000 // np.prod(shape), 1)),
    )


@pytest.mark.parametrize("func", [ffill, bfill], indirect=True)
@pytest.mark.parametrize("shape", [(10, 10, 10, 10, 1000)], indirect=True)
@pytest.mark.parametrize("library", ["numbagg"], indirect=True)
def test_benchmark_f_bfill(benchmark, func, func_callable, shape):
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
