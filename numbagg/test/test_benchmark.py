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
)
def func(request):
    return request.param


@pytest.fixture(params=[1_000, 100_000, 10_000_000])
def size(request):
    return request.param


@pytest.fixture()
def array(size):
    array = np.random.RandomState(0).rand(3, size)
    return np.where(array > 0.1, array, np.nan)


def test_benchmark(benchmark, func_callable, size):
    benchmark.group = f"{func}|{size}"
    benchmark.pedantic(
        func_callable, warmup_rounds=1, rounds=5, iterations=10_000_000 // size
    )
