import numpy as np
import pytest

from .. import (
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


def test_benchmark_small(benchmark, run, func, obj, size):
    benchmark.group = f"{func}|{size}"
    benchmark.pedantic(
        run, args=(obj,), warmup_rounds=1, rounds=3, iterations=10_000_000 // size
    )


# def setup(self, func, n):
#     array = np.random.RandomState(0).rand(3, n)
#     self.array = np.where(array > 0.1, array, np.nan)
#     self.df_ewm = pd.DataFrame(self.array.T).ewm(alpha=0.5)
#     # One run for JIT (asv states that it does this before runs, but this still
#     # seems to make a difference)
#     func[0](self.array, 0.5)
#     func[1](self.df_ewm)
