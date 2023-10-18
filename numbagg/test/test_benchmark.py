from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from numbagg import move_exp_nanmean, move_exp_nansum, move_exp_nanvar


@pytest.fixture(params=["numbagg", "pandas"])
def library(request):
    return request.param


# Tuple of (setup, run)
PANDAS_FUNCTIONS = {
    move_exp_nanmean: (lambda x: x.ewm(alpha=0.5), lambda x: x.mean()),
    move_exp_nansum: (lambda x: x.ewm(alpha=0.5), lambda x: x.sum()),
    move_exp_nanvar: (lambda x: x.ewm(alpha=0.5), lambda x: x.var()),
}


@pytest.fixture(
    params=[
        move_exp_nanmean,
        move_exp_nansum,
        move_exp_nanvar,
    ],
)
def numbagg_func(request):
    return request.param


@pytest.fixture()
def funcs(library, numbagg_func) -> tuple[Callable, Callable]:
    """
    Returns a setup function and a running function
    """
    if library == "numbagg":
        return lambda x: x, partial(numbagg_func, alpha=0.5)
    elif library == "pandas":
        return PANDAS_FUNCTIONS[numbagg_func]
    else:
        raise ValueError(f"Unknown library {library}")


@pytest.fixture(params=[1_000, 100_000, 10_000_000])
def size(request):
    return request.param


@pytest.fixture()
def array(size):
    array = np.random.RandomState(0).rand(3, size)
    return np.where(array > 0.1, array, np.nan)


@pytest.fixture()
def obj(array, library, funcs):
    if library == "numbagg":
        return array
    elif library == "pandas":
        return pd.DataFrame(array.T).pipe(funcs[0])
    else:
        raise ValueError(f"Unknown library {library}")


# @pytest.mark.parametrize("size", [1_000], indirect=True)
def test_benchmark_small(benchmark, funcs, numbagg_func, obj, size):
    benchmark.group = f"{numbagg_func}|{size}"
    benchmark.pedantic(
        funcs[1], args=(obj,), warmup_rounds=1, rounds=3, iterations=10_000_000 // size
    )


# @pytest.mark.parametrize("size", [10_000_000], indirect=True)
# def test_benchmark_big(benchmark, func, obj, library):
#     benchmark.group = f"{func} big"
#     benchmark.pedantic(func, args=(obj,), warmup_rounds=1, rounds=3, iterations=3)


# def setup(self, func, n):
#     array = np.random.RandomState(0).rand(3, n)
#     self.array = np.where(array > 0.1, array, np.nan)
#     self.df_ewm = pd.DataFrame(self.array.T).ewm(alpha=0.5)
#     # One run for JIT (asv states that it does this before runs, but this still
#     # seems to make a difference)
#     func[0](self.array, 0.5)
#     func[1](self.df_ewm)


# def time_numbagg(self, func, n):
#     func[0](self.array, 0.5)


# def time_pandas(self, func, n):
#     func[1](self.df_ewm)
