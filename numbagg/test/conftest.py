from __future__ import annotations

import logging
from functools import partial
from typing import Callable

import bottleneck as bn
import pandas as pd
import pytest
from numpy import nanquantile

from .. import (
    bfill,
    ffill,
    move_corr,
    move_cov,
    # move_count,
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

# TODO: add these as functions to the dict
# nanargmax,
# nanargmin,
# nanmax,
# nanmean,
# nanmin,
# nanstd,


def pandas_ewm_setup(func, a, alpha=0.5):
    df = pd.DataFrame(a).T.ewm(alpha=alpha)
    return lambda: func(df)


def two_array_setup(a):
    a1, a2 = a, a**2 + 1
    return a1, a2


def pandas_ewm_2_array_setup(func, a, alpha=0.5):
    a1, a2 = two_array_setup(a)
    df1, df2 = pd.DataFrame(a1).T.ewm(alpha=alpha), pd.DataFrame(a2).T
    return lambda: func(df1, df2)


def numbagg_ewm_2_array_setup(func, a, alpha=0.5):
    a1, a2 = two_array_setup(a)
    return lambda: func(a1, a2, alpha=alpha)


def pandas_ewm_nancount_setup(a, alpha=0.5):
    df = pd.DataFrame(a).T
    return lambda: df.notnull().ewm(alpha=alpha).sum().T


def pandas_move_setup(func, a, window=20, min_count=None):
    df = pd.DataFrame(a).T.rolling(window=window, min_periods=min_count)
    return lambda: func(df)


def pandas_move_2_array_setup(func, a, window=20, min_count=None):
    a1, a2 = two_array_setup(a)
    df1 = pd.DataFrame(a1).T.rolling(window=window, min_periods=min_count)
    df2 = pd.DataFrame(a2).T
    return lambda: func(df1, df2)


def numbagg_two_array_setup(func, a, *args, **kwargs):
    a1, a2 = two_array_setup(a)
    return lambda: func(a1, a2, *args, **kwargs)


# Parameterization of tests and benchmarks
#
# - Each function has a dict, which contains a mapping of library:lambda, for each
#   library we want to test.
# - The lambda takes an takes an input array and returns a callable than can be called
#   with `func()`. It should do all "setup" beforehand, so when we benchmark the
#   functions, we're not benchmarking allocating dataframes etc.
# - The lambda can also can have optional kwargs if we want to be able to test other
#   parameters.

COMPARISONS: dict[Callable, dict[str, Callable]] = {
    move_exp_nanvar: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_setup(lambda df: df.var().T, a, alpha),
        numbagg=lambda a, alpha=0.5: partial(move_exp_nanvar, a, alpha=alpha),
    ),
    move_exp_nanmean: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_setup(lambda df: df.mean().T, a, alpha),
        numbagg=lambda a, alpha=0.5: partial(move_exp_nanmean, a, alpha=alpha),
    ),
    move_exp_nancount: dict(
        pandas=pandas_ewm_nancount_setup,
        numbagg=lambda a, alpha=0.5: partial(move_exp_nancount, a, alpha=alpha),
    ),
    move_exp_nanstd: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_setup(lambda df: df.std().T, a, alpha),
        numbagg=lambda a, alpha=0.5: partial(move_exp_nanstd, a, alpha=alpha),
    ),
    move_exp_nansum: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_setup(lambda df: df.sum().T, a, alpha),
        numbagg=lambda a, alpha=0.5: partial(move_exp_nansum, a, alpha=alpha),
    ),
    move_exp_nancorr: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_2_array_setup(
            lambda df1, df2: df1.corr(df2).T, a, alpha=alpha
        ),
        numbagg=lambda a, alpha=0.5: numbagg_two_array_setup(
            move_exp_nancorr, a, alpha=alpha
        ),
    ),
    move_exp_nancov: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_2_array_setup(
            lambda df1, df2: df1.cov(df2).T, a, alpha=alpha
        ),
        numbagg=lambda a, alpha=0.5: numbagg_two_array_setup(
            move_exp_nancov, a, alpha=alpha
        ),
    ),
    move_mean: dict(
        pandas=lambda a, window=20, min_count=None: pandas_move_setup(
            lambda df: df.mean().T, a, window, min_count
        ),
        numbagg=lambda a, window=20, min_count=None: partial(
            move_mean, a, window=window, min_count=min_count
        ),
        bottleneck=lambda a, window=20, min_count=None: partial(
            bn.move_mean, a, window=window, min_count=min_count
        ),
    ),
    move_sum: dict(
        pandas=lambda a, window=20, min_count=None: pandas_move_setup(
            lambda df: df.sum().T, a, window, min_count
        ),
        numbagg=lambda a, window=20, min_count=None: partial(
            move_sum, a, window=window, min_count=min_count
        ),
        bottleneck=lambda a, window=20, min_count=None: partial(
            bn.move_sum, a, window=window, min_count=min_count
        ),
    ),
    move_std: dict(
        pandas=lambda a, window=20, min_count=None: pandas_move_setup(
            lambda df: df.std().T, a, window, min_count
        ),
        numbagg=lambda a, window=20, min_count=None: partial(
            move_std, a, window=window, min_count=min_count
        ),
        bottleneck=lambda a, window=20, min_count=None: partial(
            bn.move_std, a, window=window, min_count=min_count, ddof=1
        ),
    ),
    move_var: dict(
        pandas=lambda a, window=20, min_count=None: pandas_move_setup(
            lambda df: df.var().T, a, window, min_count
        ),
        numbagg=lambda a, window=20, min_count=None: partial(
            move_var, a, window=window, min_count=min_count
        ),
        bottleneck=lambda a, window=20, min_count=None: partial(
            bn.move_var, a, window=window, min_count=min_count, ddof=1
        ),
    ),
    move_corr: dict(
        pandas=lambda a, window=20, min_count=None: pandas_move_2_array_setup(
            lambda df1, df2: df1.corr(df2).T, a, window, min_count
        ),
        numbagg=lambda a, window=20, min_count=None: numbagg_two_array_setup(
            move_corr, a, window=window, min_count=min_count
        ),
    ),
    move_cov: dict(
        pandas=lambda a, window=20, min_count=None: pandas_move_2_array_setup(
            lambda df1, df2: df1.cov(df2).T, a, window, min_count
        ),
        numbagg=lambda a, window=20, min_count=None: numbagg_two_array_setup(
            move_cov, a, window=window, min_count=min_count
        ),
    ),
    ffill: dict(
        pandas=lambda a, limit=None: lambda: pd.DataFrame(a).T.ffill(limit=limit).T,
        numbagg=lambda a, limit=None: partial(ffill, a, limit=limit),
        bottleneck=lambda a, limit=None: partial(bn.push, a, limit),
    ),
    bfill: dict(
        pandas=lambda a, limit=None: lambda: pd.DataFrame(a).T.bfill(limit=limit).T,
        numbagg=lambda a, limit=None: partial(bfill, a, limit=limit),
        bottleneck=lambda a, limit=None: lambda: bn.push(a[..., ::-1], limit)[
            ..., ::-1
        ],
    ),
    nanquantile: dict(
        pandas=lambda a, quantiles=[0.25, 0.75]: lambda: pd.DataFrame(a)
        .T.quantile(quantiles)
        .T,
        numbagg=lambda a, quantiles=[0.25, 0.75]: partial(nanquantile, a, quantiles),
    )
    # move_count: dict(
    #     pandas=dict(
    #         setup=pandas_move_setup,
    #         run=lambda a: a.count().T,
    #     ),
    #     numbagg=dict(
    #         setup=lambda a: a,
    #         run=numbagg_move_run
    #     ),
    # ),
}


@pytest.fixture(params=["numbagg", "pandas", "bottleneck"], scope="module")
def library(request):
    return request.param


@pytest.fixture(scope="module")
def func_callable(library, func, array):
    """
    The function post-setup; just needs to be called with `func_callable()`
    """
    if len(array.shape) > 2 and library == "pandas":
        pytest.skip("pandas doesn't support array with more than 2 dimensions")
    try:
        return COMPARISONS[func][library](array)
    except KeyError:
        if library == "bottleneck":
            pytest.skip(f"Bottleneck doesn't support {func}")


@pytest.fixture(scope="module")
def obj(array, setup):
    return setup(array)


@pytest.fixture(autouse=True)
def numba_logger():
    # This is exteremly noisy, so we turn it off. We can make this a setting if it would
    # be occasionally useful.
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
