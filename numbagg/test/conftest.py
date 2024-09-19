from __future__ import annotations

import functools
import logging
import warnings
from functools import cache, partial
from typing import Callable

import bottleneck as bn
import numpy as np
import pandas as pd
import pytest

from .. import (
    allnan,
    anynan,
    bfill,
    ffill,
    group_nanall,
    group_nanany,
    group_nanargmax,
    group_nanargmin,
    group_nancount,
    group_nanfirst,
    group_nanlast,
    group_nanmax,
    group_nanmean,
    group_nanmin,
    group_nanprod,
    group_nanstd,
    group_nansum,
    group_nansum_of_squares,
    group_nanvar,
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
    nanargmax,
    nanargmin,
    nancount,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
)


def pandas_ewm_setup(func, a, alpha=0.5):
    df = _df_of_array(a).ewm(alpha=alpha)
    return lambda: func(df)


def two_array_setup(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a1, a2 = a, a**2 + 1
    return a1, a2


def pandas_ewm_2_array_setup(func, a, alpha=0.5):
    a1, a2 = two_array_setup(a)
    df1, df2 = _df_of_array(a1).ewm(alpha=alpha), _df_of_array(a2)
    return lambda: func(df1, df2)


def numbagg_ewm_2_array_setup(func, a, alpha=0.5):
    a1, a2 = two_array_setup(a)
    return lambda: func(a1, a2, alpha=alpha)


def pandas_ewm_nancount_setup(a, alpha=0.5):
    df = _df_of_array(a)
    return lambda: df.notnull().ewm(alpha=alpha).sum().T


def pandas_move_setup(func, a, window=20, min_count=None):
    df = _df_of_array(a).rolling(window=window, min_periods=min_count)
    return partial(func, df)


def pandas_move_2_array_setup(func, a, window=20, min_count=None):
    a1, a2 = two_array_setup(a)
    df1 = _df_of_array(a1).rolling(window=window, min_periods=min_count)
    df2 = _df_of_array(a2)
    return lambda: func(df1, df2)


def numbagg_two_array_setup(func, a, **kwargs):
    a1, a2 = two_array_setup(a)
    return partial(func, a1, a2, **kwargs)


@cache
def generate_labels(size):
    # TODO: could make this a few different settings:
    # - high cardinality
    # - low cardinality
    # - skewed
    # - missing values
    np.random.seed(0)
    return np.random.randint(0, 12, size=size)


def numbagg_group_setup(func, a, **kwargs):
    # For benchmarking, it's fair to factorize the labels — otherwise pandas has to do
    # the work but numbagg doesn't.
    labels = generate_labels(a.shape[-1])

    @functools.wraps(func)
    def with_factorization(*args, axis=-1, **kwargs):
        # TODO: is it possible to avoid `sort=True`? I don't think so...
        # `result.take(uniques, axis=axis)` indexes the wrong way — we want to use the
        # position of `uniques` in `codes` to index `result`...`
        codes, uniques = pd.factorize(labels, sort=True)
        result = func(*args, **kwargs, labels=codes, num_labels=len(uniques), axis=axis)
        return result

    return partial(with_factorization, a, **kwargs)


def pandas_group_setup(func_name, a, **kwargs):
    labels = generate_labels(a.shape[-1])
    df = _df_of_array(a)
    return lambda: (
        df.groupby(labels).pipe(lambda x: getattr(x, func_name)(**kwargs)).T
    )


def pandas_nan_sum_of_squares_setup(a):
    labels = generate_labels(a.shape[-1])
    df = _df_of_array(a)
    return lambda: df.pipe(lambda x: x**2).groupby(labels).sum().T


def _df_of_array(a):
    if len(a.shape) == 1:
        return pd.DataFrame(a)
    elif len(a.shape) == 2:
        return pd.DataFrame(a).T


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
    nansum: dict(
        numbagg=lambda a, axis=-1: partial(nansum, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).sum().T,
        bottleneck=lambda a, axis=-1: partial(bn.nansum, a, axis=axis),
        numpy=lambda a, axis=-1: partial(np.nansum, a, axis=axis),
    ),
    nanargmax: dict(
        numbagg=lambda a, axis=-1: partial(nanargmax, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).idxmax().T,
        bottleneck=lambda a, axis=-1: partial(bn.nanargmax, a, axis=axis),
    ),
    nanargmin: dict(
        numbagg=lambda a, axis=-1: partial(nanargmin, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).idxmin().T,
        bottleneck=lambda a, axis=-1: partial(bn.nanargmin, a, axis=axis),
    ),
    nancount: dict(
        numbagg=lambda a, axis=-1: partial(nancount, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).count().T,
        numpy=lambda a, axis=-1: lambda: a.shape[axis]
        - np.count_nonzero(np.isnan(a), axis=axis),
    ),
    nanmax: dict(
        numbagg=lambda a, axis=-1: partial(nanmax, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).max().T,
        bottleneck=lambda a, axis=-1: partial(bn.nanmax, a, axis=axis),
        numpy=lambda a, axis=-1: partial(np.nanmax, a, axis=axis),
    ),
    nanmean: dict(
        numbagg=lambda a, axis=-1: partial(nanmean, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).mean().T,
        bottleneck=lambda a, axis=-1: partial(bn.nanmean, a, axis=axis),
        numpy=lambda a, axis=-1: partial(np.nanmean, a, axis=axis),
    ),
    nanmin: dict(
        numbagg=lambda a, axis=-1: partial(nanmin, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).min().T,
        bottleneck=lambda a, axis=-1: partial(bn.nanmin, a, axis=axis),
        numpy=lambda a, axis=-1: partial(np.nanmin, a, axis=axis),
    ),
    nanstd: dict(
        numbagg=lambda a, axis=-1: partial(nanstd, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).std().T,
        bottleneck=lambda a, axis=-1: partial(bn.nanstd, a, axis=axis, ddof=1),
        numpy=lambda a, axis=-1: partial(np.nanstd, a, ddof=1, axis=axis),
    ),
    nanvar: dict(
        numbagg=lambda a, axis=-1: partial(nanvar, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).var().T,
        bottleneck=lambda a, axis=-1: partial(bn.nanvar, a, axis=axis, ddof=1),
        numpy=lambda a, axis=-1: partial(np.nanvar, a, ddof=1, axis=axis),
    ),
    anynan: dict(
        numbagg=lambda a, axis=-1: partial(anynan, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).isna().any(),
        bottleneck=lambda a, axis=-1: partial(bn.anynan, a, axis=axis),
    ),
    allnan: dict(
        numbagg=lambda a, axis=-1: partial(allnan, a, axis=axis),
        pandas=lambda a: lambda: _df_of_array(a).isna().all(),
        bottleneck=lambda a, axis=-1: partial(bn.allnan, a, axis=axis),
    ),
    move_exp_nanvar: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_setup(lambda df: df.var().T, a, alpha),
        numbagg=lambda a, alpha=0.5, **kwargs: partial(
            move_exp_nanvar, a, alpha=alpha, **kwargs
        ),
    ),
    move_exp_nanmean: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_setup(lambda df: df.mean().T, a, alpha),
        numbagg=lambda a, alpha=0.5, **kwargs: partial(
            move_exp_nanmean, a, alpha=alpha, **kwargs
        ),
    ),
    move_exp_nancount: dict(
        pandas=pandas_ewm_nancount_setup,
        numbagg=lambda a, alpha=0.5, **kwargs: partial(
            move_exp_nancount, a, alpha=alpha, **kwargs
        ),
    ),
    move_exp_nanstd: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_setup(lambda df: df.std().T, a, alpha),
        numbagg=lambda a, alpha=0.5, **kwargs: partial(
            move_exp_nanstd, a, alpha=alpha, **kwargs
        ),
    ),
    move_exp_nansum: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_setup(lambda df: df.sum().T, a, alpha),
        numbagg=lambda a, alpha=0.5, **kwargs: partial(
            move_exp_nansum, a, alpha=alpha, **kwargs
        ),
    ),
    move_exp_nancorr: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_2_array_setup(
            lambda df1, df2: df1.corr(df2).T, a, alpha=alpha
        ),
        numbagg=lambda a, alpha=0.5, **kwargs: numbagg_two_array_setup(
            move_exp_nancorr, a, alpha=alpha, **kwargs
        ),
    ),
    move_exp_nancov: dict(
        pandas=lambda a, alpha=0.5: pandas_ewm_2_array_setup(
            lambda df1, df2: df1.cov(df2).T, a, alpha=alpha
        ),
        numbagg=lambda a, alpha=0.5, **kwargs: numbagg_two_array_setup(
            move_exp_nancov, a, alpha=alpha, **kwargs
        ),
    ),
    move_mean: dict(
        pandas=lambda a, **kwargs: pandas_move_setup(
            lambda df: df.mean().T, a, **kwargs
        ),
        numbagg=lambda a, window=20, **kwargs: partial(
            move_mean, a, window=window, **kwargs
        ),
        bottleneck=lambda a, window=20, **kwargs: partial(
            bn.move_mean, a, window=window, **kwargs
        ),
    ),
    move_sum: dict(
        pandas=lambda a, **kwargs: pandas_move_setup(
            lambda df: df.sum().T, a, **kwargs
        ),
        numbagg=lambda a, window=20, **kwargs: partial(
            move_sum, a, window=window, **kwargs
        ),
        bottleneck=lambda a, window=20, **kwargs: partial(
            bn.move_sum, a, window=window, **kwargs
        ),
    ),
    move_std: dict(
        pandas=lambda a, **kwargs: pandas_move_setup(
            lambda df: df.std().T, a, **kwargs
        ),
        numbagg=lambda a, window=20, **kwargs: partial(
            move_std, a, window=window, **kwargs
        ),
        bottleneck=lambda a, window=20, **kwargs: partial(
            bn.move_std, a, window=window, ddof=1, **kwargs
        ),
    ),
    move_var: dict(
        pandas=lambda a, **kwargs: pandas_move_setup(
            lambda df: df.var().T, a, **kwargs
        ),
        numbagg=lambda a, window=20, **kwargs: partial(
            move_var, a, window=window, **kwargs
        ),
        bottleneck=lambda a, window=20, **kwargs: partial(
            bn.move_var, a, window=window, ddof=1, **kwargs
        ),
    ),
    move_corr: dict(
        pandas=lambda a, **kwargs: pandas_move_2_array_setup(
            lambda df1, df2: df1.corr(df2).T, a, **kwargs
        ),
        numbagg=lambda a, window=20, **kwargs: numbagg_two_array_setup(
            move_corr, a, window=window, **kwargs
        ),
    ),
    move_cov: dict(
        pandas=lambda a, **kwargs: pandas_move_2_array_setup(
            lambda df1, df2: df1.cov(df2).T, a, **kwargs
        ),
        numbagg=lambda a, window=20, **kwargs: numbagg_two_array_setup(
            move_cov, a, window=window, **kwargs
        ),
    ),
    ffill: dict(
        pandas=lambda a, **kwargs: lambda: _df_of_array(a).ffill(**kwargs).T,
        numbagg=lambda a, **kwargs: partial(ffill, a, **kwargs),
        bottleneck=lambda a, limit=None: partial(bn.push, a, limit),
    ),
    bfill: dict(
        pandas=lambda a, **kwargs: lambda: _df_of_array(a).bfill(**kwargs).T,
        numbagg=lambda a, **kwargs: partial(bfill, a, **kwargs),
        bottleneck=lambda a, limit=None: lambda: bn.push(a[..., ::-1], limit)[
            ..., ::-1
        ],
    ),
    nanmedian: dict(
        pandas=lambda a,: lambda: _df_of_array(a).median(),
        numbagg=lambda a, axis=-1: partial(nanmedian, a, axis=axis),
        numpy=lambda a, axis=-1: partial(np.nanmedian, a, axis=axis),
    ),
    nanquantile: dict(
        pandas=lambda a, quantiles=[0.25, 0.75]: lambda: _df_of_array(a).quantile(
            quantiles
        ),
        numbagg=lambda a, quantiles=[0.25, 0.75], axis=-1: partial(
            nanquantile, a, quantiles, axis=axis
        ),
        numpy=lambda a, quantiles=[0.25, 0.75], axis=-1: partial(
            np.nanquantile, a, quantiles, axis=axis
        ),
    ),
    group_nanmean: dict(
        pandas=partial(pandas_group_setup, "mean"),
        numbagg=partial(numbagg_group_setup, group_nanmean),
    ),
    group_nansum: dict(
        pandas=partial(pandas_group_setup, "sum"),
        numbagg=partial(numbagg_group_setup, group_nansum),
    ),
    group_nanvar: dict(
        pandas=partial(pandas_group_setup, "var"),
        numbagg=partial(numbagg_group_setup, group_nanvar),
    ),
    group_nanstd: dict(
        pandas=partial(pandas_group_setup, "std"),
        numbagg=partial(numbagg_group_setup, group_nanstd),
    ),
    group_nanall: dict(
        pandas=partial(pandas_group_setup, "all"),
        numbagg=partial(numbagg_group_setup, group_nanall),
    ),
    group_nanany: dict(
        pandas=partial(pandas_group_setup, "any"),
        numbagg=partial(numbagg_group_setup, group_nanany),
    ),
    group_nanargmax: dict(
        pandas=partial(pandas_group_setup, "idxmax"),
        numbagg=partial(numbagg_group_setup, group_nanargmax),
    ),
    group_nanargmin: dict(
        pandas=partial(pandas_group_setup, "idxmin"),
        numbagg=partial(numbagg_group_setup, group_nanargmin),
    ),
    group_nancount: dict(
        pandas=partial(pandas_group_setup, "count"),
        numbagg=partial(numbagg_group_setup, group_nancount),
    ),
    group_nanfirst: dict(
        pandas=partial(pandas_group_setup, "first"),
        numbagg=partial(numbagg_group_setup, group_nanfirst),
    ),
    group_nanlast: dict(
        pandas=partial(pandas_group_setup, "last"),
        numbagg=partial(numbagg_group_setup, group_nanlast),
    ),
    group_nanmax: dict(
        pandas=partial(pandas_group_setup, "max"),
        numbagg=partial(numbagg_group_setup, group_nanmax),
    ),
    group_nanmin: dict(
        pandas=partial(pandas_group_setup, "min"),
        numbagg=partial(numbagg_group_setup, group_nanmin),
    ),
    group_nanprod: dict(
        pandas=partial(pandas_group_setup, "prod"),
        numbagg=partial(numbagg_group_setup, group_nanprod),
    ),
    group_nansum_of_squares: dict(
        pandas=pandas_nan_sum_of_squares_setup,
        numbagg=partial(numbagg_group_setup, group_nansum_of_squares),
    ),
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


@pytest.fixture(params=["numbagg"], scope="module")
def library(request):
    """By default, limits to numbagg. But can be extended to pandas and bottleneck

    ```
    @pytest.mark.parametrize("library", ["numbagg", "pandas", "bottleneck", "numpy"], indirect=True)
        def test_func():
            # ...
    ```

    """
    return request.param


@pytest.fixture(params=COMPARISONS.keys(), scope="module")
def func(request):
    """By default, all functions here. But can be limited in a test:

    ```
    @pytest.mark.parametrize("func", [move_mean], indirect=True)
        def test_func():
            # ...
    ```
    """

    return request.param


@pytest.fixture(params=[(3, 1_000)], scope="module")
def shape(request):
    return request.param


# One disadvantage of having this in a fixture is that pytest keeps it around for the
# whole test session. So we at least use module scoping so it will keep a single one for
# all tests.
@pytest.fixture(scope="module")
def array(shape):
    array = np.random.RandomState(0).rand(*shape)
    return np.where(array > 0.1, array, np.nan)


@pytest.fixture(scope="module")
def func_callable(library, func, array):
    """
    The function post-setup; just needs to be called with `func_callable()`
    """
    if len(array.shape) > 2 and library == "pandas":
        pytest.skip("pandas doesn't support array with more than 2 dimensions")
    if (
        len(array.shape) > 1
        and library == "numbagg"
        and not getattr(func, "supports_nd", True)
    ):
        pytest.skip(f"{func} doesn't support nd")
    try:
        callable_ = COMPARISONS[func][library](array)
        assert callable(callable_)
        return callable_
    except KeyError:
        if library in ["bottleneck", "numpy"]:
            pytest.skip(f"{library} doesn't support {func}")
        else:
            raise


@pytest.fixture(autouse=True)
def numba_logger():
    # This is extremely noisy, so we turn it off. We can make this a setting if it would
    # be occasionally useful.
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.INFO)


# @pytest.fixture(scope="module")
@pytest.fixture(scope="function")
def rs():
    return np.random.RandomState(0)
