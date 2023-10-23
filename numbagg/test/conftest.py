from __future__ import annotations

import logging
from typing import Callable

import pandas as pd
import pytest

from .. import (
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


def pandas_ewm_setup(a, alpha=0.5):
    return pd.DataFrame(a).T.ewm(alpha=alpha)


def pandas_ewm_2_array_setup(a, alpha=0.5):
    a1, a2 = numbagg_2_array_setup(a)
    return pd.DataFrame(a1).T.ewm(alpha=alpha), pd.DataFrame(a2).T


def pandas_move_setup(a, window=20, min_count=None):
    return pd.DataFrame(a).T.rolling(window=window, min_periods=min_count)


def pandas_move_2_array_setup(a, window=20, min_count=None):
    a1, a2 = numbagg_2_array_setup(a)
    return (
        pd.DataFrame(a1).T.rolling(window=window, min_periods=min_count),
        pd.DataFrame(a2).T,
    )


def numbagg_2_array_setup(a):
    a1, a2 = a, a**2 + 1
    return a1, a2


def numbagg_move_run(func):
    return lambda a, window=20, min_count=None: func(
        a, window=window, min_count=min_count
    )


def numbagg_move_2_array_run(func):
    return lambda a1a2, window=20, min_count=None: func(
        a1a2[0], a1a2[1], window=window, min_count=min_count
    )


# Parameterization of tests and benchmarks
#
# - Each functions has a dict for each library we want to test.
# - Each library dict has a setup and run function.
# - The setup function takes an input array and returns an object that can be passed to
#   the run function. Sometimes this is a no-op, and it just passed back the array.
# - The run function should work by being passed the object returned by the setup, but
#   can have optional kwargs if we want to be able to test other parameters.
#   - But TODO: these aren't always consistent between pandas & numbagg. One option
#     would be for `setup` to return a function that just gets called, and package
#     anything like `alpha` in a closure.

COMPARISONS: dict[Callable, dict[str, dict[str, Callable]]] = {
    move_exp_nancount: dict(
        # There's no pandas equivalent for move_exp_nancount
        pandas=dict(
            setup=lambda a, alpha=0.5: pd.DataFrame(a).T.notnull().ewm(alpha=alpha),
            run=lambda a: a.sum(engine="numba").T,
        ),
        numbagg=dict(
            setup=lambda a: a,
            run=lambda a, alpha=0.5: move_exp_nancount(a, alpha=alpha),
        ),
    ),
    move_exp_nanvar: dict(
        pandas=dict(
            setup=pandas_ewm_setup,
            run=lambda a: a.var().T,
        ),
        numbagg=dict(
            setup=lambda a: a,
            run=lambda a, alpha=0.5: move_exp_nanvar(a, alpha=alpha),
        ),
    ),
    move_exp_nanstd: dict(
        pandas=dict(
            setup=pandas_ewm_setup,
            run=lambda a: a.std().T,
        ),
        numbagg=dict(
            setup=lambda a: a,
            run=lambda a, alpha=0.5: move_exp_nanstd(a, alpha=alpha),
        ),
    ),
    move_exp_nansum: dict(
        pandas=dict(
            setup=pandas_ewm_setup,
            run=lambda a: a.sum(engine="numba").T,
        ),
        numbagg=dict(
            setup=lambda a: a,
            run=lambda a, alpha=0.5: move_exp_nansum(a, alpha=alpha),
        ),
    ),
    move_exp_nanmean: dict(
        pandas=dict(
            setup=pandas_ewm_setup,
            run=lambda a: a.mean(engine="numba").T,
        ),
        numbagg=dict(
            setup=lambda a: a,
            run=lambda a, alpha=0.5: move_exp_nanmean(a, alpha=alpha),
        ),
    ),
    move_exp_nancorr: dict(
        pandas=dict(
            setup=pandas_ewm_2_array_setup,
            run=lambda arrays: arrays[0].corr(arrays[1]).T,
        ),
        numbagg=dict(
            setup=numbagg_2_array_setup,
            run=lambda a, alpha=0.5: move_exp_nancorr(*a, alpha=alpha),
        ),
    ),
    move_exp_nancov: dict(
        pandas=dict(
            setup=pandas_ewm_2_array_setup,
            run=lambda arrays: arrays[0].cov(arrays[1]).T,
        ),
        numbagg=dict(
            setup=numbagg_2_array_setup,
            run=lambda a, alpha=0.5: move_exp_nancov(*a, alpha=alpha),
        ),
    ),
    move_mean: dict(
        pandas=dict(
            setup=pandas_move_setup,
            run=lambda a: a.mean(engine="numba").T,
        ),
        numbagg=dict(
            setup=lambda a: a,
            run=numbagg_move_run(move_mean),
        ),
    ),
    move_sum: dict(
        pandas=dict(
            setup=pandas_move_setup,
            run=lambda a: a.sum(engine="numba").T,
        ),
        numbagg=dict(setup=lambda a: a, run=numbagg_move_run(move_sum)),
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
    move_std: dict(
        pandas=dict(
            setup=pandas_move_setup,
            run=lambda a: a.std(engine="numba").T,
        ),
        numbagg=dict(setup=lambda a: a, run=numbagg_move_run(move_std)),
    ),
    move_var: dict(
        pandas=dict(
            setup=pandas_move_setup,
            run=lambda a: a.var(engine="numba").T,
        ),
        numbagg=dict(setup=lambda a: a, run=numbagg_move_run(move_var)),
    ),
    move_corr: dict(
        pandas=dict(
            setup=pandas_move_2_array_setup,
            run=lambda arrays: arrays[0].corr(arrays[1]).T,
        ),
        numbagg=dict(
            setup=numbagg_2_array_setup, run=numbagg_move_2_array_run(move_corr)
        ),
    ),
    move_cov: dict(
        pandas=dict(
            setup=pandas_move_2_array_setup,
            run=lambda arrays: arrays[0].cov(arrays[1]).T,
        ),
        numbagg=dict(
            setup=numbagg_2_array_setup, run=numbagg_move_2_array_run(move_cov)
        ),
    ),
}


@pytest.fixture(params=["numbagg", "pandas"])
def library(request):
    return request.param


@pytest.fixture
def setup(library, func):
    return COMPARISONS[func][library]["setup"]


@pytest.fixture
def run(library, func):
    return COMPARISONS[func][library]["run"]


@pytest.fixture()
def obj(array, setup):
    return setup(array)


@pytest.fixture(autouse=True)
def numba_logger():
    # This is exteremly noisy, so we turn it off. We can make this a setting if it would
    # be occasionally useful.
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
