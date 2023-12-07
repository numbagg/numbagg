from functools import partial

import numpy as np
import pandas as pd
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

import numbagg
from numbagg import bfill, ffill
from numbagg.test.util import arrays

from .conftest import COMPARISONS


@pytest.fixture(scope="module")
def rand_array():
    arr = np.random.RandomState(0).rand(2000).reshape(10, -1)
    arr[0, 0] = np.nan
    return np.where(arr > 0.1, arr, np.nan)


@pytest.mark.parametrize(
    "func",
    [ffill, bfill],
)
@pytest.mark.parametrize("limit", [1, 3, None])
def test_fill_pandas_comp(rand_array, limit, func):
    c = COMPARISONS[func]
    array = rand_array[:3]

    result = c["numbagg"](array, limit=limit)()
    expected = c["pandas"](array, limit=limit)()
    if c.get("bottleneck"):
        expected_bottleneck = c["bottleneck"](array, limit=limit)()
        assert_allclose(result, expected_bottleneck)

    assert_allclose(result, expected)


def functions():
    # TODO: test tuple axes
    yield numbagg.nansum, np.nansum, np.inf
    yield numbagg.nanmax, np.nanmax, np.inf
    yield numbagg.nanargmin, np.nanargmin, np.inf
    yield numbagg.nanargmax, np.nanargmax, np.inf
    yield numbagg.nanmin, np.nanmin, np.inf
    yield numbagg.nanmean, np.nanmean, 5
    yield numbagg.nanmean, np.nanmean, True
    yield numbagg.nanstd, partial(np.nanstd, ddof=1), 5
    yield numbagg.nanvar, partial(np.nanvar, ddof=1), 5
    # yield numbagg.anynan, bn.anynan, np.inf
    # yield numbagg.allnan, bn.allnan, np.inf
    yield numbagg.nancount, slow_count, np.inf
    yield (
        lambda x: numbagg.nanquantile(x, [0.25, 0.75]),
        lambda x: np.nanquantile(x, [0.25, 0.75]),
        5,
    )
    yield (
        lambda x: numbagg.nanquantile(x, 0.5),
        lambda x: np.nanquantile(x, 0.5),
        5,
    )


@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice")
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.parametrize("numbagg_func,comp_func,decimal", functions())
def test_numerical_results_identical(numbagg_func, comp_func, decimal):
    "Test that bn.xxx gives the same output as bn.slow.xxx."
    msg = "\nfunc %s | input %s (%s) | shape %s | axis %s\n"
    msg += "\nInput array:\n%s\n"
    for i, arr in enumerate(arrays(numbagg_func.__name__)):
        for axis in list(range(-arr.ndim, arr.ndim)) + [None]:
            with np.errstate(invalid="ignore"):
                desiredraised = False
                desired_arr = arr.copy()
                if desired_arr.dtype == np.float16:
                    # don't use float16 for computation
                    desired_arr = desired_arr.astype(np.float32)
                try:
                    desired = comp_func(desired_arr, axis=axis)
                except Exception as err:
                    desired = str(err)
                    desiredraised = True
                actualraised = False
                try:
                    actual = numbagg_func(arr.copy(), axis=axis)
                except Exception as err:
                    if not desiredraised:
                        raise
                    actual = str(err)
                    actualraised = True
            if actualraised and desiredraised:
                assert desired == actual
            elif desiredraised and actual.size == 0:
                # there are no array values, so don't worry about not raising
                pass
            else:
                assert actualraised == desiredraised

                actual = np.asarray(actual)
                desired = np.asarray(desired)

                tup = (
                    numbagg_func.__name__,
                    "a" + str(i),
                    str(arr.dtype),
                    str(arr.shape),
                    str(axis),
                    arr,
                )
                err_msg = msg % tup
                if (decimal < np.inf) and (np.isfinite(arr).sum() > 0):
                    assert_array_almost_equal(actual, desired, decimal, err_msg)
                else:
                    assert_array_equal(actual, desired, err_msg)

                err_msg += "\n dtype mismatch %s %s"
                da = actual.dtype
                dd = desired.dtype
                assert_equal(da, dd, err_msg % (da, dd))


def slow_count(x, axis=None):
    return np.sum(~np.isnan(x), axis=axis)


@pytest.mark.parametrize("axis", [None, -1, 1, (1, 2), (0,), (-1, -2)])
@pytest.mark.parametrize("quantiles", [0.5, [0.25, 0.75]])
def test_nan_quantile(axis, quantiles):
    arr = np.random.RandomState(0).rand(2000).reshape(10, 10, -1)
    arr = np.arange(60).reshape(3, 4, 5).astype(np.float64)

    # quantiles = np.array([0.25, 0.75])
    result = numbagg.nanquantile(arr, quantiles, axis=axis)
    expected = np.nanquantile(arr, quantiles, axis=axis)

    assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("limit", [1, 3, None])
def test_ffill(rand_array, limit):
    a = rand_array[0]
    expected = pd.Series(a).ffill(limit=limit).values
    result = ffill(a, limit=limit)

    assert_allclose(result, expected)


@pytest.mark.parametrize("limit", [1, 3, None])
def test_bfill(rand_array, limit):
    a = rand_array[0]
    expected = pd.Series(a).bfill(limit=limit).values
    result = bfill(a, limit=limit)

    assert_allclose(result, expected)
