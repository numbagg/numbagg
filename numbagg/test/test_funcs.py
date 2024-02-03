from functools import partial

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

from numbagg import (
    AGGREGATION_FUNCS,
    bfill,
    ffill,
    nanargmax,
    nanargmin,
    nancount,
    nanmax,
    nanmean,
    nanmin,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
)
from numbagg.moving_exp import move_exp_nanmean
from numbagg.test.util import arrays

from .conftest import COMPARISONS


@pytest.mark.parametrize(
    "func",
    [ffill, bfill],
)
@pytest.mark.parametrize("limit", [1, 3, None])
@pytest.mark.parametrize("shape", [(3, 500)], indirect=True)
def test_fill_pandas_comp(func, array, limit):
    c = COMPARISONS[func]

    result = c["numbagg"](array, limit=limit)()
    expected = c["pandas"](array, limit=limit)()
    assert_allclose(result, expected)

    if c.get("bottleneck"):
        expected_bottleneck = c["bottleneck"](array, limit=limit)()
        assert_allclose(result, expected_bottleneck)


@pytest.mark.parametrize(
    "func",
    AGGREGATION_FUNCS,
)
@pytest.mark.parametrize("shape", [(2, 500)], indirect=True)
def test_aggregation_comparison(func, array):
    c = COMPARISONS[func]
    kwargs: dict = {}

    result = c["numbagg"](array, **kwargs)()
    expected = c["pandas"](array, **kwargs)()
    assert_allclose(result, expected)

    if c.get("bottleneck"):
        expected = c["bottleneck"](array, **kwargs)()
        assert_allclose(result, expected)

    if c.get("numpy"):
        expected = c["numpy"](array, **kwargs)()
        assert_allclose(result, expected)


@pytest.mark.parametrize("limit", [1, 3, None])
@pytest.mark.parametrize(
    "func",
    [
        ffill,
        bfill,
    ],
)
def test_fill_comparison(func, array, limit):
    c = COMPARISONS[func]
    kwargs = dict(limit=limit)

    result = c["numbagg"](array, **kwargs)()
    expected = c["pandas"](array, **kwargs)()
    assert_allclose(result, expected)

    if c.get("bottleneck"):
        expected = c["bottleneck"](array, **kwargs)()
        assert_allclose(result, expected)

    if c.get("numpy"):
        expected = c["numpy"](array, **kwargs)()
        assert_allclose(result, expected)


@pytest.mark.parametrize("quantiles", [0.5, [0.25, 0.75]])
def test_quantile_comparison(array, quantiles):
    c = COMPARISONS[nanquantile]
    kwargs = dict(quantiles=quantiles)

    result = c["numbagg"](array, **kwargs)()
    assert_allclose(result, c["numbagg"](array, **kwargs)())
    expected = c["pandas"](array, **kwargs)().values
    assert_allclose(result, expected)

    if c.get("bottleneck"):
        expected = c["bottleneck"](array, **kwargs)()
        assert_allclose(result, expected)

    if c.get("numpy"):
        expected = c["numpy"](array, **kwargs)()
        assert_allclose(result, expected)


def functions():
    # TODO: test tuple axes
    yield nansum, np.nansum, np.inf
    yield nanmax, np.nanmax, np.inf
    yield nanargmin, np.nanargmin, np.inf
    yield nanargmax, np.nanargmax, np.inf
    yield nanmin, np.nanmin, np.inf
    yield nanmean, np.nanmean, 5
    yield nanmean, np.nanmean, True
    yield nanstd, partial(np.nanstd, ddof=1), 5
    yield nanvar, partial(np.nanvar, ddof=1), 5
    # yield anynan, bn.anynan, np.inf
    # yield allnan, bn.allnan, np.inf
    yield nancount, slow_count, np.inf
    yield (
        lambda x: nanquantile(x, [0.25, 0.75]),
        lambda x: np.nanquantile(x, [0.25, 0.75]),
        5,
    )
    yield (
        lambda x: nanquantile(x, 0.5),
        lambda x: np.nanquantile(x, 0.5),
        5,
    )


@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice")
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.parametrize("numbagg_func,comp_func,decimal", functions())
def test_numerical_results_identical(numbagg_func, comp_func, decimal):
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
def test_nanquantile(axis, quantiles, rs):
    arr = rs.rand(2000).reshape(10, 10, -1)
    arr = np.arange(60).reshape(3, 4, 5).astype(np.float64)

    result = nanquantile(arr, quantiles, axis=axis)
    expected = np.nanquantile(arr, quantiles, axis=axis)

    assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("quantiles", [-0.5, [0.25, -0.75], [1.5], [0.5, 1.5]])
def test_nanquantile_errors(quantiles):
    array = np.random.rand(10, 10)
    with pytest.raises(ValueError, match="quantiles must be in the range"):
        nanquantile(array, quantiles)


def test_nanquantile_no_valid_obs():
    array = np.full((10, 10), np.nan)
    result = nanquantile(array, 0.5)
    assert np.isnan(result)


def test_wraps():
    assert move_exp_nanmean.__name__ == "move_exp_nanmean"  # type: ignore
    assert move_exp_nanmean.__repr__() == "numbagg.move_exp_nanmean"
    assert "Exponentially" in move_exp_nanmean.__doc__  # type: ignore
