# import bottleneck as bn
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal

import numbagg
from numbagg.test.util import arrays


def functions():
    # TODO: test tuple axes
    yield numbagg.nansum, np.nansum, np.inf
    yield numbagg.nanmax, np.nanmax, np.inf
    yield numbagg.nanargmin, np.nanargmin, np.inf
    yield numbagg.nanargmax, np.nanargmax, np.inf
    yield numbagg.nanmin, np.nanmin, np.inf
    yield numbagg.nanmean, np.nanmean, 5
    yield numbagg.nanmean, np.nanmean, True
    yield numbagg.nanstd, np.nanstd, 5
    yield numbagg.nanvar, np.nanvar, 5
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
            # if numbagg_func == numbagg.nanquantile:
            #     breakpoint()
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


def test_nan_quantile():
    arr = np.random.RandomState(0).rand(2000).reshape(10, 10, -1)

    quantiles = np.array([0.25, 0.75])
    result = numbagg.nanquantile(arr, quantiles, axis=1)
    expected = np.nanquantile(arr, quantiles, axis=1)

    assert_array_almost_equal(result, expected)
