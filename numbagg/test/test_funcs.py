import pytest
import numbagg

import bottleneck as bn
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal


def arrays(dtypes=numbagg.dtypes, nans=True):
    "Iterator that yields arrays to use for unit testing."
    ss = {}
    ss[0] = {"size": 0, "shapes": [(0,), (0, 0), (2, 0), (2, 0, 1)]}
    # ss[0] = {'size':  0, 'shapes': [(0,), (0, 0)]}
    ss[1] = {"size": 4, "shapes": [(4,)]}
    ss[2] = {"size": 6, "shapes": [(1, 6), (2, 3)]}
    ss[3] = {"size": 6, "shapes": [(1, 2, 3)]}
    ss[4] = {"size": 24, "shapes": [(1, 2, 3, 4)]}  # Unaccelerated
    for ndim in ss:
        size = ss[ndim]["size"]
        shapes = ss[ndim]["shapes"]
        for dtype in dtypes:
            a = np.arange(size, dtype=dtype)
            for shape in shapes:
                a = a.reshape(shape)
                yield a
                yield -a
                # nanargmax/nanargmin regression tests
                yield np.zeros_like(a)
            if issubclass(a.dtype.type, np.inexact):
                if nans:
                    for i in range(a.size):
                        a.flat[i] = np.nan
                        yield a
                        yield -a
                for i in range(a.size):
                    a.flat[i] = np.inf
                    yield a
                    yield -a
    if nans:
        # nanmedian regression tests
        a = np.array([1, np.nan, np.nan, 2])
        yield a
        a = np.vstack((a, a))
        yield a
        yield a.reshape(1, 2, 4)


def functions():
    # TODO: test tuple axes
    yield numbagg.nansum, np.nansum, np.inf
    yield numbagg.nanmax, np.nanmax, np.inf
    yield numbagg.nanargmin, np.nanargmin, np.inf
    yield numbagg.nanargmax, np.nanargmax, np.inf
    yield numbagg.nanmin, np.nanmin, np.inf
    yield numbagg.nanmean, np.nanmean, 5
    yield numbagg.nanstd, np.nanstd, 5
    yield numbagg.nanvar, np.nanvar, 5
    yield numbagg.anynan, bn.anynan, np.inf
    yield numbagg.allnan, bn.allnan, np.inf
    yield numbagg.count, slow_count, np.inf


@pytest.mark.parametrize("func,func0,decimal", functions())
def test_numerical_results_identical(func, func0, decimal, nans=True):
    "Test that bn.xxx gives the same output as bn.slow.xxx."
    msg = "\nfunc %s | input %s (%s) | shape %s | axis %s\n"
    msg += "\nInput array:\n%s\n"
    for i, arr in enumerate(arrays(nans=nans)):
        for axis in list(range(-arr.ndim, arr.ndim)) + [None]:
            with np.errstate(invalid="ignore"):
                desiredraised = False
                try:
                    desired = func0(arr.copy(), axis=axis)
                except Exception as err:
                    desired = str(err)
                    desiredraised = True
                actualraised = False
                try:
                    actual = func(arr.copy(), axis=axis)
                except Exception as err:
                    if not desiredraised:
                        raise
                    actual = str(err)
                    actualraised = True
            if actualraised and desiredraised:
                pass
            else:
                actual = np.asarray(actual)
                desired = np.asarray(desired)

                if desiredraised:
                    # unlike bottleneck, numbagg cannot raise on invalid data,
                    # so check for sentinel values instead
                    if np.issubdtype(actual.dtype, np.dtype(int)):
                        fill_value = -1
                    else:
                        fill_value = np.nan
                    desired = np.empty(actual.shape, actual.dtype)
                    all_missing = np.asarray(numbagg.allnan(arr, axis=axis))
                    desired[all_missing] = fill_value
                    # for now, assume the non-missing values are calculated
                    # correctly
                    desired[~all_missing] = actual[~all_missing]

                tup = (
                    func.__name__,
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
