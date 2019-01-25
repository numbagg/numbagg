import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from numbagg import move_mean, move_exp_nanmean
from .util import arrays, array_order


@pytest.fixture
def rand_array():
    arr = np.random.RandomState(0).rand(2000).reshape(10, -1)
    return np.where(arr > 0.1, arr, np.nan)


@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_move_exp_nanmean(rand_array, alpha):

    array = rand_array[0]
    expected = pd.Series(array).ewm(alpha=alpha).mean()
    result = move_exp_nanmean(array, alpha)

    assert_almost_equal(expected, result)


def test_move_exp_nanmean_2d(rand_array):

    expected = pd.DataFrame(rand_array).T.ewm(alpha=0.1).mean().T
    result = move_exp_nanmean(rand_array, 0.1)

    assert_almost_equal(expected, result)


def test_move_mean():
    array = np.arange(100.0)
    array[::7] = np.nan

    expected = pd.Series(array).rolling(window=5, min_periods=1).mean().values
    result = move_mean(array, 5, min_count=1)
    assert_almost_equal(expected, result)


def test_move_mean_random(rand_array):
    array = rand_array[0]

    expected = pd.Series(array).rolling(window=10, min_periods=1).mean().values
    result = move_mean(array, 10, min_count=1)
    assert_almost_equal(expected, result)

    expected = pd.Series(array).rolling(window=3, min_periods=3).mean().values
    result = move_mean(array, 3, min_count=3)
    assert_almost_equal(expected, result)


def test_move_mean_window(rand_array):

    with pytest.raises(TypeError):
        move_mean(rand_array, window=0.5)
    with pytest.raises(ValueError):
        move_mean(rand_array, window=-1)
    with pytest.raises(ValueError):
        move_mean(rand_array, window=1, min_count=-1)


def functions():
    yield move_mean, slow_move_mean


@pytest.mark.parametrize("func,func0", functions())
def test_numerical_results_identical(func, func0):
    "Test that bn.xxx gives the same output as a reference function."
    fmt = (
        "\nfunc %s | window %d | min_count %s | input %s (%s) | shape %s | "
        "axis %s | order %s\n"
    )
    fmt += "\nInput array:\n%s\n"
    func_name = func.__name__
    if func_name == "move_var":
        decimal = 3
    else:
        decimal = 5
    for i, a in enumerate(arrays(func_name)):
        axes = range(-1, a.ndim)
        for axis in axes:
            windows = range(1, a.shape[axis])
            for window in windows:
                min_counts = list(range(1, window + 1)) + [None]
                for min_count in min_counts:
                    actual = func(a, window, min_count, axis=axis)
                    desired_a = a.astype(np.float32) if a.dtype == np.float16 else a
                    desired = func0(desired_a, window, min_count, axis=axis)
                    tup = (
                        func_name,
                        window,
                        str(min_count),
                        "a" + str(i),
                        str(a.dtype),
                        str(a.shape),
                        str(axis),
                        array_order(a),
                        a,
                    )
                    err_msg = fmt % tup
                    np.testing.assert_array_almost_equal(
                        actual, desired, decimal, err_msg
                    )
                    err_msg += "\n dtype mismatch %s %s"
                    da = actual.dtype
                    dd = desired.dtype
                    # don't require an exact dtype match, since we don't care
                    # about endianness of the result
                    assert da.kind == dd.kind, err_msg % (da, dd)
                    assert da.itemsize == dd.itemsize, err_msg % (da, dd)


def slow_move_sum(a, window, min_count=None, axis=-1):
    "Slow move_sum for unaccelerated dtype"
    return move_func(np.nansum, a, window, min_count, axis=axis)


def slow_move_mean(a, window, min_count=None, axis=-1):
    "Slow move_mean for unaccelerated dtype"
    return move_func(np.nanmean, a, window, min_count, axis=axis)


def slow_move_std(a, window, min_count=None, axis=-1, ddof=0):
    "Slow move_std for unaccelerated dtype"
    return move_func(np.nanstd, a, window, min_count, axis=axis, ddof=ddof)


def slow_move_var(a, window, min_count=None, axis=-1, ddof=0):
    "Slow move_var for unaccelerated dtype"
    return move_func(np.nanvar, a, window, min_count, axis=axis, ddof=ddof)


def slow_move_min(a, window, min_count=None, axis=-1):
    "Slow move_min for unaccelerated dtype"
    return move_func(np.nanmin, a, window, min_count, axis=axis)


def slow_move_max(a, window, min_count=None, axis=-1):
    "Slow move_max for unaccelerated dtype"
    return move_func(np.nanmax, a, window, min_count, axis=axis)


def slow_move_argmin(a, window, min_count=None, axis=-1):
    "Slow move_argmin for unaccelerated dtype"

    def argmin(a, axis):
        a = np.array(a, copy=False)
        flip = [slice(None)] * a.ndim
        flip[axis] = slice(None, None, -1)
        a = a[flip]  # if tie, pick index of rightmost tie
        try:
            idx = np.nanargmin(a, axis=axis)
        except ValueError:
            # an all nan slice encountered
            a = a.copy()
            mask = np.isnan(a)
            np.copyto(a, np.inf, where=mask)
            idx = np.argmin(a, axis=axis).astype(np.float64)
            if idx.ndim == 0:
                idx = np.nan
            else:
                mask = np.all(mask, axis=axis)
                idx[mask] = np.nan
        return idx

    return move_func(argmin, a, window, min_count, axis=axis)


def slow_move_argmax(a, window, min_count=None, axis=-1):
    "Slow move_argmax for unaccelerated dtype"

    def argmax(a, axis):
        a = np.array(a, copy=False)
        flip = [slice(None)] * a.ndim
        flip[axis] = slice(None, None, -1)
        a = a[flip]  # if tie, pick index of rightmost tie
        try:
            idx = np.nanargmax(a, axis=axis)
        except ValueError:
            # an all nan slice encountered
            a = a.copy()
            mask = np.isnan(a)
            np.copyto(a, -np.inf, where=mask)
            idx = np.argmax(a, axis=axis).astype(np.float64)
            if idx.ndim == 0:
                idx = np.nan
            else:
                mask = np.all(mask, axis=axis)
                idx[mask] = np.nan
        return idx

    return move_func(argmax, a, window, min_count, axis=axis)


def slow_move_median(a, window, min_count=None, axis=-1):
    "Slow move_median for unaccelerated dtype"
    return move_func(np.nanmedian, a, window, min_count, axis=axis)


def slow_move_rank(a, window, min_count=None, axis=-1):
    "Slow move_rank for unaccelerated dtype"
    return move_func(lastrank, a, window, min_count, axis=axis)


# magic utility functions ---------------------------------------------------


def move_func(func, a, window, min_count=None, axis=-1, **kwargs):
    "Generic moving window function implemented with a python loop."
    a = np.array(a, copy=False)
    if min_count is None:
        mc = window
    else:
        mc = min_count
        if mc > window:
            msg = "min_count (%d) cannot be greater than window (%d)"
            raise ValueError(msg % (mc, window))
        elif mc <= 0:
            raise ValueError("`min_count` must be greater than zero.")
    if a.ndim == 0:
        raise ValueError("moving window functions require ndim > 0")
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[axis]:
        raise ValueError("`window` is too long.")
    if issubclass(a.dtype.type, np.inexact):
        y = np.empty_like(a)
    else:
        y = np.empty(a.shape)
    idx1 = [slice(None)] * a.ndim
    idx2 = list(idx1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(a.shape[axis]):
            win = min(window, i + 1)
            idx1[axis] = slice(i + 1 - win, i + 1)
            idx2[axis] = i
            y[tuple(idx2)] = func(a[tuple(idx1)], axis=axis, **kwargs)
    idx = _mask(a, window, mc, axis)
    y[idx] = np.nan
    return y


def _mask(a, window, min_count, axis):
    n = (a == a).cumsum(axis)
    idx1 = [slice(None)] * a.ndim
    idx2 = [slice(None)] * a.ndim
    idx3 = [slice(None)] * a.ndim
    idx1[axis] = slice(window, None)
    idx2[axis] = slice(None, -window)
    idx3[axis] = slice(None, window)
    idx1 = tuple(idx1)
    idx2 = tuple(idx2)
    idx3 = tuple(idx3)
    nidx1 = n[idx1]
    nidx1 = nidx1 - n[idx2]
    idx = np.empty(a.shape, dtype=np.bool)
    idx[idx1] = nidx1 < min_count
    idx[idx3] = n[idx3] < min_count
    return idx


def lastrank(a, axis=-1):
    """
    The ranking of the last element along the axis, ignoring NaNs.

    The ranking is normalized to be between -1 and 1 instead of the more
    common 1 and N. The results are adjusted for ties.

    Parameters
    ----------
    a : ndarray
        Input array. If `a` is not an array, a conversion is attempted.
    axis : int, optional
        The axis over which to rank. By default (axis=-1) the ranking
        (and reducing) is performed over the last axis.

    Returns
    -------
    d : array
        In the case of, for example, a 2d array of shape (n, m) and
        axis=1, the output will contain the rank (normalized to be between
        -1 and 1 and adjusted for ties) of the the last element of each row.
        The output in this example will have shape (n,).

    Examples
    --------
    Create an array:

    >>> y1 = larry([1, 2, 3])

    What is the rank of the last element (the value 3 in this example)?
    It is the largest element so the rank is 1.0:

    >>> import numpy as np
    >>> from la.afunc import lastrank
    >>> x1 = np.array([1, 2, 3])
    >>> lastrank(x1)
    1.0

    Now let's try an example where the last element has the smallest
    value:

    >>> x2 = np.array([3, 2, 1])
    >>> lastrank(x2)
    -1.0

    Here's an example where the last element is not the minimum or maximum
    value:

    >>> x3 = np.array([1, 3, 4, 5, 2])
    >>> lastrank(x3)
    -0.5

    """
    a = np.array(a, copy=False)
    ndim = a.ndim
    if a.size == 0:
        # At least one dimension has length 0
        shape = list(a.shape)
        shape.pop(axis)
        r = np.empty(shape, dtype=a.dtype)
        r.fill(np.nan)
        if (r.ndim == 0) and (r.size == 1):
            r = np.nan
        return r
    indlast = [slice(None)] * ndim
    indlast[axis] = slice(-1, None)
    indlast = tuple(indlast)
    indlast2 = [slice(None)] * ndim
    indlast2[axis] = -1
    indlast2 = tuple(indlast2)
    n = (~np.isnan(a)).sum(axis)
    a_indlast = a[indlast]
    g = (a_indlast > a).sum(axis)
    e = (a_indlast == a).sum(axis)
    r = (g + g + e - 1.0) / 2.0
    r = r / (n - 1.0)
    r = 2.0 * (r - 0.5)
    if ndim == 1:
        if n == 1:
            r = 0
        if np.isnan(a[indlast2]):  # elif?
            r = np.nan
    else:
        np.putmask(r, n == 1, 0)
        np.putmask(r, np.isnan(a[indlast2]), np.nan)
    return r
