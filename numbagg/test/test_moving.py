import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from numbagg import (
    move_exp_nancorr,
    move_exp_nancount,
    move_exp_nancov,
    move_exp_nanmean,
    move_exp_nanstd,
    move_exp_nansum,
    move_exp_nanvar,
    move_mean,
)

from .util import array_order, arrays


@pytest.fixture
def rand_array():
    arr = np.random.RandomState(0).rand(2000).reshape(10, -1)
    arr[0, 0] = np.nan
    return np.where(arr > 0.1, arr, np.nan)


@pytest.mark.parametrize(
    "functions",
    [
        (move_exp_nanmean, lambda x: x.mean()),
        (move_exp_nansum, lambda x: x.sum()),
        (move_exp_nanvar, lambda x: x.var()),
        (move_exp_nanstd, lambda x: x.std()),
    ],
)
@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_move_exp_pandas_comp(rand_array, alpha, functions):
    array = rand_array[0]
    result = functions[0](array, alpha=alpha)
    expected = functions[1](pd.Series(array).ewm(alpha=alpha))

    assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "functions",
    [
        (move_exp_nancov, lambda x, y: x.cov(y)),
        (move_exp_nancorr, lambda x, y: x.corr(y)),
    ],
)
@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_move_exp_pandas_comp_two_arr(rand_array, alpha, functions):
    array = rand_array[0]
    array_2 = rand_array[0] + rand_array[1]
    result = functions[0](array, array_2, alpha=alpha)
    expected = functions[1](pd.Series(array).ewm(alpha=alpha), pd.Series(array_2))

    assert_almost_equal(result, expected)


def test_move_exp_nanmean_2d(rand_array):
    expected = pd.DataFrame(rand_array).T.ewm(alpha=0.1).mean().T
    result = move_exp_nanmean(rand_array, alpha=0.1)

    assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "func",
    [
        move_exp_nancount,
        move_exp_nanmean,
        move_exp_nansum,
        move_exp_nanvar,
        move_exp_nanstd,
    ],
)
def test_move_exp_min_weight(func):
    # Make an array of 25 values, with the first 5 being NaN, and then look at the final
    # 19. We can't look at the whole series, because `nanvar` will always return NaN for
    # the first value.
    array = np.ones(25)
    array[:5] = np.nan

    # min_weight of 0 should produce values everywhere
    result = np.sum(~np.isnan(func(array, min_weight=0.0, alpha=0.2))[6:])
    expected = 19
    assert result == expected
    result = np.sum(~np.isnan(func(array, min_weight=0.0, alpha=0.8))[6:])
    expected = 19
    assert result == expected

    result = np.sum(~np.isnan(func(array, min_weight=0.5, alpha=0.2))[6:])
    expected = 17
    assert result == expected
    result = np.sum(~np.isnan(func(array, min_weight=0.5, alpha=0.8))[6:])
    expected = 19
    assert result == expected

    result = np.sum(~np.isnan(func(array, min_weight=0.9, alpha=0.2))[6:])
    expected = 10
    assert result == expected
    result = np.sum(~np.isnan(func(array, min_weight=0.9, alpha=0.8))[6:])
    expected = 19
    assert result == expected

    # min_weight of 1 should never produce values
    result = np.sum(~np.isnan(func(array, min_weight=1.0, alpha=0.2))[6:])
    expected = 0
    assert result == expected
    result = np.sum(~np.isnan(func(array, min_weight=1.0, alpha=0.8))[6:])
    expected = 0
    assert result == expected


@pytest.mark.parametrize("n", [10, 200])
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("test_nans", [True, False])
def test_move_exp_min_weight_numerical(n, alpha, rand_array, test_nans):
    array = rand_array[0, :n]
    if not test_nans:
        array = np.nan_to_num(array)
    # High alphas mean fast decays, mean initial weights are higher
    initial_weight = alpha
    weights = (
        np.array([(1 - alpha) ** (i - 1) for i in range(n, 0, -1)]) * initial_weight
    )
    assert_almost_equal(weights[-1], initial_weight)
    # Fill weights with NaNs where array has them
    weights = np.where(np.isnan(array), np.nan, weights)

    # This is the weight of the final value
    weight = np.nansum(weights)

    # Run with min_weight slightly above the final value required, assert it doesn't let
    # it through
    result = move_exp_nanmean(array, alpha=alpha, min_weight=weight + 0.01)
    assert np.isnan(result[-1])

    # And with min_weight slightly below
    result = move_exp_nanmean(array, alpha=alpha, min_weight=weight - 0.01)
    assert not np.isnan(result[-1])


def test_move_exp_nancount_numeric():
    array = np.array([1, 0, np.nan, np.nan, 1, 0])

    result = move_exp_nancount(array, alpha=0.5)
    expected = np.array([1.0, 1.5, 0.75, 0.375, 1.1875, 1.59375])
    assert_almost_equal(result, expected)

    result = move_exp_nancount(array, alpha=0.25)
    expected = np.array([1.0, 1.75, 1.3125, 0.984375, 1.7382812, 2.3037109])
    assert_almost_equal(result, expected)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_move_exp_nancount_nansum(alpha):
    # An array with NaNs & 1s should be the same for count & sum
    array = np.array([1, 1, np.nan, np.nan, 1, 1])

    result = move_exp_nancount(array, alpha=alpha)
    expected = move_exp_nansum(array, alpha=alpha)
    assert_almost_equal(result, expected)


def test_move_exp_nanmean_numeric():
    array = np.array([10, 0, np.nan, 10])

    result = move_exp_nanmean(array, alpha=0.5)
    expected = np.array([10.0, 3.3333333, 3.3333333, 8.1818182])
    assert_almost_equal(result, expected)

    result = move_exp_nanmean(array, alpha=0.25)
    expected = np.array([10.0, 4.2857143, 4.2857143, 7.1653543])
    assert_almost_equal(result, expected)


def test_move_exp_nansum_numeric():
    array = np.array([10, 0, np.nan, 10])

    result = move_exp_nansum(array, alpha=0.5)
    expected = np.array([10.0, 5.0, 2.5, 11.25])
    assert_almost_equal(result, expected)

    result = move_exp_nansum(array, alpha=0.25)
    expected = np.array([10.0, 7.5, 5.625, 14.21875])
    assert_almost_equal(result, expected)


def test_move_exp_nancorr_numeric():
    array1 = np.array([10, 0, 5, 10])
    array2 = np.array([10, 0, 10, 5])

    result = move_exp_nancorr(array1, array2, alpha=0.5)
    expected = np.array([np.nan, 1.0, 0.8485281, 0.2274294])
    assert_almost_equal(result, expected)

    result = move_exp_nancorr(array1, array2, alpha=0.25)
    expected = np.array([np.nan, 1.0, 0.85, 0.4789468])
    assert_almost_equal(result, expected)


def test_move_mean():
    array = np.arange(100.0)
    array[::7] = np.nan

    expected = pd.Series(array).rolling(window=5, min_periods=1).mean().values
    result = move_mean(array, 5, min_count=1)
    assert_almost_equal(result, expected)  # type: ignore[arg-type,unused-ignore]


def test_move_mean_random(rand_array):
    array = rand_array[0]

    expected = pd.Series(array).rolling(window=10, min_periods=1).mean().values
    result = move_mean(array, 10, min_count=1)
    assert_almost_equal(result, expected)

    expected = pd.Series(array).rolling(window=3, min_periods=3).mean().values
    result = move_mean(array, 3, min_count=3)
    assert_almost_equal(result, expected)


def test_move_mean_window(rand_array):
    with pytest.raises(TypeError):
        move_mean(rand_array, window=0.5)
    with pytest.raises(ValueError):
        move_mean(rand_array, window=-1)
    with pytest.raises(ValueError):
        move_mean(rand_array, window=1, min_count=-1)


def test_tuple_axis_arg(rand_array):
    result = move_exp_nanmean(rand_array, alpha=0.1, axis=())
    assert_equal(result, rand_array)


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


def slow_move_std(a, window, min_count=None, axis=-1, ddof=1):
    "Slow move_std for unaccelerated dtype"
    return move_func(np.nanstd, a, window, min_count, axis=axis, ddof=ddof)


def slow_move_var(a, window, min_count=None, axis=-1, ddof=1):
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
    idx2: Any = list(idx1)
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
    idx1_ = [slice(None)] * a.ndim
    idx2_ = [slice(None)] * a.ndim
    idx3_ = [slice(None)] * a.ndim
    idx1_[axis] = slice(window, None)
    idx2_[axis] = slice(None, -window)
    idx3_[axis] = slice(None, window)
    idx1 = tuple(idx1_)
    idx2 = tuple(idx2_)
    idx3 = tuple(idx3_)
    nidx1 = n[idx1]
    nidx1 = nidx1 - n[idx2]
    idx = np.empty(a.shape, dtype=np.bool_)
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
        r: Any = np.empty(shape, dtype=a.dtype)
        r.fill(np.nan)
        if (r.ndim == 0) and (r.size == 1):
            r = np.nan
        return r
    indlast_ = [slice(None)] * ndim
    indlast_[axis] = slice(-1, None)
    indlast = tuple(indlast_)
    indlast2_: Any = [slice(None)] * ndim
    indlast2_[axis] = -1
    indlast2 = tuple(indlast2_)
    n = (~np.isnan(a)).sum(axis)
    a_indlast = a[indlast]
    g = (a_indlast > a).sum(axis)
    e = (a_indlast == a).sum(axis)
    r = (g + g + e - 1.0) / 2.0
    r = r / (n - 1.0)
    r = 2.0 * (r - 0.5)
    if ndim == 1:
        if n == 1:
            r = 0.0
        if np.isnan(a[indlast2]):  # elif?
            r = np.nan
    else:
        np.putmask(r, n == 1, 0)
        np.putmask(r, np.isnan(a[indlast2]), np.nan)
    return r
