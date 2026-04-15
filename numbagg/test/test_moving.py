import warnings
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from numbagg import (
    MOVE_FUNCS,
    move_corrmatrix,
    move_covmatrix,
    move_mean,
    move_sum,
)

from .conftest import COMPARISONS
from .util import array_order, arrays


@pytest.fixture(scope="function")
def rs():
    return np.random.RandomState(0)


@pytest.mark.parametrize(
    "func",
    MOVE_FUNCS,
)
@pytest.mark.parametrize("shape", [(3, 500)], indirect=True)
@pytest.mark.parametrize("window", [10, 50])
@pytest.mark.parametrize("min_count", [None, 0, 1, 3, "window"])
def test_move_pandas_comp(array, func, window, min_count):
    if func.__name__ in ["move_corrmatrix", "move_covmatrix"]:
        pytest.skip(
            "Matrix functions tested separately in test_move_matrix_pandas_comp"
        )
    c = COMPARISONS[func]

    if min_count == "window":
        min_count = window

    result = c["numbagg"](array, window=window, min_count=min_count)()
    expected_pandas = c["pandas"](array, window=window, min_count=min_count)()

    assert_allclose(result, expected_pandas)

    if c.get("bottleneck"):
        if min_count == 0:
            pytest.skip("bottleneck doesn't support min_count=0")
        expected_bottleneck = c["bottleneck"](
            array, window=window, min_count=min_count
        )()
        assert_allclose(result, expected_bottleneck)


@pytest.mark.parametrize("func", [move_corrmatrix, move_covmatrix], indirect=True)
@pytest.mark.parametrize(
    "shape", [(5, 100)], indirect=True
)  # (vars, obs) benchmark convention
@pytest.mark.parametrize("window", [10, 30])
@pytest.mark.parametrize("min_count", [None, "window"])
def test_move_matrix_pandas_comp(array, func, window, min_count):
    """Test matrix functions against pandas with various parameters."""
    if array.ndim < 2:
        pytest.skip("Matrix functions require at least 2D input")

    c = COMPARISONS[func]

    if min_count == "window":
        min_count = window

    # Get numbagg result
    result = c["numbagg"](array, window=window, min_count=min_count)()

    # Get pandas result - need to handle the different output format
    pandas_callable = c["pandas"](array, window=window, min_count=min_count)
    pandas_result = pandas_callable()

    # Convert pandas MultiIndex DataFrame to 3D array for comparison
    # Result shape is (..., obs, vars, vars), so we can infer dimensions from result
    n_obs = result.shape[-3]  # obs dimension
    n_vars = result.shape[-2]  # vars dimension (should equal result.shape[-1])
    expected_pandas = np.full((n_obs, n_vars, n_vars), np.nan)

    # Only include windows where we have at least min_count observations
    actual_min_count = min_count if min_count is not None else window
    for t in range(n_obs):
        # Check if we have enough observations in this window
        window_size = min(t + 1, window)
        if (
            window_size >= actual_min_count
            and t in pandas_result.index.get_level_values(0)
        ):
            expected_pandas[t] = pandas_result.loc[t].values

    assert_allclose(result, expected_pandas)


@pytest.mark.parametrize("func_name", ["move_corrmatrix", "move_covmatrix"])
@pytest.mark.parametrize("window", [5])
@pytest.mark.parametrize("min_count", [1, 2, 3, 4, 5])
def test_move_matrix_pandas_min_count_simple(func_name, window, min_count):
    """Test matrix functions against pandas with different min_count values."""
    # Create test array directly in benchmark format (vars, obs)
    rs = np.random.RandomState(0)
    array = rs.rand(4, 15)  # (vars=4, obs=15)

    # Get the function
    func = move_corrmatrix if func_name == "move_corrmatrix" else move_covmatrix

    # Get comparisons
    c = COMPARISONS[func]

    # Get numbagg result
    result = c["numbagg"](array, window=window, min_count=min_count)()

    # Get pandas result and convert to our format
    pandas_callable = c["pandas"](array, window=window, min_count=min_count)
    pandas_result = pandas_callable()

    # Convert pandas MultiIndex DataFrame to 3D array
    # Result shape is (..., obs, vars, vars), so we can infer dimensions from result
    n_obs = result.shape[-3]  # obs dimension
    n_vars = result.shape[-2]  # vars dimension (should equal result.shape[-1])
    expected_pandas = np.full((n_obs, n_vars, n_vars), np.nan)

    for t in range(n_obs):
        if t in pandas_result.index.get_level_values(0):
            expected_pandas[t] = pandas_result.loc[t].values

    assert_allclose(result, expected_pandas)


@pytest.mark.parametrize("shape", [(5, 20)], indirect=True)
@pytest.mark.parametrize("window", [5, 10])
@pytest.mark.parametrize("min_count", [1, 3, 5, 10])
def test_move_matrix_min_count(array, window, min_count):
    """Test that matrix functions handle min_count correctly."""
    # Transpose array for new (obs, vars) convention
    array_T = array.T

    # Test correlation matrix
    result_corr = move_corrmatrix(array_T, window=window, min_count=min_count)

    # Test covariance matrix
    result_cov = move_covmatrix(array_T, window=window, min_count=min_count)

    # Check that results are NaN where we don't have enough observations
    n_obs = array_T.shape[0]  # obs dimension in (obs, vars) format

    for t in range(n_obs):
        window_size = min(t + 1, window)

        if window_size < min_count:
            # Should be all NaN when we don't have enough observations
            assert np.all(np.isnan(result_corr[t])), (
                f"Expected NaN at position {t} for correlation"
            )
            assert np.all(np.isnan(result_cov[t])), (
                f"Expected NaN at position {t} for covariance"
            )
        else:
            # Check correlation matrix properties when we have enough data
            # Diagonal should be 1 for correlation (where not NaN)
            diag_corr = np.diag(result_corr[t])
            valid_diag = ~np.isnan(diag_corr)
            if np.any(valid_diag):
                assert_allclose(diag_corr[valid_diag], 1.0, rtol=1e-7)


@pytest.mark.parametrize("shape", [(3, 500)], indirect=True)
def test_move_mean_window(array):
    with pytest.raises(TypeError):
        move_mean(array, window=0.5)  # type: ignore
    with pytest.raises(ValueError):
        move_mean(array, window=-1)
    with pytest.raises(ValueError):
        move_mean(array, window=array.shape[-1] + 1)
    with pytest.raises(ValueError):
        move_mean(array, window=1, min_count=-1)


def test_numerical_issues_float32_move_mean_1(rs):
    arr = (rs.random(1000) * 1e13).astype(np.float32)
    result = move_mean(arr, window=1)
    assert_allclose(result, arr)


def test_numerical_issues_float32_move_sum_100(rs):
    # Does running over a repeated array accumulate any errors, compared to just running
    # over one of the tiles?
    arr = np.tile((rs.rand(10) * 1e13).astype(np.float32), 100)
    result = move_sum(arr, window=10)
    expected = np.sum(arr[:10])
    assert result[-1] == expected, result[-1] - expected


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
        if a.size >= 1_000:
            continue
        axes = range(-1, a.ndim)
        for axis in axes:
            windows = range(1, a.shape[axis])
            for window in windows:
                min_counts = list(range(1, window + 1)) + [None]
                for min_count in min_counts:
                    actual = func(a, window=window, min_count=min_count, axis=axis)
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


def slow_move_mean(a, window, min_count=None, axis=-1):
    "Slow move_mean for unaccelerated dtype"
    return move_func(np.nanmean, a, window, min_count, axis=axis)


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
    idx2: list[Any] = list(idx1)
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
