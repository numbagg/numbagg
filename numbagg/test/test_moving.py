import warnings
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from numbagg import (
    MOVE_FUNCS,
    move_corr,
    move_corrmatrix,
    move_cov,
    move_covmatrix,
    move_mean,
    move_std,
    move_sum,
    move_var,
)

from .conftest import COMPARISONS
from .util import array_order, arrays

# Matrix functions return `(..., obs, vars, vars)` rather than the input shape, so
# they're compared against pandas in `test_move_matrix_pandas_comp` instead.
MOVE_MATRIX_FUNCS = [move_corrmatrix, move_covmatrix]


@pytest.fixture(scope="function")
def rs():
    return np.random.RandomState(0)


@pytest.mark.parametrize(
    "func",
    [f for f in MOVE_FUNCS if f not in MOVE_MATRIX_FUNCS],
)
@pytest.mark.parametrize("shape", [(3, 500)], indirect=True)
@pytest.mark.parametrize("window", [10, 50])
@pytest.mark.parametrize("min_count", [None, 0, 1, 3, "window"])
def test_move_pandas_comp(array, func, window, min_count):
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


@pytest.mark.parametrize("func", MOVE_MATRIX_FUNCS, indirect=True)
@pytest.mark.parametrize(
    "shape", [(5, 100)], indirect=True
)  # (vars, obs) benchmark convention
@pytest.mark.parametrize("window", [10, 30])
@pytest.mark.parametrize("min_count", [None, "window"])
def test_move_matrix_pandas_comp(array, func, window, min_count):
    """Test matrix functions against pandas with various parameters."""
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


def _exact_rolling_moments(a, b, window, min_count):
    """Variance, covariance, and correlation of represented values, exactly."""
    variance = np.full(len(a), np.nan)
    covariance = np.full(len(a), np.nan)
    correlation = np.full(len(a), np.nan)
    for i in range(len(a)):
        start = max(0, i - window + 1)
        values = [
            Fraction(float(value)) for value in a[start : i + 1] if not np.isnan(value)
        ]
        pairs = [
            (Fraction(float(value_a)), Fraction(float(value_b)))
            for value_a, value_b in zip(a[start : i + 1], b[start : i + 1])
            if not (np.isnan(value_a) or np.isnan(value_b))
        ]
        if len(values) >= max(min_count, 2):
            mean = sum(values) / len(values)
            variance[i] = float(
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            )
        if len(pairs) >= max(min_count, 2):
            mean_a = sum(value_a for value_a, _ in pairs) / len(pairs)
            mean_b = sum(value_b for _, value_b in pairs) / len(pairs)
            denominator = len(pairs) - 1
            var_a = sum((value_a - mean_a) ** 2 for value_a, _ in pairs) / denominator
            var_b = sum((value_b - mean_b) ** 2 for _, value_b in pairs) / denominator
            cov = (
                sum(
                    (value_a - mean_a) * (value_b - mean_b)
                    for value_a, value_b in pairs
                )
                / denominator
            )
            covariance[i] = float(cov)
            if var_a > 0 and var_b > 0:
                correlation[i] = float(cov) / np.sqrt(float(var_a * var_b))
    return variance, covariance, correlation


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_move_moments_persistent_offset(dtype):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(80).astype(dtype)
    b = (0.4 * a + rng.standard_normal(80)).astype(dtype)
    offset = dtype(1e4 if dtype is np.float32 else 1e8)
    a += offset
    b += offset
    window = 12
    min_count = 5
    expected_var, expected_cov, expected_corr = _exact_rolling_moments(
        a, b, window, min_count
    )
    # These inputs have unit spread. Use the same tolerance as both a relative and
    # absolute error scale so a covariance that happens to be near zero is not
    # divided by itself; covariance accuracy is defined by marginal spread.
    rtol = 2e-6 if dtype is np.float32 else 2e-12

    assert_allclose(
        move_var(a, window=window, min_count=min_count),
        expected_var,
        rtol=rtol,
        atol=rtol,
    )
    assert_allclose(
        move_std(a, window=window, min_count=min_count),
        np.sqrt(expected_var),
        rtol=rtol,
        atol=rtol,
    )
    assert_allclose(
        move_cov(a, b, window=window, min_count=min_count),
        expected_cov,
        rtol=rtol,
        atol=rtol,
    )
    assert_allclose(
        move_corr(a, b, window=window, min_count=min_count),
        expected_corr,
        rtol=rtol,
        atol=rtol,
    )


@pytest.mark.parametrize("event", ["leading-outlier", "level-change"])
def test_move_moments_recover_from_scale_change(event):
    rng = np.random.default_rng(1)
    a = rng.standard_normal(80)
    b = 0.4 * a + rng.standard_normal(80)
    window = 10
    min_count = 4
    if event == "leading-outlier":
        a[0] = 1e12
        b[0] = -1e12
    else:
        a[30:] += 1e8
        b[30:] += 1e8

    expected_var, expected_cov, expected_corr = _exact_rolling_moments(
        a, b, window, min_count
    )
    actual = (
        move_var(a, window=window, min_count=min_count),
        move_std(a, window=window, min_count=min_count),
        move_cov(a, b, window=window, min_count=min_count),
        move_corr(a, b, window=window, min_count=min_count),
    )
    expected = (expected_var, np.sqrt(expected_var), expected_cov, expected_corr)
    skip = window if event == "leading-outlier" else 0
    for result, reference in zip(actual, expected):
        assert_allclose(result[skip:], reference[skip:], rtol=2e-12, atol=1e-14)


def test_move_moments_reanchor_after_repeated_float32_outliers():
    rng = np.random.default_rng(233)
    a = rng.standard_normal(50).astype(np.float32)
    b = (0.4 * a + rng.standard_normal(50)).astype(np.float32)
    a[[0, 3, 19]] = 1e6
    b[[0, 3, 19]] = -1e6
    window = 4
    min_count = 3
    expected_var, expected_cov, expected_corr = _exact_rolling_moments(
        a, b, window, min_count
    )

    actual = (
        move_var(a, window=window, min_count=min_count),
        move_std(a, window=window, min_count=min_count),
        move_cov(a, b, window=window, min_count=min_count),
        move_corr(a, b, window=window, min_count=min_count),
    )
    expected = (expected_var, np.sqrt(expected_var), expected_cov, expected_corr)
    for result, reference in zip(actual, expected):
        assert_allclose(result, reference, rtol=2e-6, atol=2e-6)


def test_move_moments_recover_when_outlier_expires_below_min_count():
    close_float32 = np.array([-0.99379987, -1.0016286], dtype=np.float32)
    close_var, _, close_corr = _exact_rolling_moments(
        close_float32, close_float32[::-1].copy(), 2, 2
    )
    assert_allclose(move_var(close_float32, window=2), close_var, rtol=2e-6)
    assert_allclose(
        move_corr(close_float32, close_float32[::-1].copy(), window=2),
        close_corr,
        rtol=2e-6,
    )

    # With a two-value window, the destructive removal happens while only two
    # values are in state. It still has to trigger recovery.
    variance_data = np.array([1e14, 0.0, 1.0, 1.1, 0.9])
    expected_var, _, _ = _exact_rolling_moments(variance_data, variance_data, 2, 2)
    assert_allclose(move_var(variance_data, window=2), expected_var, rtol=2e-12)
    assert_allclose(
        move_std(variance_data, window=2), np.sqrt(expected_var), rtol=2e-12
    )

    # The outlier leaves while only one valid pair remains, so that output is
    # NaN. Recovery must remain latched for the next emit-capable window.
    a = np.array([1e14, np.nan, 0.483, np.nan, 0.023])
    b = np.array([-1e14, np.nan, 0.980, np.nan, 0.977])
    _, expected_cov, expected_corr = _exact_rolling_moments(a, b, 3, 2)
    assert_allclose(move_cov(a, b, window=3, min_count=2), expected_cov, rtol=2e-12)
    assert_allclose(move_corr(a, b, window=3, min_count=2), expected_corr, rtol=2e-12)


def test_move_moments_slow_drift_accuracy_floor():
    rng = np.random.default_rng(4)
    size = 100_000
    window = 52
    min_count = 25
    drift = np.linspace(0.0, 1e8, size)
    a = rng.standard_normal(size) + drift
    b = 0.4 * a + rng.standard_normal(size)
    indices = np.unique(np.linspace(window - 1, size - 1, 64, dtype=np.int64))
    actual_var = move_var(a, window=window, min_count=min_count)
    actual_std = move_std(a, window=window, min_count=min_count)
    actual_cov = move_cov(a, b, window=window, min_count=min_count)
    actual_corr = move_corr(a, b, window=window, min_count=min_count)

    for i in indices:
        window_a = a[i - window + 1 : i + 1]
        window_b = b[i - window + 1 : i + 1]
        expected_var, expected_cov, expected_corr = _exact_rolling_moments(
            window_a, window_b, window, min_count
        )
        actual = (
            actual_var[i],
            actual_std[i],
            actual_cov[i],
            actual_corr[i],
        )
        expected = (
            expected_var[-1],
            np.sqrt(expected_var[-1]),
            expected_cov[-1],
            expected_corr[-1],
        )
        assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_move_moments_do_not_look_ahead():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(50) + 1e8
    b = rng.standard_normal(50) + 1e8
    window = 8
    min_count = 3

    for function, arrays_ in (
        (move_var, (a,)),
        (move_std, (a,)),
        (move_cov, (a, b)),
        (move_corr, (a, b)),
    ):
        callable_: Any = function
        complete = callable_(*arrays_, window=window, min_count=min_count)
        for length in (min_count, window, window + 1, len(a) - 1):
            prefix = callable_(
                *(array[:length] for array in arrays_),
                window=min(window, length),
                min_count=min_count,
            )
            np.testing.assert_array_equal(
                prefix,
                complete[:length],
                err_msg=f"{function.__name__}, prefix={length}",
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_move_moment_invariants_on_degenerate_windows(dtype):
    representable_step = 1e-2 if dtype is np.float32 else 1e-7
    pattern = np.resize(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * representable_step, 100)
    a = (1e4 if dtype is np.float32 else 1e8) + pattern
    a = a.astype(dtype)
    constant = np.full(100, a[0], dtype=dtype)

    variance = move_var(a, window=9, min_count=2)
    correlation = move_corr(a, a[::-1].copy(), window=9, min_count=2)
    finite_correlation = correlation[np.isfinite(correlation)]
    assert np.all(variance[np.isfinite(variance)] >= 0.0)
    assert np.all(np.abs(finite_correlation) <= 1.0)
    assert_allclose(move_var(constant, window=9)[8:], 0.0)
    assert_allclose(move_std(constant, window=9)[8:], 0.0)
    assert_allclose(move_cov(constant, constant, window=9)[8:], 0.0)
    assert np.all(np.isnan(move_corr(constant, constant, window=9)[8:]))


def test_move_covariance_pairwise_nan_reference():
    a = np.array([1e8, np.nan, 1e8 + 1, 1e8 + 2, np.nan, 1e8 + 3])
    b = np.array([1e8 + 3, 1e8 + 2, np.nan, 1e8 + 1, 1e8, 1e8 - 1])
    expected_var, expected_cov, expected_corr = _exact_rolling_moments(a, b, 5, 2)

    assert_allclose(move_var(a, window=5, min_count=2), expected_var, rtol=2e-12)
    assert_allclose(move_cov(a, b, window=5, min_count=2), expected_cov, rtol=2e-12)
    assert_allclose(move_corr(a, b, window=5, min_count=2), expected_corr, rtol=2e-12)


def test_move_pairwise_moments_recover_from_constant_prefix():
    """The first emitted window can already contain an old and a new regime."""
    rng = np.random.default_rng(3501)
    a = rng.standard_normal(69)
    b = 0.4 * a + rng.standard_normal(69)
    a[:23] = 1e8
    b[:23] = -1e8
    window = 36
    min_count = 32
    _, expected_cov, expected_corr = _exact_rolling_moments(a, b, window, min_count)

    assert_allclose(
        move_cov(a, b, window=window, min_count=min_count),
        expected_cov,
        rtol=2e-12,
        atol=2e-12,
    )
    assert_allclose(
        move_corr(a, b, window=window, min_count=min_count),
        expected_corr,
        rtol=2e-12,
        atol=2e-12,
    )


def test_move_variance_recovers_after_late_scale_expansion_and_contraction():
    """A small initial min_count must not freeze an unrepresentative scale."""
    rng = np.random.default_rng(3701)
    a = np.empty(119, dtype=np.float32)
    a[:59] = (rng.standard_normal(59) * 1e4).astype(np.float32)
    a[59:] = (1e4 + rng.standard_normal(60)).astype(np.float32)
    window = 15
    min_count = 3
    expected_var, _, _ = _exact_rolling_moments(a, a, window, min_count)

    assert_allclose(
        move_var(a, window=window, min_count=min_count),
        expected_var,
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("seed", [2901, 2902, 2903])
def test_move_pairwise_moments_same_magnitude_contraction(seed):
    """A spread can collapse without observations growing in magnitude."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(100) * 1e4
    b = 0.4 * a + rng.standard_normal(100) * 1e4
    a[50:] = 1e4 + rng.standard_normal(50)
    b[50:] = -1e4 + rng.standard_normal(50)
    window = 20
    min_count = 8
    expected_var_a, expected_cov, expected_corr = _exact_rolling_moments(
        a, b, window, min_count
    )
    expected_var_b, _, _ = _exact_rolling_moments(b, a, window, min_count)

    actual_cov = move_cov(a, b, window=window, min_count=min_count)
    pair_scale = np.sqrt(expected_var_a * expected_var_b)
    finite = np.isfinite(expected_cov)
    assert np.all(
        np.abs(actual_cov[finite] - expected_cov[finite]) < 3e-7 * pair_scale[finite]
    )
    assert_allclose(
        move_corr(a, b, window=window, min_count=min_count),
        expected_corr,
        rtol=3e-7,
        atol=3e-7,
    )


def test_move_pairwise_near_constant_float32_accuracy_floor():
    """Pin the selected pandas-compatible speed/accuracy tradeoff."""
    a = np.array([9999.9970703125, 9999.998046875, 9999.998046875], np.float32)
    b = np.array([10000.0029296875, 9999.9990234375, 10000.001953125], np.float32)
    _, expected_cov, expected_corr = _exact_rolling_moments(a, b, 3, 3)

    # Promoted raw products match pandas covariance on this worst generated case.
    # NumPy's two-pass result is more accurate; keep the discrepancy below 1% of
    # marginal spread and 0.003 absolute correlation rather than pretending the
    # fast path is exact.
    actual_cov = move_cov(a, b, window=3)
    actual_corr = move_corr(a, b, window=3)
    pair_scale = abs(expected_cov[-1] / expected_corr[-1])
    assert abs(actual_cov[-1] - expected_cov[-1]) < 0.01 * pair_scale
    assert abs(actual_corr[-1] - expected_corr[-1]) < 0.003


def slow_move_mean(a, window, min_count=None, axis=-1):
    "Slow move_mean for unaccelerated dtype"
    return move_func(np.nanmean, a, window, min_count, axis=axis)


def functions():
    yield move_mean, slow_move_mean


@pytest.mark.parametrize("func,func0", list(functions()))
def test_numerical_results_identical(func, func0):
    "Test that the numbagg function matches a slow reference implementation."
    fmt = (
        "\nfunc %s | window %d | min_count %s | input %s (%s) | shape %s | "
        "axis %s | order %s\n"
    )
    fmt += "\nInput array:\n%s\n"
    func_name = func.__name__
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


# magic utility functions ---------------------------------------------------


def move_func(func, a, window, min_count=None, axis=-1, **kwargs):
    "Generic moving window function implemented with a python loop."
    a = np.asarray(a)
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
