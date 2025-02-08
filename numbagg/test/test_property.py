from __future__ import annotations

import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import numbagg
from numbagg import move_exp_nanmean

from .conftest import COMPARISONS

pytestmark = pytest.mark.nightly
pytestmark = pytest.mark.skip(
    reason="These need more work; in particular they overflow with very large values, but arguably in an acceptable way"
)


@given(
    numbagg_func=st.sampled_from([move_exp_nanmean]),
    array=hnp.arrays(
        dtype=hnp.floating_dtypes(), shape=hnp.array_shapes(min_dims=2, max_dims=2)
    ).filter(
        # Pandas doesn't handle inf values well
        lambda x: not np.isinf(x).any()
    ),
    alpha=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
)
@settings(deadline=500)
def test_move_exp_pandas_comparison(
    numbagg_func,
    array,
    alpha,
):
    kwargs = dict(alpha=alpha)

    if np.sum(np.isfinite(array) > 0) and np.nanmax(array) > 1e300:
        # We don't always handle overflows well
        return
    if alpha == 1 & ~np.isnan(array) > 0:
        # Pandas doesn't agree with us on arrays such as `[0, np.nan]`, see unit tests
        # for more details.
        return

    func = COMPARISONS[numbagg_func]["numbagg"](array, **kwargs)
    comp_func = COMPARISONS[numbagg_func]["pandas"](array, **kwargs)

    with np.errstate(invalid="ignore"):
        # Execute functions and capture exceptions if they occur
        try:
            expected = comp_func()
        except Exception as e:
            expected_exception: None | Exception = e
        else:
            expected_exception = None

        try:
            result = func()
        except Exception as e:
            result_exception: None | Exception = e
        else:
            result_exception = None

        # Check if both functions raised exceptions
        if expected_exception:
            if "Big-endian buffer not supported on little-endian compiler" in str(
                expected_exception
            ):
                # pandas doesn't support this but it's OK that we do
                return
            assert type(result_exception) is type(expected_exception)

            # If only one function raised an exception, the test should fail
            return  # Both raised exceptions, test passes
        else:
            assert result_exception is None

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

        assert result.dtype == expected.dtypes[0]


@given(
    numbagg_func=st.sampled_from(numbagg.MOVE_EXP_FUNCS),
    array=hnp.arrays(
        dtype=hnp.floating_dtypes(), shape=hnp.array_shapes(min_dims=1, max_dims=6)
    ),
    axis=st.one_of(st.integers(min_value=-6, max_value=6)),
    alpha=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
)
@settings(deadline=500)
def test_moving_exp_bigger_arrays_have_same_beginning(
    numbagg_func,
    array,
    alpha,
    axis,
):
    axis = axis % array.ndim

    if array.shape[axis] < 2:
        # Array too small
        return

    kwargs = dict(alpha=alpha, axis=axis)

    result = COMPARISONS[numbagg_func]["numbagg"](array, **kwargs)()
    sliced_array = np.take(array, indices=range(array.shape[axis] - 1), axis=axis)
    sliced_result = COMPARISONS[numbagg_func]["numbagg"](sliced_array, **kwargs)()

    result_sliced = np.take(result, indices=range(result.shape[axis] - 1), axis=axis)

    np.testing.assert_array_equal(sliced_result, result_sliced)


@given(
    numbagg_func=st.sampled_from(numbagg.MOVE_FUNCS),
    array=hnp.arrays(
        dtype=hnp.floating_dtypes(), shape=hnp.array_shapes(min_dims=1, max_dims=6)
    ),
    axis=st.one_of(st.integers(min_value=-6, max_value=6)),
    window=st.integers(min_value=1),
)
@settings(deadline=500)
def test_moving_bigger_arrays_have_same_beginning(
    numbagg_func,
    array,
    window,
    axis,
):
    axis = axis % array.ndim

    if array.shape[axis] < 2:
        # Array too small
        return
    if array.shape[axis] - 1 < window:
        # Array too small
        return

    kwargs = dict(window=window, axis=axis)

    result = COMPARISONS[numbagg_func]["numbagg"](array, **kwargs)()
    sliced_array = np.take(array, indices=range(array.shape[axis] - 1), axis=axis)
    sliced_result = COMPARISONS[numbagg_func]["numbagg"](sliced_array, **kwargs)()

    result_sliced = np.take(result, indices=range(result.shape[axis] - 1), axis=axis)

    np.testing.assert_array_equal(sliced_result, result_sliced)
