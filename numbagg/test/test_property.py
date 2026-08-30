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

# Numba compiles each gufunc on its first call, and numbagg leaves numba's on-disk
# cache off by default (`NUMBAGG_CACHE`), so every process pays that cost afresh.
# No fixed per-example deadline survives it on a cold runner.
no_deadline = settings(deadline=None)


@pytest.mark.skip(
    reason="numbagg and pandas disagree by more than the tolerance under catastrophic "
    "cancellation — move_exp_nanmean([[9.00738e10, 0.0]], alpha=1-eps) on float32 gives "
    "1.00002e-05 against pandas' 2.00004e-05"
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
@no_deadline
def test_move_exp_pandas_comparison(
    numbagg_func,
    array,
    alpha,
):
    kwargs = dict(alpha=alpha)

    # Compare as a Python float: `array.max() > 1e300` would cast 1e300 down to
    # the array's dtype, which itself overflows for e.g. float16.
    finite = array[np.isfinite(array)]
    if finite.size and float(np.abs(finite).max()) > 1e300:
        # We don't always handle overflows well
        return
    if alpha == 1 and (~np.isnan(array)).any():
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

        if expected_exception:
            if "Big-endian buffer not supported on little-endian compiler" in str(
                expected_exception
            ):
                # pandas doesn't support this but it's OK that we do
                return
            # If only one function raised an exception, the test should fail
            assert type(result_exception) is type(expected_exception)
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
    axis=st.integers(min_value=-6, max_value=6),
    alpha=st.floats(min_value=0.0, max_value=1.0, exclude_min=True),
)
@no_deadline
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

    # Very large values overflow the accumulators; that's acceptable here, since both
    # sides of the comparison overflow identically and we're only asserting that the
    # shorter array's result is a prefix of the longer one's.
    with np.errstate(over="ignore", invalid="ignore"):
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
    axis=st.integers(min_value=-6, max_value=6),
    window=st.integers(min_value=1),
)
@no_deadline
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

    # See the note on overflow in the test above.
    with np.errstate(over="ignore", invalid="ignore"):
        result = COMPARISONS[numbagg_func]["numbagg"](array, **kwargs)()
        sliced_array = np.take(array, indices=range(array.shape[axis] - 1), axis=axis)
        sliced_result = COMPARISONS[numbagg_func]["numbagg"](sliced_array, **kwargs)()

    result_sliced = np.take(result, indices=range(result.shape[axis] - 1), axis=axis)

    np.testing.assert_array_equal(sliced_result, result_sliced)
