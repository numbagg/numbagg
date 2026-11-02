import numpy as np
import pytest

import numbagg as nb


@pytest.mark.parametrize("func", [nb.ffill, nb.bfill])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.int32, np.uint32])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("limit", [None, 0, 1])
def test_fill_preserves_integer_values(func, dtype, axis, limit):
    bounds = np.iinfo(dtype)
    values = [bounds.min, bounds.min + 1, bounds.max - 1, bounds.max]
    if np.dtype(dtype).itemsize == 8:
        values += [2**53 + 1, 2**53 + 3]
    array = np.array([values, values[::-1]], dtype=dtype)
    array.setflags(write=False)
    result = func(array, axis=axis, limit=limit)
    assert result.dtype == array.dtype
    np.testing.assert_array_equal(result, array)


@pytest.mark.parametrize("func", [nb.ffill, nb.bfill])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_fill_preserves_strided_integer_values(func, dtype):
    array = np.arange(12, dtype=dtype).reshape(3, 4) + 2**53 + 1
    view = array.T[::-1, ::-1]
    np.testing.assert_array_equal(func(view), view)
