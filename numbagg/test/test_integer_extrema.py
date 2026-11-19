import numpy as np
import pytest
from numpy.testing import assert_array_equal

import numbagg as nb


@pytest.mark.parametrize("function", ["nanmin", "nanmax", "nanargmin", "nanargmax"])
@pytest.mark.parametrize(
    "values",
    [
        [2**53 + 1, 2**53],
        [2**53, 2**53 + 1],
        [-(2**53) - 1, -(2**53)],
        [-(2**53), -(2**53) - 1],
        [np.iinfo(np.int64).max - 1, np.iinfo(np.int64).max],
        [np.iinfo(np.int64).min + 1, np.iinfo(np.int64).min],
    ],
)
@pytest.mark.parametrize("axis", [None, 0, -1])
def test_integer_extrema(function, values, axis):
    a = np.array([values, values[::-1]], dtype=np.int64)
    result = getattr(nb, function)(a, axis=axis)
    expected = getattr(np, function)(a, axis=axis)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "function", ["group_nanmin", "group_nanmax", "group_nanargmin", "group_nanargmax"]
)
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("sign", [-1, 1])
def test_group_integer_extrema(function, dtype, sign):
    start = 2**53 if dtype == np.int64 else 2**24
    values = np.array([start + 1, start, start + 3, start + 2], dtype=dtype)
    values *= sign
    labels = np.array([0, 0, 1, 1])
    expected = []
    for i in range(2):
        indices = np.flatnonzero(labels == i)
        result = getattr(np, function.removeprefix("group_"))(values[indices])
        expected.append(indices[result] if "arg" in function else result)
    assert_array_equal(getattr(nb, function)(values, labels), expected)
