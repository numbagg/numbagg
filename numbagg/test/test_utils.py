import numpy as np
import pytest

from numbagg.utils import move_axes


def test_move_axes_rejects_out_of_bounds():
    """move_axes should raise AxisError for out-of-bounds axis indices."""
    arr = np.ones((3, 4, 5))

    # Positive out-of-bounds
    with pytest.raises(np.exceptions.AxisError):
        move_axes(arr, (3,))
    with pytest.raises(np.exceptions.AxisError):
        move_axes(arr, (5,))

    # Negative out-of-bounds
    with pytest.raises(np.exceptions.AxisError):
        move_axes(arr, (-4,))
    with pytest.raises(np.exceptions.AxisError):
        move_axes(arr, (-10,))


def test_move_axes():
    # Test 1: Simple ndmove with no zero dimensions
    arr1 = np.ones((3, 4, 5))
    assert move_axes(arr1, (0,)).shape == (4, 5, 3)

    # Test 2: Move with a zero dimension not in the specified axes
    arr2 = np.ones((3, 0, 5))
    assert move_axes(arr2, (0, 2)).shape == (0, 0)

    # Test 3: Move where the specified axis is zero-length
    arr3 = np.ones((3, 3, 0))
    assert move_axes(arr3, (2,)).shape == (3, 3, 0)

    # Test 4: Complex ndmove with multiple axes
    arr4 = np.ones((3, 4, 5, 6))
    assert move_axes(arr4, (0, 2)).shape == (4, 6, 15)

    # Test 5: All axes are zero-length, some specified
    arr5 = np.empty((0, 0, 5))
    assert move_axes(arr5, (0,)).shape == (0, 5, 0)

    # Test 6: Mixed zero and non-zero dimensions specified
    arr6 = np.empty((5, 0, 7, 0))
    assert move_axes(arr6, (1, 3)).shape == (5, 7, 0)
