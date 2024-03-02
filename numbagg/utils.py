import numpy as np


def move_axes(arr: np.ndarray, axes: tuple[int, ...]):
    """
    Move & reshape a tuple of axes to an array's final axis, handling zero-length axes.
    """
    # Normalize axes values to positive
    axes = tuple(a % arr.ndim for a in axes)

    # Move specified axes to the end
    moved_arr = np.moveaxis(arr, axes, range(arr.ndim - len(axes), arr.ndim))
    final_axis = 0 if 0 in arr.shape else -1

    # Calculate the new shape
    new_shape = moved_arr.shape[: -len(axes)] + (final_axis,)
    return moved_arr.reshape(new_shape)
