from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray


class Type(Protocol):
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...


class Array(Type):
    def __init__(self, dtype: Any, ndim: Any, layout: Any) -> None: ...


class Integer(Type):
    def __getitem__(self, args: Any) -> Array: ...


class Float(Type):
    def __getitem__(self, args: Any) -> Array: ...


Targets = Literal["cpu", "parallel"]
type NumbaTypes = tuple[Type, ...]
type FloatScalar = np.float64 | np.float32
type IntScalar = np.int64 | np.int32
type NumericScalar = FloatScalar | IntScalar
type IntArray = NDArray[np.int64] | NDArray[np.int32]
type FloatArray = NDArray[np.float64] | NDArray[np.float32]
type NumericArray = IntArray | FloatArray
type GenericArray = NumericArray | NDArray[np.bool_]


def move_axes[T: NumericScalar](arr: NDArray[T], axes: tuple[int, ...]) -> NDArray[T]:
    """
    Move & reshape a tuple of axes to an array's final axis, handling zero-length axes.
    """
    # Normalize axes values to positive
    axes = tuple(a % arr.ndim for a in axes)

    # Move specified axes to the end
    moved_arr: NDArray[T] = np.moveaxis(
        arr, axes, range(arr.ndim - len(axes), arr.ndim)
    )
    final_axis: Literal[0, -1] = 0 if 0 in arr.shape else -1

    # Calculate the new shape
    new_shape = moved_arr.shape[: -len(axes)] + (final_axis,)
    return moved_arr.reshape(new_shape)
