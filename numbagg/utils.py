import operator
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np
from numba.core.types import Type
from numpy.typing import NDArray

Targets = Literal["cpu", "parallel"]
NumbaTypes: TypeAlias = tuple[Type, ...]
FloatScalar: TypeAlias = np.float64 | np.float32
IntScalar: TypeAlias = np.int64 | np.int32
NumericScalar: TypeAlias = FloatScalar | IntScalar
IntArray: TypeAlias = NDArray[np.int64] | NDArray[np.int32]
FloatArray: TypeAlias = NDArray[np.float64] | NDArray[np.float32]
NumericArray: TypeAlias = IntArray | FloatArray
GenericArray: TypeAlias = NumericArray | NDArray[np.bool_]

# Implementation-side equivalents of the aliases above, spelled as constrained
# TypeVars rather than `bound=<union>`: ty reports false positives on any
# indexing or member access through a TypeVar whose bound is a union
# (https://github.com/astral-sh/ty/issues/2585). The public `.pyi` stubs keep
# the `bound=` form, so the types callers see are unchanged.
FloatArrayT = TypeVar("FloatArrayT", NDArray[np.float64], NDArray[np.float32])
NumericArrayT = TypeVar(
    "NumericArrayT",
    NDArray[np.int64],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float32],
)

T = TypeVar("T", bound=NumericScalar)


def normalize_axis(axis: Any) -> tuple[int, ...]:
    """
    Normalize any axis spelling numpy accepts into a tuple of ints.

    numpy takes `SupportsIndex | Sequence[SupportsIndex]`, so a Python `int`, a numpy
    integer and a 0-d array are all single axes, while a list, tuple, range or 1-d
    array is a set of them — dask passes a list.
    """
    try:
        return (operator.index(axis),)
    except TypeError:
        return tuple(operator.index(ax) for ax in axis)


def move_axes(arr: NDArray[T], axes: tuple[int, ...]):
    """
    Move & reshape a tuple of axes to an array's final axis, handling zero-length axes.
    """
    if not axes:
        # `axis=()` reduces nothing, and numpy expresses that as a reduction over a
        # fresh length-1 axis: `np.nansum(a, axis=())` returns `a` with NaNs filled,
        # `np.nanvar(a, axis=())` returns zeros. Appending that axis reproduces it.
        # The general path below can't: `shape[:-len(axes)]` is `shape[:0]` when
        # `axes` is empty, so it would flatten the whole array into the reduced axis
        # and return a full reduction instead.
        return arr.reshape(arr.shape + (1,))

    # np.moveaxis handles negative indices and raises AxisError for out-of-bounds

    # Move specified axes to the end
    moved_arr = np.moveaxis(arr, axes, range(arr.ndim - len(axes), arr.ndim))
    final_axis: Literal[0, -1] = 0 if 0 in arr.shape else -1

    # Calculate the new shape
    new_shape = moved_arr.shape[: -len(axes)] + (final_axis,)
    return moved_arr.reshape(new_shape)
