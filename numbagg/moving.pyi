from typing import overload

import numpy as np
from numpy.typing import NDArray

type FloatScalar = np.float64 | np.float32
type FloatArray = NDArray[np.float64] | NDArray[np.float32]

def move_mean[T: FloatArray](
    arr: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
def move_sum[T: FloatArray](
    arr: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
def move_std[T: FloatArray](
    arr: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
def move_var[T: FloatArray](
    arr: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
@overload
def move_cov(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> NDArray[np.float64]: ...
@overload
def move_cov(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> NDArray[np.float32]: ...
@overload
def move_corr(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> NDArray[np.float64]: ...
@overload
def move_corr(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> NDArray[np.float32]: ...
