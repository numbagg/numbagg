import numpy as np
from numpy.typing import NDArray

type FloatScalar = np.float64 | np.float32
type FloatArray = NDArray[np.float64] | NDArray[np.float32]

def move_exp_nancount[T: FloatArray](
    arr: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
) -> T: ...
def move_exp_nanmean[T: FloatArray](
    arr: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
) -> T:
    """
    Exponentially weighted moving mean
    """
    ...

def move_exp_nansum[T: FloatArray](
    arr: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
) -> T: ...
def move_exp_nanvar[T: FloatArray](
    arr: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
) -> T: ...
def move_exp_nanstd[T: FloatArray](
    arr: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
) -> T:
    """
    Calculates the exponentially decayed standard deviation.

    Note that technically the unbiased weighted standard deviation is not exactly the
    same as the square root of the unbiased weighted variance, since the bias is
    concave. But it's close, and it's what pandas does.

    (If anyone knows the math well and wants to take a pass at improving it,
    contributions are welcome.)
    """
    ...

def move_exp_nancov[T: FloatScalar](
    a1: NDArray[T],
    a2: NDArray[T],
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
) -> NDArray[T]: ...
def move_exp_nancorr[T: FloatScalar](
    a1: NDArray[T],
    a2: NDArray[T],
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
) -> NDArray[T]: ...
