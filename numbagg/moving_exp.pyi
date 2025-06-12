from numpy.typing import NDArray

from numbagg.utils import FloatArray, FloatScalar

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
