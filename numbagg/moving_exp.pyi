from typing import TypeVar

from numbagg.utils import FloatArray

T = TypeVar("T", bound=FloatArray)

def move_exp_nancount(
    arr: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int = -1,
) -> T: ...
def move_exp_nanmean(
    arr: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int = -1,
) -> T: ...
def move_exp_nansum(
    arr: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int = -1,
) -> T: ...
def move_exp_nanvar(
    arr: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int = -1,
) -> T: ...
def move_exp_nanstd(
    arr: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int = -1,
) -> T: ...
def move_exp_nancov(
    a1: T,
    a2: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int = -1,
) -> T: ...
def move_exp_nancorr(
    a1: T,
    a2: T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int = -1,
) -> T: ...
