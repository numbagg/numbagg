from typing import TypeVar

from numbagg.utils import FloatArray

_T = TypeVar("_T", bound=FloatArray)

def move_exp_nancount(
    arr: _T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_exp_nanmean(
    arr: _T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_exp_nansum(
    arr: _T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_exp_nanvar(
    arr: _T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_exp_nanstd(
    arr: _T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_exp_nancov(
    a1: _T,
    a2: _T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_exp_nancorr(
    a1: _T,
    a2: _T,
    /,
    *,
    alpha: float | FloatArray,
    min_weight: float = 0,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
