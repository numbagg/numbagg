from typing import TypeVar

from numbagg.moving_matrix import move_corrmatrix, move_covmatrix
from numbagg.utils import AxisLike, FloatArray

__all__ = [
    "move_mean",
    "move_sum",
    "move_std",
    "move_var",
    "move_cov",
    "move_corr",
    "move_covmatrix",
    "move_corrmatrix",
]

_T = TypeVar("_T", bound=FloatArray)

def move_mean(
    arr: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: AxisLike = -1,
) -> _T: ...
def move_sum(
    arr: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: AxisLike = -1,
) -> _T: ...
def move_std(
    arr: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: AxisLike = -1,
) -> _T: ...
def move_var(
    arr: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: AxisLike = -1,
) -> _T: ...
def move_cov(
    a: _T,
    b: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: AxisLike = -1,
) -> _T: ...
def move_corr(
    a: _T,
    b: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: AxisLike = -1,
) -> _T: ...
