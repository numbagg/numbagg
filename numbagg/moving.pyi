from typing import TypeVar

import numpy as np

from numbagg.utils import FloatArray

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
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_sum(
    arr: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_std(
    arr: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_var(
    arr: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_cov(
    a: _T,
    b: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_corr(
    a: _T,
    b: _T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int | tuple[int, ...] = -1,
) -> _T: ...
def move_covmatrix(
    a: np.ndarray,
    window: int,
    min_count: int | None = None,
    **kwargs,
) -> np.ndarray: ...
def move_corrmatrix(
    a: np.ndarray,
    window: int,
    min_count: int | None = None,
    **kwargs,
) -> np.ndarray: ...
