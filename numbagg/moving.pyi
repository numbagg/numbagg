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

T = TypeVar("T", bound=FloatArray)

def move_mean(
    arr: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
def move_sum(
    arr: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
def move_std(
    arr: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
def move_var(
    arr: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
def move_cov(
    a: T,
    b: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
def move_corr(
    a: T,
    b: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
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
