from typing import TypeVar

from numbagg.utils import FloatArray

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
