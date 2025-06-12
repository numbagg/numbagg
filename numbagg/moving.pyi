from numbagg.utils import FloatArray

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

def move_cov[T: FloatArray](
    a: T,
    b: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
def move_corr[T: FloatArray](
    a: T,
    b: T,
    /,
    *,
    window: int,
    min_count: int | None = None,
    axis: int = -1,
) -> T: ...
