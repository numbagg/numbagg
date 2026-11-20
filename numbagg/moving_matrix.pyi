import numpy as np

from numbagg.utils import FloatArray

__all__ = [
    "move_corrmatrix",
    "move_covmatrix",
    "move_exp_nancorrmatrix",
    "move_exp_nancovmatrix",
]

def move_corrmatrix(
    a: np.ndarray,
    window: int,
    min_count: int | None = None,
    **kwargs,
) -> np.ndarray: ...
def move_covmatrix(
    a: np.ndarray,
    window: int,
    min_count: int | None = None,
    **kwargs,
) -> np.ndarray: ...
def move_exp_nancorrmatrix(
    a: np.ndarray,
    alpha: float | FloatArray,
    min_weight: float = 0,
    **kwargs,
) -> np.ndarray: ...
def move_exp_nancovmatrix(
    a: np.ndarray,
    alpha: float | FloatArray,
    min_weight: float = 0,
    **kwargs,
) -> np.ndarray: ...
