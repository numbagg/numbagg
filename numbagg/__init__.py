from .funcs import (
    allnan,
    anynan,
    count,
    nanargmax,
    nanargmin,
    nanmax,
    nanmean,
    nanmin,
    nanstd,
    nansum,
    nanvar,
)
from .moving import move_exp_nanmean, move_exp_nansum, move_mean

try:
    from importlib.metadata import version as _version
except ImportError:
    # if the fallback library is missing, we are doomed.
    from importlib_metadata import version as _version

try:
    __version__ = _version("numbagg")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"

__all__ = [
    allnan,
    anynan,
    count,
    nanargmax,
    nanargmin,
    nanmax,
    nanmean,
    nanmin,
    nanstd,
    nansum,
    nanvar,
    move_exp_nanmean,
    move_exp_nansum,
    move_mean,
    __version__,
]
