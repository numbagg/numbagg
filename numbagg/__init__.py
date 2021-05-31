import pkg_resources

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
    __version__ = pkg_resources.get_distribution("numbagg").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

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
