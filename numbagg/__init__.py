from importlib.metadata import version as _version

from .funcs import (
    allnan,
    anynan,
    count,
    nanargmax,
    nanargmin,
    nancount,
    nanmax,
    nanmean,
    nanmin,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
)
from .moving import move_corr, move_cov, move_mean, move_std, move_sum, move_var
from .moving_exp import (
    move_exp_nancorr,
    move_exp_nancount,
    move_exp_nancov,
    move_exp_nanmean,
    move_exp_nanstd,
    move_exp_nansum,
    move_exp_nanvar,
)

try:
    __version__ = _version("numbagg")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"

__all__ = [
    "__version__",
    "allnan",
    "anynan",
    "count",
    # "move_count",
    "move_exp_nancorr",
    "move_exp_nancount",
    "move_exp_nancov",
    "move_exp_nanmean",
    "move_exp_nanstd",
    "move_exp_nansum",
    "move_exp_nanvar",
    "move_mean",
    "move_std",
    "move_var",
    "move_cov",
    "move_corr",
    "move_sum",
    "nanargmax",
    "nanargmin",
    "nancount",
    "nanmax",
    "nanmean",
    "nanmin",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
]
