from importlib.metadata import version as _version

from numba.core.typed_passes import _reload_parfors

# https://github.com/numba/numba/issues/8578
_reload_parfors()

from .funcs import (
    allnan,
    anynan,
    bfill,
    count,
    ffill,
    nanargmax,
    nanargmin,
    nancount,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
)
from .grouped import (
    group_nanall,
    group_nanany,
    group_nanargmax,
    group_nanargmin,
    group_nancount,
    group_nanfirst,
    group_nanlast,
    group_nanmax,
    group_nanmean,
    group_nanmin,
    group_nanprod,
    group_nanstd,
    group_nansum,
    group_nansum_of_squares,
    group_nanvar,
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

GROUPED_FUNCS = [
    group_nanall,
    group_nanany,
    group_nanargmax,
    group_nanargmin,
    group_nancount,
    group_nanfirst,
    group_nanlast,
    group_nanmax,
    group_nanmean,
    group_nanmin,
    group_nanprod,
    group_nanstd,
    group_nansum,
    group_nansum_of_squares,
    group_nanvar,
]

MOVE_EXP_FUNCS = [
    move_exp_nancorr,
    move_exp_nancount,
    move_exp_nancov,
    move_exp_nanmean,
    move_exp_nanstd,
    move_exp_nansum,
    move_exp_nanvar,
]

MOVE_FUNCS = [move_corr, move_cov, move_mean, move_std, move_sum, move_var]

AGGREGATION_FUNCS = [
    allnan,
    anynan,
    count,
    nanargmax,
    nanargmin,
    nancount,
    nanmax,
    nanmean,
    nanmedian,
    nanmin,
    nanquantile,
    nanstd,
    nansum,
    nanvar,
]

OTHER_FUNCS = [bfill, ffill]


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
    "bfill",
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
    "group_nanall",
    "group_nanany",
    "group_nanargmax",
    "group_nanargmin",
    "group_nancount",
    "group_nanfirst",
    "group_nanlast",
    "group_nanmax",
    "group_nanmean",
    "group_nanmin",
    "group_nanprod",
    "group_nanstd",
    "group_nansum",
    "group_nansum_of_squares",
    "group_nanvar",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
    "nanquantile",
    "nancount",
    "ffill",
    "move_exp_nanmean",
    "move_exp_nansum",
    "move_exp_nancount",
    "move_exp_nancorr",
    "move_exp_nancov",
    "move_exp_nanvar",
    "move_exp_nanstd",
    "move_mean",
    "__version__",
]
