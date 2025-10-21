from importlib.metadata import version as _version

from .funcs import (
    allnan,
    anynan,
    bfill,
    count,
    ffill,
    nanargmax,
    nanargmin,
    nancorrmatrix,
    nancount,
    nancovmatrix,
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
from .moving import (
    move_corr,
    move_cov,
    move_mean,
    move_std,
    move_sum,
    move_var,
)
from .moving_exp import (
    move_exp_nancorr,
    move_exp_nancount,
    move_exp_nancov,
    move_exp_nanmean,
    move_exp_nanstd,
    move_exp_nansum,
    move_exp_nanvar,
)
from .moving_matrix import (
    move_corrmatrix,
    move_covmatrix,
    move_exp_nancorrmatrix,
    move_exp_nancovmatrix,
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

MOVE_EXP_MATRIX_FUNCS = [
    move_exp_nancorrmatrix,
    move_exp_nancovmatrix,
]

MOVE_FUNCS = [
    move_corr,
    move_cov,
    move_mean,
    move_corrmatrix,
    move_covmatrix,
    move_std,
    move_sum,
    move_var,
]

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

MATRIX_FUNCS = [nancorrmatrix, nancovmatrix]

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
    "ffill",
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
    "move_corr",
    "move_corrmatrix",
    "move_cov",
    "move_covmatrix",
    "move_exp_nancorr",
    "move_exp_nancorrmatrix",
    "move_exp_nancount",
    "move_exp_nancov",
    "move_exp_nancovmatrix",
    "move_exp_nanmean",
    "move_exp_nanstd",
    "move_exp_nansum",
    "move_exp_nanvar",
    "move_mean",
    "move_std",
    "move_sum",
    "move_var",
    "nanargmax",
    "nanargmin",
    "nancorrmatrix",
    "nancount",
    "nancovmatrix",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
]
