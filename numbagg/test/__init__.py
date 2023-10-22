import pandas as pd

from .. import (
    move_exp_nancorr,
    move_exp_nancount,
    move_exp_nancov,
    move_exp_nanmean,
    move_exp_nanstd,
    move_exp_nansum,
    move_exp_nanvar,
)

# move_mean,
# nanargmax,
# nanargmin,
# nanmax,
# nanmean,
# nanmin,
# nanstd,


def pandas_ewm_setup(a, alpha):
    return pd.DataFrame(a).T.ewm(alpha=alpha)


def pandas_ewm_2arg_setup(a1, a2, alpha):
    return pd.DataFrame(a1).T.ewm(alpha=alpha), pd.DataFrame(a2).T


COMPARISONS = {
    move_exp_nancount: dict(
        # There's no pandas equivalent for move_exp_nancount
        pandas=dict(
            setup=lambda x, alpha: pd.DataFrame(x).T.notnull().ewm(alpha=alpha),
            run=lambda x: x.sum().T,
        )
    ),
    move_exp_nanvar: dict(
        pandas=dict(
            setup=pandas_ewm_setup,
            run=lambda x: x.var().T,
        )
    ),
    move_exp_nanstd: dict(
        pandas=dict(
            setup=pandas_ewm_setup,
            run=lambda x: x.std().T,
        )
    ),
    move_exp_nansum: dict(
        pandas=dict(
            setup=pandas_ewm_setup,
            run=lambda x: x.sum().T,
        )
    ),
    move_exp_nanmean: dict(
        pandas=dict(
            setup=pandas_ewm_setup,
            run=lambda x: x.mean().T,
        )
    ),
    move_exp_nancorr: dict(
        pandas=dict(
            setup=pandas_ewm_2arg_setup,
            run=lambda arrays: arrays[0].corr(arrays[1]).T,
        ),
    ),
    move_exp_nancov: dict(
        pandas=dict(
            setup=pandas_ewm_2arg_setup,
            run=lambda arrays: arrays[0].cov(arrays[1]).T,
        ),
    ),
}
