import pandas as pd

from .. import (
    move_exp_nancorr,
    move_exp_nancov,
    move_exp_nanmean,
    move_exp_nanstd,
    move_exp_nansum,
    move_exp_nanvar,
    move_mean,
    nanargmax,
    nanargmin,
    nanmax,
    nanmean,
    nanmin,
    nanstd,
)


def two_arrays(array):
    # The second array as different enough that it'll have non-trivial correlation
    return array, array**2 + 1


def pandas_ewm_setup(array, alpha):
    return pd.DataFrame(array).T.ewm(alpha=alpha)


def pandas_ewm_2arg_setup(array, alpha):
    return pd.DataFrame(array).T.ewm(alpha=alpha), pd.DataFrame(array) ** 2 + 1


COMPARISONS = {
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
            run=lambda x, y: x.corr(y).T,
        ),
        setup=two_arrays,
    ),
    move_exp_nancov: dict(
        pandas=dict(
            setup=pandas_ewm_2arg_setup,
            run=lambda x, y: x.cov(y).T,
        ),
        setup=two_arrays,
    ),
}
