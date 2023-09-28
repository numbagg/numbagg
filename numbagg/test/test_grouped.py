import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from numbagg.grouped import (
    group_nanargmax,
    group_nanargmin,
    group_nancount,
    group_nanfirst,
    group_nanlast,
    group_nanmean,
    group_nanprod,
    group_nansum,
    group_nansum_of_squares,
)

ALL_FUNCS = [
    group_nanmean,
    group_nansum,
]
NP_FUNCS = [
    np.mean,
    np.sum,
]

FUNCTIONS = [
    (group_nanmean, lambda x: x.mean()),
    (group_nansum, lambda x: x.sum()),
    (group_nancount, lambda x: x.count()),
    (group_nanfirst, lambda x: x.first()),
    (group_nanlast, lambda x: x.last()),
    (group_nanprod, lambda x: x.prod()),
    (group_nanargmax, lambda x: x.idxmax()),
    (group_nanargmin, lambda x: x.idxmin()),
    (group_nansum_of_squares, lambda x: x.agg(lambda y: (y**2).sum())),
]


@pytest.mark.parametrize("numbagg_func, pandas_func", FUNCTIONS)
def test_group_pandas_comparison(numbagg_func, pandas_func):
    rs = np.random.RandomState(0)
    values = rs.rand(2000)
    values = np.where(values > 0.1, values, np.nan)
    # TODO: do we need to support nan here?
    # group = rs.choice([np.nan, 1, 2, 3, 4, 5], size=values.shape)
    group = rs.choice([0, 1, 2, 3, 4, 5], size=values.shape)
    expected = pandas_func(pd.Series(values).groupby(group))
    result = numbagg_func(values, group)
    assert_almost_equal(expected.values, result)


def groupby_mean_pandas(values, group):
    # like pd.Series(values).groupby(group).mean()
    labels, uniques = pd.factorize(group, sort=True)
    result = group_nanmean(values, labels, num_labels=len(uniques))
    return pd.Series(result, index=uniques)


@pytest.mark.parametrize(
    "type_", [np.float32, np.float64, np.bool_, np.int32, np.int64]
)
def test_groupby_mean_pandas(type_):
    rs = np.random.RandomState(0)
    values = rs.rand(2000).astype(type_)
    group = rs.choice([np.nan, 1, 2, 3, 4, 5], size=values.shape)
    expected = pd.Series(values).groupby(group).mean()
    result = groupby_mean_pandas(values, group)
    assert_almost_equal(expected.values, result.values)


@pytest.mark.parametrize("func, npfunc", zip(ALL_FUNCS, NP_FUNCS))
def test_group_func_axis_1d_labels(func, npfunc):
    values = np.arange(5.0)
    labels = np.arange(5)
    result = func(values, labels)
    assert_almost_equal(values, result)

    values = np.arange(25.0).reshape(5, 5)
    labels = np.arange(5)

    with pytest.raises(ValueError) as excinfo:
        result = func(values, labels)
    assert "axis required" in str(excinfo.value)

    result = func(values, labels, axis=1)
    assert_almost_equal(values, result)

    result = func(values, labels, axis=(1,))
    assert_almost_equal(values, result)

    result = func(values, labels, axis=0)
    assert_almost_equal(values.T, result)

    with pytest.raises(ValueError) as excinfo:
        func(values, labels[:4], axis=0)
    assert "must have same shape" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        func(values, labels[:4], axis=(0,))
    assert "must have same shape" in str(excinfo.value)

    result = func(values, [0, 0, -1, 1, 1], axis=1)
    expected = np.stack(
        [npfunc(values[:, :2], axis=1), npfunc(values[:, 3:], axis=1)], axis=-1
    )
    assert_almost_equal(expected, result)


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_group_nanmean_axis_2d_labels(func):
    values = np.arange(25.0).reshape(5, 5)
    labels = np.arange(25).reshape(5, 5)
    result = func(values, labels)
    assert_almost_equal(values.ravel(), result)

    values = np.arange(125.0).reshape(5, 5, 5)
    result = func(values, labels, axis=(1, 2))
    assert_almost_equal(values.reshape(5, -1), result)
