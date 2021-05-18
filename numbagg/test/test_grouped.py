import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from numbagg.grouped import group_nanmean


def groupby_mean_pandas(values, group):
    # like pd.Series(values).groupby(group).mean()
    labels, uniques = pd.factorize(group, sort=True)
    result = group_nanmean(values, labels, num_labels=len(uniques))
    return pd.Series(result, index=uniques)


def test_groupby_mean_pandas():
    rs = np.random.RandomState(0)
    values = rs.rand(2000)
    group = rs.choice([np.nan, 1, 2, 3, 4, 5], size=values.shape)
    expected = pd.Series(values).groupby(group).mean()
    result = groupby_mean_pandas(values, group)
    assert_almost_equal(expected.values, result.values)


def test_group_nanmean_axis_1d_labels():
    values = np.arange(5.0)
    labels = np.arange(5)
    result = group_nanmean(values, labels)
    assert_almost_equal(values, result)

    values = np.arange(25.0).reshape(5, 5)
    labels = np.arange(5)

    with pytest.raises(ValueError) as excinfo:
        result = group_nanmean(values, labels)
    assert "axis required" in str(excinfo.value)

    result = group_nanmean(values, labels, axis=1)
    assert_almost_equal(values, result)

    result = group_nanmean(values, labels, axis=(1,))
    assert_almost_equal(values, result)

    result = group_nanmean(values, labels, axis=0)
    assert_almost_equal(values.T, result)

    with pytest.raises(ValueError) as excinfo:
        group_nanmean(values, labels[:4], axis=0)
    assert "must have same shape" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        group_nanmean(values, labels[:4], axis=(0,))
    assert "must have same shape" in str(excinfo.value)

    result = group_nanmean(values, [0, 0, -1, 1, 1], axis=1)
    expected = np.stack(
        [values[:, :2].mean(axis=1), values[:, 3:].mean(axis=1)], axis=-1
    )
    assert_almost_equal(expected, result)


def test_group_nanmean_axis_2d_labels():
    values = np.arange(25.0).reshape(5, 5)
    labels = np.arange(25).reshape(5, 5)
    result = group_nanmean(values, labels)
    assert_almost_equal(values.ravel(), result)

    values = np.arange(125.0).reshape(5, 5, 5)
    result = group_nanmean(values, labels, axis=(1, 2))
    assert_almost_equal(values.reshape(5, -1), result)
