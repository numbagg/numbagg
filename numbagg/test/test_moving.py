import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from numbagg.moving import move_mean, rolling_exp_nanmean


@pytest.fixture
def array():
    arr = np.random.rand(2000).reshape(10, -1)
    return np.where(arr > 0.1, arr, np.nan)


@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_rolling_exp_nanmean(array, alpha):

    array = array[0]
    expected = pd.Series(array).ewm(alpha=alpha).mean()
    result = rolling_exp_nanmean(array, alpha)

    assert_almost_equal(expected, result)


def test_rolling_exp_nanmean_2d(array):

    expected = pd.DataFrame(array).T.ewm(alpha=0.1).mean().T
    result = rolling_exp_nanmean(array, 0.1)

    assert_almost_equal(expected, result)


def test_move_mean(array):

    array = array[0]

    expected = pd.Series(array).rolling(window=10, min_periods=1).mean()
    result = move_mean(array, 10, min_count=1)
    assert_almost_equal(expected, result)

    expected = pd.Series(array).rolling(window=3, min_periods=3).mean()
    result = move_mean(array, 3, min_count=3)
    assert_almost_equal(expected, result)


def test_move_mean_window(array):

    with pytest.raises(TypeError):
        move_mean(array, window=0.5)
    with pytest.raises(ValueError):
        move_mean(array, window=-1)
    with pytest.raises(ValueError):
        move_mean(array, window=1, min_count=-1)
