import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from numbagg import move_mean, move_exp_nanmean


@pytest.fixture
def rand_array():
    arr = np.random.RandomState(0).rand(2000).reshape(10, -1)
    return np.where(arr > 0.1, arr, np.nan)


@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_move_exp_nanmean(rand_array, alpha):

    array = rand_array[0]
    expected = pd.Series(array).ewm(alpha=alpha).mean()
    result = move_exp_nanmean(array, alpha)

    assert_almost_equal(expected, result)


def test_move_exp_nanmean_2d(rand_array):

    expected = pd.DataFrame(rand_array).T.ewm(alpha=0.1).mean().T
    result = move_exp_nanmean(rand_array, 0.1)

    assert_almost_equal(expected, result)


def test_move_mean():
    array = np.arange(100.0)
    array[::7] = np.nan

    expected = pd.Series(array).rolling(window=5, min_periods=1).mean().values
    result = move_mean(array, 5, min_count=1)
    assert_almost_equal(expected, result)


def test_move_mean_random(rand_array):
    array = rand_array[0]

    expected = pd.Series(array).rolling(window=10, min_periods=1).mean().values
    result = move_mean(array, 10, min_count=1)
    assert_almost_equal(expected, result)

    expected = pd.Series(array).rolling(window=3, min_periods=3).mean().values
    result = move_mean(array, 3, min_count=3)
    assert_almost_equal(expected, result)


def test_move_mean_window(rand_array):

    with pytest.raises(TypeError):
        move_mean(rand_array, window=0.5)
    with pytest.raises(ValueError):
        move_mean(rand_array, window=-1)
    with pytest.raises(ValueError):
        move_mean(rand_array, window=1, min_count=-1)
