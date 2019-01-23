import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from numbagg import move_nanmean, move_exp_nanmean


@pytest.fixture
def array():
    arr = np.random.rand(2000).reshape(10, -1)
    return np.where(arr > 0.1, arr, np.nan)


@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_move_exp_nanmean(array, alpha):

    array = array[0]
    expected = pd.Series(array).ewm(alpha=alpha).mean()
    result = move_exp_nanmean(array, alpha)

    assert_almost_equal(expected, result)


def test_move_exp_nanmean_2d(array):

    expected = pd.DataFrame(array).T.ewm(alpha=0.1).mean().T
    result = move_exp_nanmean(array, 0.1)

    assert_almost_equal(expected, result)


def test_movemean(array):

    array = array[0]
    # algo doesn't currently conform to pandas handling of NaNs / min_periods
    array = np.where(np.isnan(array), 0, array)
    expected = pd.Series(array).rolling(window=10).mean()
    result = move_nanmean(array, 10)

    assert_almost_equal(expected, result)


def test_movemean_window(array):

    with pytest.raises(ValueError):
        move_nanmean(array, window=0.5)
