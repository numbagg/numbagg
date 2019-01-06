import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from numbagg.moving import ewm_nanmean, move_nanmean


@pytest.fixture
def array():
    return np.random.rand(2000).reshape(200, -1)


@pytest.mark.parametrize('com', [0.5, 3])
def test_ewma(array, com):

    array = array[0]
    expected = pd.Series(array).ewm(com=com).mean()
    result = ewm_nanmean(array, com)

    assert_almost_equal(expected, result)


def test_ewma_2d(array):

    expected = pd.DataFrame(array).T.ewm(com=3).mean().T
    result = ewm_nanmean(array, 3)

    assert_almost_equal(expected, result)


def test_movemean(array):

    array = array[0]
    expected = pd.Series(array).rolling(window=10).mean()
    result = move_nanmean(array, 10)

    assert_almost_equal(expected, result)


def test_movemean_window(array):

    with pytest.raises(ValueError):
        move_nanmean(array, window=0.5)
