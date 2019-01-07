import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from numbagg.moving import ewm_nanmean


def test_ewma():

    array = np.random.rand(2000)
    expected = pd.Series(array).ewm(com=3).mean()
    result = ewm_nanmean(array, 3)

    assert_almost_equal(expected, result)


def test_ewma_nd():

    array = np.random.rand(4, 2000)
    result = np.empty(array.shape)
    ewm_nanmean(array, 3)
