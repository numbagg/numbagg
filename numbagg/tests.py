import numbagg

import numpy as np


reference_funcs = {
    numbagg.nansum: np.nansum,
    numbagg.nanmean: np.nanmean,
    numbagg.nanmin: np.nanmin,
    numbagg.count: lambda x, **kwargs: np.sum(~np.isnan(x), **kwargs),
}


def check_func(numbagg_func, ref_func, x, axis):
    actual = numbagg_func(x, axis=axis)
    desired = ref_func(x, axis=axis)
    assert np.allclose(actual, desired), (numbagg_func, axis)


def test_funcs():
    x = np.random.RandomState(42).randn(100, 100)
    x[x < -1] = np.NaN

    for numbagg_func, ref_func in reference_funcs.items():
        for axis in [None, 0, -1, (0, 1)]:
            yield check_func, numbagg_func, ref_func, x, axis
