import numbagg

import numpy as np


def wrap_pure_python(func):
    def wrapper(x, *args, **kwargs):
        kwargs.pop('axis', None)
        out = np.empty_like(x)
        func(x, *args, out=out, **kwargs)
        return out
    return wrapper


funcs_reference_funcs = {
    numbagg.allnan: lambda x, **kwargs: np.all(np.isnan(x), **kwargs),
    numbagg.nansum: np.nansum,
    numbagg.nanmean: np.nanmean,
    numbagg.nanmin: np.nanmin,
    numbagg.count: lambda x, **kwargs: np.sum(~np.isnan(x), **kwargs),
}

moving_references_funcs = {
    numbagg.move_nanmean: wrap_pure_python(numbagg.move_nanmean.func),
}


def allclose(actual, desired, **kwargs):
    if getattr(actual, 'shape', ()) != getattr(desired, 'shape', ()):
        return False
    return np.all(np.isclose(actual, desired, equal_nan=True, **kwargs))


def check_func(numbagg_func, ref_func, x, axis):
    actual = numbagg_func(x, axis=axis)
    desired = ref_func(x, axis=axis)
    assert allclose(actual, desired), (numbagg_func, axis)


def test_funcs():
    x = np.random.RandomState(42).randn(100, 100)
    x[x < -1] = np.NaN

    for numbagg_func, ref_func in funcs_reference_funcs.items():
        for axis in [None, 0, -1, (0, 1)]:
            yield check_func, numbagg_func, ref_func, x, axis


def check_moving_func(numbagg_func, ref_func, x, window, axis):
    actual = numbagg_func(x, window, axis=axis)
    desired = ref_func(x, window, axis=axis)
    assert allclose(actual, desired), (numbagg_func, window, axis)


def test_moving():
    y = np.random.RandomState(42).randn(1000)
    y[y < -1] = np.NaN

    for numbagg_func, ref_func in moving_references_funcs.items():
        for window in [1, 3, 6]:
            yield check_moving_func, numbagg_func, ref_func, y, window, 0
