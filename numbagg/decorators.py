import inspect

import numba
import numpy as np

from .cache import cached_property, FunctionCache
from .transform import _transform_agg_source, _transform_moving_source


def _nd_func_maker(cls, arg, *args, **kwargs):
    if callable(arg) and not args and not kwargs:
        return cls(arg)
    else:
        return lambda func: cls(func, arg, *args, **kwargs)


def ndreduce(*args, **kwargs):
    """Turn a function the aggregates an array into a single value, into a
    multi-dimensional aggregation function accelerated by numba.
    """
    return _nd_func_maker(NumbaNDReduce, *args, **kwargs)


def ndmoving(*args, **kwargs):
    """Accelerate a moving window function.
    """
    return _nd_func_maker(NumbaNDMoving, *args, **kwargs)


def _validate_axis(axis, ndim):
    """Helper function to convert axis into a non-negative integer, or raise if
    it's invalid.
    """
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError('invalid axis %s' % axis)
    return axis


class NumbaNDReduce(object):
    def __init__(self, func, dtype_map=['float64,float64']):
        self.func = func
        self.dtype_map = dtype_map
        self._gufunc_cache = FunctionCache(self._create_gufunc)

    @cached_property
    def transformed_func(self):
        return _transform_agg_source(self.func)

    def _create_gufunc(self, ndim):
        # creating compiling gufunc has some significant overhead (~130ms per
        # function and number of dimensions to aggregate), so do this in a
        # lazy fashion
        colons = ','.join(':' for _ in range(ndim))
        dtype_str = []
        for d in self.dtype_map:
            k, v = d.split(',')
            dtype_str.append('void(%s[%s], %s[:])' % (k, colons, v))

        sig = '(%s)->()' % ','.join(list('abcdefgijk')[:ndim])
        vectorize = numba.guvectorize(dtype_str, sig, nopython=True)
        return vectorize(self.transformed_func)

    @cached_property
    def _jit_func(self):
        return numba.jit(self.func, nopython=True)

    def __call__(self, arr, axis=None):
        if axis is None:
            # axis = range(arr.ndim)
            # use @jit instead since numba accelerates it better
            f = self._jit_func
        elif np.isscalar(axis):
            axis = _validate_axis(axis, arr.ndim)
            arr = arr.swapaxes(axis, -1)
            f = self._gufunc_cache[1]
        else:
            axis = [_validate_axis(a, arr.ndim) for a in axis]
            all_axes = [n for n in range(arr.ndim)
                        if n not in axis] + list(axis)
            arr = arr.transpose(all_axes)
            f = self._gufunc_cache[len(axis)]
        return f(arr)


MOVE_WINDOW_ERR_MSG = "invalid window (not between 1 and %d, inclusive): %r"


class NumbaNDMoving(object):
    def __init__(self, func, dtype_map=['float64,int64,float64']):
        self.func = func
        self.dtype_map = dtype_map

    @cached_property
    def transformed_func(self):
        return _transform_moving_source(self.func)

    @cached_property
    def gufunc(self):
        extra_args = len(inspect.getargspec(self.func).args) - 2
        dtype_str = ['void(%s)' % ','.join('%s[:]' % e
                                           for e in d.split(','))
                     for d in self.dtype_map]
        sig = '(n)%s->(n)' % ''.join(',()' for _ in range(extra_args))
        vectorize = numba.guvectorize(dtype_str, sig, nopython=True)
        return vectorize(self.transformed_func)

    def __call__(self, arr, window, axis=-1):
        axis = _validate_axis(axis, arr.ndim)
        window = np.asarray(window)
        # TODO: test this validation
        if (window < 1).any() or (window > arr.shape[axis]).any():
            raise ValueError(MOVE_WINDOW_ERR_MSG % (arr.shape[axis], window))
        arr = arr.swapaxes(axis, -1)
        return self.gufunc(arr, window)
