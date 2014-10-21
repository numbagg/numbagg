import inspect
import re

import numba
import numpy as np


def ndreduce(arg, *args, **kwargs):
    """Turn a function the aggregates an array into a single value, into a
    multi-dimensional aggregation function accelerated by numba.
    """
    if callable(arg) and not args and not kwargs:
        return NumbaAggregator(arg)
    else:
        return lambda func: NumbaAggregator(func, arg, *args, **kwargs)


def _validate_axis(axis, ndim):
    """Helper function to convert axis into a non-negative integer, or raise if
    it's invalid.
    """
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError('invalid axis %s' % axis)
    return axis


class NumbaAggregator(object):
    def __init__(self, func, dtype_map=['float64->float64']):
        self.func = func
        self.dtype_map = dtype_map
        self._mangled_func = None
        self._gufunc_cache = {}
        self._jit_func = None

    @property
    def _guvec_func(self):
        if self._mangled_func is None:
            self._mangled_func = _mangle_gufunc_source(self.func)
        return self._mangled_func

    def _create_gufunc(self, ndim):
        # creating compiling gufunc has some significant overhead (~130ms per
        # function and number of dimensions to aggregate), so do this in a
        # lazy fashion
        colons = ','.join(':' for _ in range(ndim))
        dtype_str = []
        for d in self.dtype_map:
            k, v = d.split('->')
            dtype_str.append('void(%s[%s], %s[:])' % (k, colons, v))

        sig = '(%s)->()' % ','.join(list('abcdefgijk')[:ndim])

        return numba.guvectorize(dtype_str, sig)(self._guvec_func)

    def _get_gufunc(self, ndim):
        if ndim not in self._gufunc_cache:
            self._gufunc_cache[ndim] = self._create_gufunc(ndim)
        return self._gufunc_cache[ndim]

    def _get_jit_func(self):
        if self._jit_func is None:
            self._jit_func = numba.jit(self.func)
        return self._jit_func

    def __call__(self, arr, axis=None):
        if axis is None:
            # use @jit instead since numba accelerates it better
            f = self._get_jit_func()
            # axis = range(arr.ndim)
        else:
            if np.isscalar(axis):
                axis = [axis]
            axis = [_validate_axis(a, arr.ndim) for a in axis]
            all_axes = [n for n in range(arr.ndim)
                        if n not in axis] + list(axis)
            arr = arr.transpose(all_axes)
            f = self._get_gufunc(len(axis))
        return f(arr)


def _mangle_gufunc_source(func):
    """Transforms aggregation functions into something numba can handle.

    To be more precise, it converts functions with source that looks like

        def my_func(x)
            ...
            return foo

    into

        def __mangled_gufunc(x, __out):
            ...
            __out[0] = foo

    which is the form numba needs for writing a gufunc that returns a scalar
    value.

    This function shouldn't really need to exist. Also, it will fail for some
    edge-cases, because it really should use ast instead of re.
    """
    orig_source = inspect.getsource(func)
    # nb. the right way to do this would be use Python's ast module instead
    # of regular expressions.
    source = re.sub(r'^@ndreduce[^\n]*\n'
                    r'def\s+[a-zA-Z_][a-zA-Z_0-9]*\((.*?)\)\:',
                    r'def __mangled_gufunc(\1, __out):',
                    orig_source, flags=re.DOTALL)
    source = re.sub(r'return\s+(.*)', r'__out[0] = \1', source)
    exec(source, globals(), locals())
    try:
        return __mangled_gufunc
    except NameError:
        raise ValueError('failed to rewrite function definition:\n%s'
                         % orig_source)
