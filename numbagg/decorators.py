import collections
import inspect
import re

import numba
import numpy as np


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


class FunctionCache(dict):
    """A simple dict-subclass for caching the return values of a function.
    """
    def __init__(self, func):
        self.func = func

    def __missing__(self, key):
        value = self[key] = self.func(key)
        return value


class cached_property(object):
    """A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.

    Source:
    https://github.com/pydanny/cached-property
    https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """
    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class NumbaNDReduce(object):
    def __init__(self, func, dtype_map=['float64,float64']):
        self.func = func
        self.dtype_map = dtype_map
        self._gufunc_cache = FunctionCache(self._create_gufunc)

    @cached_property
    def transformed_func(self):
        return _transform_agg_gufunc_source(self.func)

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
        return _transform_moving_gufunc_source(self.func)

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


def _apply_source_transform(func, transform_source):
    """A horrible hack to make the syntax for writing aggregators more
    Pythonic.

    This should go away once numba is more fully featured.
    """
    orig_source = inspect.getsource(func)
    source = transform_source(orig_source)
    scope = {}
    exec(source, globals(), scope)
    try:
        return scope['__transformed_func']
    except KeyError:
        raise TypeError('failed to rewrite function definition:\n%s'
                        % orig_source)


def _transform_agg_gufunc_source(func):
    """Transforms aggregation functions into something numba can handle.

    To be more precise, it converts functions with source that looks like

        @ndreduce
        def my_func(x)
            ...
            return foo

    into

        def __sub__gufunc(x, __out):
            ...
            __out[0] = foo

    which is the form numba needs for writing a gufunc that returns a scalar
    value.
    """
    def transform_source(source):
        # nb. the right way to do this would be use Python's ast module instead
        # of regular expressions.
        source = re.sub(
            r'^@ndreduce[^\n]*\ndef\s+[a-zA-Z_][a-zA-Z_0-9]*\((.*?)\)\:',
            r'def __transformed_func(\1, __out):', source, flags=re.DOTALL)
        source = re.sub(r'return\s+(.*)', r'__out[0] = \1', source)
        return source
    return _apply_source_transform(func, transform_source)


def _transform_moving_gufunc_source(func):
    """Transforms moving aggregation functions into something numba can handle.
    """
    def transform_source(source):
        source = re.sub(
            r'^@ndmoving[^\n]*\ndef\s+[a-zA-Z_][a-zA-Z_0-9]*\((.*?)\)\:',
            r'def __transformed_func(\1):', source, flags=re.DOTALL)
        source = re.sub(r'^(\s+.*)(window)', r'\1window[0]', source,
                        flags=re.MULTILINE)
        return source
    return _apply_source_transform(func, transform_source)
