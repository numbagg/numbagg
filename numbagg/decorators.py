import numbers

import numba
import numpy as np
import pandas as pd

from .cache import FunctionCache, cached_property
from .transform import rewrite_ndreduce


def _nd_func_maker(cls, arg, **kwargs):
    if callable(arg) and not kwargs:
        return cls(arg)
    else:
        return lambda func: cls(func, signature=arg, **kwargs)


def ndreduce(*args, **kwargs):
    """Create an N-dimensional aggregation function.

    Functions should have signatures of the form output_type(input_type), where
    input_type and output_type are numba dtypes. This decorator rewrites them
    to accept input arrays of arbitrary dimensionality, with an additional
    optional `axis`, which accepts integers or tuples of integers (defaulting
    to `axis=None` for all axes).

    For example, to write a simplified version of `np.sum(arr, axis=None)`::

        from numba import float64

        @ndreduce([
            float64(float64)
        ])
        def sum(a):
            asum = 0.0
            for ai in a.flat:
                asum += ai
            return asum
    """
    return _nd_func_maker(NumbaNDReduce, *args, **kwargs)


def ndmoving(*args, **kwargs):
    """Create an N-dimensional moving window function along one dimension.

    Functions should accept arguments for the input array, a window
    size and the output array.

    For example, to write a simplified (and naively implemented) moving window
    sum::

        @ndmoving([
            (float64[:], int64, float64[:]),
        ])
        def move_sum(a, window, out):
            for i in range(a.size):
                for j in range(window):
                    if i - j > 0:
                        out[i] += a[i - j]

    The default validator for the window argument is between 1 and the array 
    length. Pass a function to ndmoving as `window_validator` for an 
    alternative validator
    """
    return _nd_func_maker(NumbaNDMoving, *args, **kwargs)


def groupndreduce(*args, **kwargs):
    """Create an N-dimensional grouped aggregation function."""
    return _nd_func_maker(NumbaGroupNDReduce, *args, **kwargs)


def _validate_axis(axis, ndim):
    """Helper function to convert axis into a non-negative integer, or raise if
    it's invalid.
    """
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError('invalid axis %s' % axis)
    return axis


def ndim(arg):
    return getattr(arg, 'ndim', 0)


_ALPHABET = 'abcdefghijkmnopqrstuvwxyz'


def _gufunc_arg_str(arg):
    return '(%s)' % ','.join(_ALPHABET[:ndim(arg)])


def gufunc_string_signature(numba_args):
    """Convert a tuple of numba types into a numpy gufunc signature.

    The last type is used as output argument.

    Example:

    >>> gufunc_string_signature((float64[:], float64))
    '(a)->()'
    """
    return (','.join(map(_gufunc_arg_str, numba_args[:-1]))
            + '->' + _gufunc_arg_str(numba_args[-1]))


DEFAULT_REDUCE_SIGNATURE = (numba.float32(numba.float32),
                            numba.float64(numba.float64))


class NumbaNDReduce(object):
    def __init__(self, func, signature=DEFAULT_REDUCE_SIGNATURE):
        self.func = func

        for sig in signature:
            if not hasattr(sig, 'return_type'):
                raise ValueError(
                    'signatures for ndreduce must be functions: {}'
                    .format(signature))
            if any(ndim(arg) != 0 for arg in sig.args):
                raise ValueError(
                    'all arguments in signature for ndreduce must be scalars: '
                    ' {}'.format(signature))
            if ndim(sig.return_type) != 0:
                raise ValueError(
                    'return type for ndreduce must be a scalar: {}'
                    .format(signature))
        self.signature = signature

        self._gufunc_cache = FunctionCache(self._create_gufunc)

    @property
    def __name__(self):
        return self.func.__name__

    def __repr__(self):
        return '<numbagg.decorators.NumbaNDReduce %s>' % self.__name__

    @cached_property
    def transformed_func(self):
        return rewrite_ndreduce(self.func)

    @cached_property
    def _jit_func(self):
        vectorize = numba.jit(self.signature, nopython=True)
        return vectorize(self.func)

    def _create_gufunc(self, core_ndim):
        # creating compiling gufunc has some significant overhead (~130ms per
        # function and number of dimensions to aggregate), so do this in a
        # lazy fashion
        numba_sig = []
        for input_sig in self.signature:
            new_sig = ((input_sig.args[0][(slice(None),) * core_ndim],)
                       + input_sig.args[1:] + (input_sig.return_type[:],))
            numba_sig.append(new_sig)

        first_sig = self.signature[0]
        gufunc_sig = gufunc_string_signature(
            (first_sig.args[0][(slice(None),) * core_ndim],)
            + first_sig.args[1:] + (first_sig.return_type,))

        vectorize = numba.guvectorize(numba_sig, gufunc_sig, nopython=True)
        return vectorize(self.transformed_func)

    def __call__(self, arr, axis=None):
        if axis is None:
            # TODO: switch to using jit_func (it's faster), once numba reliably
            # returns the right dtype
            # see: https://github.com/numba/numba/issues/1087
            # f = self._jit_func
            f = self._gufunc_cache[arr.ndim]
        elif isinstance(axis, numbers.Number):
            arr = np.moveaxis(arr, axis, -1)
            f = self._gufunc_cache[1]
        else:
            arr = np.moveaxis(arr, axis, range(-len(axis), 0, 1))
            f = self._gufunc_cache[len(axis)]
        return f(arr)


MOVE_WINDOW_ERR_MSG = "invalid window (not between 1 and %d, inclusive): %r"


def rolling_validator(arr, window):
    if (window < 1) or (window > arr.shape[-1]):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (arr.shape[-1], window))


DEFAULT_MOVING_SIGNATURE = ((numba.float64[:], numba.int64, numba.float64[:]),)


class NumbaNDMoving(object):
    def __init__(self, func, signature=DEFAULT_MOVING_SIGNATURE,
                 window_validator=rolling_validator):
        self.func = func
        self.window_validator = window_validator

        for sig in signature:
            if not isinstance(sig, tuple):
                raise TypeError('signatures for ndmoving must be tuples: {}'
                                .format(signature))
        self.signature = signature

    @property
    def __name__(self):
        return self.func.__name__

    def __repr__(self):
        return '<numbagg.decorators.NumbaNDMoving %s>' % self.__name__

    @cached_property
    def gufunc(self):
        gufunc_sig = gufunc_string_signature(self.signature[0])
        vectorize = numba.guvectorize(
            self.signature, gufunc_sig, nopython=True)
        return vectorize(self.func)

    def __call__(self, arr, window, axis=-1):
        axis = _validate_axis(axis, arr.ndim)
        arr = np.moveaxis(arr, axis, -1)
        self.window_validator(arr, window)
        result = self.gufunc(arr, window)
        return np.moveaxis(result, -1, axis)


class NumbaGroupNDReduce(object):
    def __init__(self, func, signature=DEFAULT_MOVING_SIGNATURE):
        self.func = func

        for sig in signature:
            if not isinstance(sig, tuple):
                raise TypeError('signatures for ndmoving must be tuples: {}'
                                .format(signature))
            if len(sig) != 3:
                raise TypeError('signature has wrong number of argument != 3: '
                                '{}'.format(signature))
            if any(ndim(arg) != 0 for arg in sig):
                raise ValueError(
                    'all arguments in signature for ndreduce must be scalars: '
                    ' {}'.format(signature))
        self.signature = signature
        self._gufunc_cache = FunctionCache(self._create_gufunc)

    @property
    def __name__(self):
        return self.func.__name__

    def __repr__(self):
        return '<numbagg.decorators.NumbaGroupNDReduce %s>' % self.__name__

    def _create_gufunc(self, core_ndim):
        # compiling gufuncs has some significant overhead (~130ms per function
        # and number of dimensions to aggregate), so do this in a lazy fashion
        numba_sig = []
        slices = (slice(None),) * core_ndim
        for input_sig in self.signature:
            values, labels, out = input_sig
            new_sig = (values[slices], labels[slices], out[:])
            numba_sig.append(new_sig)

        first_sig = numba_sig[0]
        gufunc_sig = ','.join(2 * [_gufunc_arg_str(first_sig[0])]) + ',(z)'
        vectorize = numba.guvectorize(numba_sig, gufunc_sig, nopython=True)
        return vectorize(self.func)

    def __call__(self, values, labels, axis=None, num_labels=None):

        values = np.asarray(values)
        labels = np.asarray(labels)

        # passing num_lables avoids checking for factorized labels
        if num_labels is None:
            if is_factorized(labels):
                num_labels = np.max(labels) + 1
                uniques = np.arange(num_labels)
            else:
                labels_factorized, uniques = pd.factorize(
                    labels.ravel(), sort=True)
                labels_factorized = labels_factorized.reshape(labels.shape)
                num_labels = uniques.size
                # overwrite with the factorized verison
                labels = labels_factorized
        else:
            uniques = np.arange(num_labels)

        if axis is None:
            if values.shape != labels.shape:
                raise ValueError(
                    'axis required if values and labels have different '
                    'shapes: {} vs {}'.format(values.shape, labels.shape))
            gufunc = self._gufunc_cache[values.ndim]
        elif isinstance(axis, numbers.Number):
            if labels.shape != (values.shape[axis],):
                raise ValueError(
                    'values must have same shape along axis as labels: '
                    '{} vs {}'.format((values.shape[axis],), labels.shape))
            values = np.moveaxis(values, axis, -1)
            gufunc = self._gufunc_cache[1]
        else:
            values_shape = tuple(values.shape[ax] for ax in axis)
            if labels.shape != values_shape:
                raise ValueError(
                    'values must have same shape along axis as labels: '
                    '{} vs {}'.format(values_shape, labels.shape))
            values = np.moveaxis(values, axis, range(-len(axis), 0, 1))
            gufunc = self._gufunc_cache[len(axis)]

        broadcast_ndim = values.ndim - labels.ndim
        broadcast_shape = values.shape[:broadcast_ndim]
        result = np.zeros(broadcast_shape + (num_labels,), values.dtype)
        gufunc(values, labels, result)
        return result, uniques


def is_factorized(labels):
    if not np.issubdtype(labels.dtype, np.int_):
        return False
    # exclude labels below zero
    # we could force the caller to do this?
    included_labels = labels[np.where(labels >= 0)]
    if np.unique(included_labels).size != np.max(included_labels) + 1:
        return False
    return True
