import numbers
import numba
import numpy as np

from .cache import cached_property, FunctionCache
from .transform import rewrite_ndreduce


def _nd_func_maker(cls, arg, **kwargs):
    if callable(arg) and not kwargs:
        return cls(arg)
    else:
        return lambda func: cls(func, signature=arg, **kwargs)


def ndreduce(*args, **kwargs):
    """Create an N-dimensional aggregation function."""
    return _nd_func_maker(NumbaNDReduce, *args, **kwargs)


def ndmoving(*args, **kwargs):
    """Create an N-dimensional moving window function along one dimension."""
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

        # gufunc_sig = gufunc_string_signature(rewriten_sig[0])
        # print(gufunc_sig)
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


DEFAULT_MOVING_SIGNATURE = ((numba.float64[:], numba.int64, numba.float64[:]),)


class NumbaNDMoving(object):
    def __init__(self, func, signature=DEFAULT_MOVING_SIGNATURE):
        self.func = func

        ndims = tuple(ndim(arg) for arg in signature[0])
        for sig in signature:
            if not isinstance(sig, tuple):
                raise TypeError('signatures for ndmoving must be tuples: {}'
                                .format(signature))
            if not (ndim(sig[0]) == 1
                    and all(ndim(s) == 0 for s in sig[1:-1])
                    and ndim(sig[-1]) == 1):
                raise ValueError('invalid signature for ndmoving: {}'
                                 .format(signature))
            if ndims != tuple(ndim(arg) for arg in sig):
                raise ValueError('inconsistent signatures for ndmoving: {}'
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
        return vectorize(self.transformed_func)

    def __call__(self, arr, window, axis=-1):
        axis = _validate_axis(axis, arr.ndim)
        window = np.asarray(window)
        # TODO: test this validation
        if (window < 1).any() or (window > arr.shape[axis]).any():
            raise ValueError(MOVE_WINDOW_ERR_MSG % (arr.shape[axis], window))
        arr = np.moveaxis(arr, axis, -1)
        result = self.gufunc(arr, window)
        return np.moveaxis(result, -1, axis)
