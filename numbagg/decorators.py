from __future__ import annotations

from functools import cache, cached_property
from typing import Callable

import numba
import numpy as np

from .transform import rewrite_ndreduce


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
    return lambda func: NumbaNDReduce(func, *args, **kwargs)


def ndmoving(*args, **kwargs):
    """Create an N-dimensional moving window function along one dimension.

    Functions should accept arguments for the input array, a window
    size and the output array.

    For example, to write a simplified (and naively implemented) moving window
    sum::

        @ndmoving([
            (float64[:], int64, int64, float64[:]),
        ])
        def move_sum(a, window, min_count, out):
            for i in range(a.size):
                for j in range(window):
                    if i - j > min_count:
                        out[i] += a[i - j]
    """
    return lambda func: NumbaNDMoving(func, *args, **kwargs)


def ndmovingexp(*args, **kwargs):
    """N-dimensional exponential moving window function."""
    return lambda func: NumbaNDMovingExp(func, *args, **kwargs)


def groupndreduce(*args, **kwargs):
    """Create an N-dimensional grouped aggregation function."""
    return lambda func: NumbaGroupNDReduce(func, *args, **kwargs)


def ndfill(*args, **kwargs):
    return lambda func: NumbaNDFill(func, *args, **kwargs)


def ndim(arg):
    return getattr(arg, "ndim", 0)


_ALPHABET = "abcdefghijkmnopqrstuvwxyz"


def _gufunc_arg_str(arg):
    return "(%s)" % ",".join(_ALPHABET[: ndim(arg)])


def gufunc_string_signature(numba_args):
    """Convert a tuple of numba types into a numpy gufunc signature.

    The last type is used as output argument.

    Example:

    >>> gufunc_string_signature((float64[:], float64))
    '(a)->()'
    """
    return (
        ",".join(map(_gufunc_arg_str, numba_args[:-1]))
        + "->"
        + _gufunc_arg_str(numba_args[-1])
    )


class NumbaNDReduce:
    def __init__(self, func, signature):
        self.func = func

        for sig in signature:
            if not hasattr(sig, "return_type"):
                raise ValueError(
                    f"signatures for ndreduce must be functions: {signature}"
                )
            if any(ndim(arg) != 0 for arg in sig.args):
                raise ValueError(
                    "all arguments in signature for ndreduce must be scalars: "
                    f" {signature}"
                )
            if ndim(sig.return_type) != 0:
                raise ValueError(
                    f"return type for ndreduce must be a scalar: {signature}"
                )
        self.signature = signature

    @property
    def __name__(self):
        return self.func.__name__

    def __repr__(self):
        return f"numbagg.{self.__name__}"

    @cached_property
    def transformed_func(self):
        return rewrite_ndreduce(self.func)

    @cached_property
    def _jit_func(self):
        vectorize = numba.jit(self.signature, nopython=True)
        return vectorize(self.func)

    @cache
    def _create_gufunc(self, core_ndim):
        # creating compiling gufunc has some significant overhead (~130ms per
        # function and number of dimensions to aggregate), so do this in a
        # lazy fashion
        numba_sig = []
        for input_sig in self.signature:
            new_sig = (
                (input_sig.args[0][(slice(None),) * max(core_ndim, 1)],)
                + input_sig.args[1:]
                + (input_sig.return_type[:],)
            )
            numba_sig.append(new_sig)

        first_sig = self.signature[0]
        gufunc_sig = gufunc_string_signature(
            (
                first_sig.args[0][(slice(None),) * core_ndim]
                if core_ndim
                else first_sig.args[0],
            )
            + first_sig.args[1:]
            + (first_sig.return_type,)
        )

        # TODO: can't use `cache=True` because of the dynamic ast transformation
        vectorize = numba.guvectorize(numba_sig, gufunc_sig, nopython=True)
        return vectorize(self.transformed_func)

    def __call__(self, arr, *args, axis=None):
        if axis is None:
            # TODO: switch to using jit_func (it's faster), once numba reliably
            # returns the right dtype
            # see: https://github.com/numba/numba/issues/1087
            # f = self._jit_func
            f = self._create_gufunc(arr.ndim)
        elif isinstance(axis, int):
            arr = np.moveaxis(arr, axis, -1)
            f = self._create_gufunc(1)
        else:
            arr = np.moveaxis(arr, axis, range(-len(axis), 0, 1))
            f = self._create_gufunc(len(axis))
        return f(arr, *args)


MOVE_WINDOW_ERR_MSG = "invalid window (not between 1 and %d, inclusive): %r"


def rolling_validator(arr, window):
    if (window < 1) or (window > arr.shape[-1]):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (arr.shape[-1], window))


class NumbaNDMoving:
    def __init__(
        self,
        func: Callable,
        signature: list[tuple] = [
            (numba.float64[:], numba.int64, numba.float64[:]),
            (numba.float32[:], numba.int32, numba.float32[:]),
        ],
        window_validator=rolling_validator,
    ):
        self.func = func
        self.window_validator = window_validator

        for sig in signature:
            if not isinstance(sig, tuple):
                raise TypeError(
                    f"signatures for {self.__class__} must be tuples: {signature}"
                )
        self.signature = signature

    @property
    def __name__(self):
        return self.func.__name__

    def __repr__(self):
        return f"numbagg.{self.__name__}"

    @cached_property
    def gufunc(self):
        gufunc_sig = gufunc_string_signature(self.signature[0])
        vectorize = numba.guvectorize(
            self.signature, gufunc_sig, nopython=True, cache=True
        )
        return vectorize(self.func)

    def __call__(self, *arr: np.ndarray, window, min_count=None, axis=-1):
        if min_count is None:
            min_count = window
        if not 0 < window < arr[0].shape[axis]:
            raise ValueError(f"window not in valid range: {window}")
        if min_count < 0:
            raise ValueError(f"min_count must be positive: {min_count}")
        return self.gufunc(*arr, window, min_count, axis=axis)


class NumbaNDMovingExp(NumbaNDMoving):
    def __call__(self, *arr, alpha, min_weight=0, axis=-1):
        if alpha < 0:
            raise ValueError(f"alpha must be positive: {alpha}")
        # If an empty tuple is passed, there's no reduction to do, so we return the
        # original array.
        # Ref https://github.com/pydata/xarray/pull/5178/files#r616168398
        if axis == ():
            if len(arr) > 1:
                raise ValueError(
                    "`axis` cannot be an empty tuple when passing more than one array; since we default to returning the input."
                )
            return arr[0]
        # For the sake of speed, we ignore divide-by-zero and NaN warnings, and test for
        # their correct handling in our tests.
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.gufunc(*arr, alpha, min_weight, axis=axis)


# TODO: some copypasta from `NumbaNDMoving`; we could have a base class and reduce the duplication.


class NumbaNDFill:
    def __init__(
        self,
        func: Callable,
        signature: list[tuple] = [
            (numba.float64[:], numba.int64, numba.float64[:]),
            (numba.float32[:], numba.int32, numba.float32[:]),
        ],
    ):
        self.func = func

        for sig in signature:
            if not isinstance(sig, tuple):
                raise TypeError(
                    f"signatures for {self.__class__} must be tuples: {signature}"
                )
        self.signature = signature

    @property
    def __name__(self):
        return self.func.__name__

    def __repr__(self):
        return f"numbagg.{self.__name__}"

    @cached_property
    def gufunc(self):
        gufunc_sig = gufunc_string_signature(self.signature[0])
        vectorize = numba.guvectorize(
            self.signature, gufunc_sig, nopython=True, cache=True
        )
        return vectorize(self.func)

    def __call__(self, arr: np.ndarray, *, limit: None | int, axis=-1):
        if limit is None:
            limit = arr.shape[axis]
        if limit < 0:
            raise ValueError(f"limit must be positive: {limit}")
        return self.gufunc(arr, limit, axis=axis)


class NumbaGroupNDReduce:
    def __init__(
        self,
        func,
        signature: list[tuple] | None = None,
        *,
        supports_nd=True,
        supports_bool=True,
        supports_ints=True,
    ):
        self.func = func
        self.supports_nd = supports_nd
        self.supports_bool = supports_bool
        self.supports_ints = supports_ints

        if signature is None:
            signature = [
                (numba.float32, numba.int32, numba.float32),
                (numba.float32, numba.int64, numba.float32),
                (numba.float64, numba.int32, numba.float64),
                (numba.float64, numba.int64, numba.float64),
            ]

            if supports_ints:
                signature += [
                    (numba.int32, numba.int32, numba.int32),
                    (numba.int32, numba.int64, numba.int32),
                    (numba.int64, numba.int32, numba.int64),
                    (numba.int64, numba.int64, numba.int64),
                ]

        for sig in signature:
            if not isinstance(sig, tuple):
                raise TypeError(f"signatures for ndmoving must be tuples: {signature}")
            if len(sig) != 3:
                raise TypeError(
                    "signature has wrong number of argument != 3: " f"{signature}"
                )
            if any(ndim(arg) != 0 for arg in sig):
                raise ValueError(
                    "all arguments in signature for ndreduce must be scalars: "
                    f" {signature}"
                )
        self.signature = signature

    @property
    def __name__(self):
        return self.func.__name__

    def __repr__(self):
        return f"numbagg.{self.__name__}"

    @cache
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
        gufunc_sig = ",".join(2 * [_gufunc_arg_str(first_sig[0])]) + ",(z)"
        vectorize = numba.guvectorize(numba_sig, gufunc_sig, nopython=True, cache=True)
        return vectorize(self.func)

    def __call__(self, values, labels, axis=None, num_labels=None):
        values = np.asarray(values)
        labels = np.asarray(labels)
        if not self.supports_nd and (values.ndim != 1 or labels.ndim != 1):
            # TODO: it might be possible to allow returning an extra dimension for the
            # indices by using the technique at
            # https://stackoverflow.com/a/66372474/3064736. Or we could have the numba
            # function return indices for the flattened array, and we stack them into nd
            # indices.
            raise ValueError(
                f"values and labels must be 1-dimensional for {self.func.__name__}. "
                f"Arguments had {values.ndim} & {labels.ndim} dimensions. "
                "Please raise an issue if this feature would be particularly helpful."
            )

        if values.dtype == np.bool_:
            values = values.astype(np.int32)

        if num_labels is None:
            num_labels = np.max(labels) + 1

        if axis is None:
            if values.shape != labels.shape:
                raise ValueError(
                    "axis required if values and labels have different "
                    f"shapes: {values.shape} vs {labels.shape}"
                )
            gufunc = self._create_gufunc(values.ndim)
        elif isinstance(axis, int):
            if labels.shape != (values.shape[axis],):
                raise ValueError(
                    "values must have same shape along axis as labels: "
                    f"{(values.shape[axis],)} vs {labels.shape}"
                )
            values = np.moveaxis(values, axis, -1)
            gufunc = self._create_gufunc(1)
        else:
            values_shape = tuple(values.shape[ax] for ax in axis)
            if labels.shape != values_shape:
                raise ValueError(
                    "values must have same shape along axis as labels: "
                    f"{values_shape} vs {labels.shape}"
                )
            values = np.moveaxis(values, axis, range(-len(axis), 0, 1))
            gufunc = self._create_gufunc(len(axis))

        broadcast_ndim = values.ndim - labels.ndim
        broadcast_shape = values.shape[:broadcast_ndim]
        # Different functions optimize with different inits — e.g. `sum` uses 0, while
        # `prod` uses 1. So we don't initialize here, and instead rely on the function to do so.
        result = np.empty(broadcast_shape + (num_labels,), values.dtype)
        gufunc(values, labels, result)
        return result
