from __future__ import annotations

import abc
import itertools
import logging
import threading
from collections.abc import Iterable
from functools import cache, cached_property
from typing import Any, Callable, Literal, TypeVar

import numba
import numpy as np

from .transform import rewrite_ndreduce

logger = logging.getLogger(__name__)


def ndim(arg):
    return getattr(arg, "ndim", 0)


_ALPHABET = "abcdefghijkmnopqrstuvwxyz"


def _gufunc_arg_str(arg):
    return f"({','.join(_ALPHABET[: ndim(arg)])})"


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


T = TypeVar("T", bound="NumbaBase")


class NumbaBase:
    func: Callable
    signature: Any

    def __init__(self, func: Callable, supports_parallel: bool = True):
        self.func = func
        # https://github.com/numba/numba/issues/4807
        self.cache = False
        self.supports_parallel = supports_parallel
        self._target_cpu = not supports_parallel

    @property
    def __name__(self):
        return self.func.__name__

    def __repr__(self):
        return f"numbagg.{self.__name__}"

    @classmethod
    def wrap(cls: type[T], *args, **kwargs) -> Callable[..., T]:
        """
        Decorate a function
        """
        return lambda func: cls(func, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def target(self):
        if self._target_cpu:
            return "cpu"
        else:
            if _is_in_unsafe_thread_pool():
                logger.debug(
                    "Numbagg detected that we're in a thread pool with workqueue threading. "
                    "As a result, we're turning off parallel support to ensure numba doesn't abort. "
                    "This will result in lower performance on parallelizable arrays on multi-core systems. "
                    "To enable parallel support, run outside a multithreading context, or install TBB or OpenMP. "
                    "Numbagg won't re-check on every call — restart your python session to reset the check. "
                    "For more details, check out https://numba.readthedocs.io/en/stable/developer/threading_implementation.html#caveats"
                )
                self._target_cpu = True
                return "cpu"
            else:
                return "parallel"

    @cache
    def gufunc(self, *, target):
        gufunc_sig = gufunc_string_signature(self.signature[0])
        vectorize = numba.guvectorize(
            self.signature,
            gufunc_sig,
            nopython=True,
            target=target,
            cache=self.cache,
        )
        return vectorize(self.func)


class NumbaBaseSimple(NumbaBase, metaclass=abc.ABCMeta):
    """
    Decorators which don't do any rewriting and all operands share core dims (all except
    the reduction functions + quantiles)
    """

    signature: list[tuple]

    def __init__(
        self, func: Callable, signature: list[tuple], supports_parallel: bool = True
    ):
        for sig in signature:
            if not isinstance(sig, tuple):
                raise TypeError(
                    f"signatures for {self.__class__} must be tuples: {signature}"
                )
        self.signature = signature
        super().__init__(func=func, supports_parallel=supports_parallel)


class ndreduce(NumbaBase):
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

    def __init__(self, func, signature, **kwargs):
        self.func = func
        # NDReduce uses different types than the other funcs, and they seem difficult to
        # type, so ignoring for the moment.
        self.signature: Any = signature

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

        super().__init__(func=func, **kwargs)

    @cached_property
    def transformed_func(self):
        return rewrite_ndreduce(self.func)

    @cached_property
    def _jit_func(self):
        vectorize = numba.jit(self.signature, nopython=True)
        return vectorize(self.func)

    @cache
    def gufunc(self, core_ndim, *, target):
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

        # Can't use `cache=True` because of the dynamic ast transformation
        vectorize = numba.guvectorize(
            numba_sig, gufunc_sig, nopython=True, target=target
        )
        return vectorize(self.transformed_func)

    def __call__(self, arr, *args, axis=None):
        # TODO: `nanmin` & `nanmix` raises a warning here for the default test
        # fixture; I can't figure out where it's coming from, and can't reproduce it
        # locally. So I'm ignoring so that we can still raise errors on other
        # warnings.
        if self.func.__name__ in ["nanmin", "nanmax"]:
            warn: Literal["ignore", "warn"] = "ignore"
        else:
            warn = "warn"

        with np.errstate(invalid=warn):
            if axis is None:
                # TODO: switch to using jit_func (it's faster), once numba reliably
                # returns the right dtype
                # see: https://github.com/numba/numba/issues/1087
                # f = self._jit_func
                f = self.gufunc(arr.ndim, target=self.target)
            elif isinstance(axis, int):
                arr = np.moveaxis(arr, axis, -1)
                f = self.gufunc(1, target=self.target)
            else:
                arr = np.moveaxis(arr, axis, range(-len(axis), 0, 1))
                f = self.gufunc(len(axis), target=self.target)
            return f(arr, *args)


class ndmoving(NumbaBaseSimple):
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

    def __init__(
        self,
        func: Callable,
        signature: list[tuple] = [
            (numba.float64[:], numba.int64, numba.float64[:]),
            (numba.float32[:], numba.int32, numba.float32[:]),
        ],
        **kwargs,
    ):
        super().__init__(func, signature, **kwargs)

    def __call__(
        self,
        *arr: np.ndarray,
        window,
        min_count=None,
        axis: int | tuple[int, ...] = -1,
        **kwargs,
    ):
        if min_count is None:
            min_count = window
        elif min_count < 0:
            raise ValueError(f"min_count must be positive: {min_count}")

        # If an empty tuple is passed, there's no reduction to do, so we return the
        # original array.
        # Ref https://github.com/pydata/xarray/pull/5178/files#r616168398
        if isinstance(axis, tuple):
            if axis == ():
                if len(arr) > 1:
                    raise ValueError(
                        "`axis` cannot be an empty tuple when passing more than one array; since we default to returning the input."
                    )
                return arr[0]
            elif len(axis) > 1:
                raise ValueError(
                    f"only one axis can be passed to {self.func}; got {axis}"
                )
            (axis,) = axis
        if not 0 < window <= arr[0].shape[axis]:
            raise ValueError(f"window not in valid range: {window}")
        gufunc = self.gufunc(target=self.target)
        return gufunc(*arr, window, min_count, axis=axis, **kwargs)


class ndmovingexp(NumbaBaseSimple):
    """
    Exponential moving window function.

    `alpha` supports either:
    - a scalar (similar to pandas)
    - a 1D array with the same length as the core dimension on the main array(s)
    - an ND array with the same shape as the main array(s)

    If a scalar is passed, the same alpha is used to decay each value. Passing an array
    allows decaying values different amounts.

    The function is designed for alpha values between 0 and 1 inclusive. A value of 0
    decays the whole value such that `moving_exp_sum` returns the input. A value of 1
    doesn't decay at all, such that `moving_exp_sum` is equivalent to an accumulating
    sum. The function doesn't proactively check for valid values.

    """

    def __init__(
        self,
        func: Callable,
        signature: list[tuple] = [
            (numba.float64[:], numba.int64, numba.float64[:]),
            (numba.float32[:], numba.int32, numba.float32[:]),
        ],
        **kwargs,
    ):
        super().__init__(func, signature, **kwargs)

    def __call__(
        self,
        *arr: np.ndarray,
        alpha: float,
        min_weight: float = 0,
        axis: int = -1,
        **kwargs,
    ):
        if not isinstance(alpha, np.ndarray):
            alpha = np.broadcast_to(alpha, arr[0].shape[axis])  # type: ignore[assignment,unused-ignore]
            alpha_axis = -1
        elif alpha.ndim == 1:
            alpha_axis = -1
        else:
            alpha_axis = axis

        # If an empty tuple is passed, there's no reduction to do, so we return the
        # original array.
        # Ref https://github.com/pydata/xarray/pull/5178/files#r616168398
        if isinstance(axis, tuple):
            if axis == ():
                if len(arr) > 1:
                    raise ValueError(
                        "`axis` cannot be an empty tuple when passing more than one array; since we default to returning the input."
                    )
                return arr[0]
            if len(axis) > 1:
                raise ValueError(
                    f"Only one axis can be passed to {self.func}; got {axis}"
                )
            (axis,) = axis

        # Axes is `axis` for each array (most often just one array), and then either
        # `-1` or `axis` for alphas, depending on whether a full array was passed or not.
        # Then `()` for the min_weight, and `axis` for the output
        axes = [axis for _ in range(len(arr))] + [alpha_axis, (), axis]
        # For the sake of speed, we ignore divide-by-zero and NaN warnings, and test for
        # their correct handling in our tests.
        with np.errstate(invalid="ignore", divide="ignore"):
            gufunc = self.gufunc(target=self.target)
            return gufunc(*arr, alpha, min_weight, axes=axes, **kwargs)


class ndfill(NumbaBaseSimple):
    def __init__(
        self,
        func: Callable,
        signature: list[tuple] = [
            (numba.float64[:], numba.int64, numba.float64[:]),
            (numba.float32[:], numba.int32, numba.float32[:]),
        ],
        **kwargs,
    ):
        super().__init__(func, signature, **kwargs)

    def __call__(
        self,
        arr: np.ndarray,
        *,
        limit: None | int = None,
        axis: int = -1,
        **kwargs,
    ):
        if limit is None:
            limit = arr.shape[axis]
        if limit < 0:
            raise ValueError(f"`limit` must be positive: {limit}")
        gufunc = self.gufunc(target=self.target)
        return gufunc(arr, limit, axis=axis, **kwargs)


class groupndreduce(NumbaBase):
    """Create an N-dimensional grouped aggregation function."""

    def __init__(
        self,
        func,
        signature: list[tuple] | None = None,
        *,
        supports_ddof=False,
        supports_nd=True,
        supports_bool=True,
        supports_ints=True,
    ):
        self.supports_nd = supports_nd
        self.supports_bool = supports_bool
        self.supports_ints = supports_ints
        self.supports_ddof = supports_ddof
        self.func = func

        if signature is None:
            values_dtypes: tuple[numba.dtype, ...] = (numba.float32, numba.float64)
            labels_dtypes = (numba.int8, numba.int16, numba.int32, numba.int64)
            if supports_ints:
                values_dtypes += (numba.int32, numba.int64)

            signature = [
                (value_type, label_type, numba.int64, value_type)
                if supports_ddof
                else (value_type, label_type, value_type)
                for value_type, label_type in itertools.product(
                    values_dtypes, labels_dtypes
                )
            ]
        for sig in signature:
            if not isinstance(sig, tuple):
                raise TypeError(f"signatures for ndmoving must be tuples: {signature}")
            n_args = 3 + supports_ddof
            if len(sig) != n_args:
                raise TypeError(
                    f"signature has wrong number of arguments != {n_args}: {signature}"
                )
            if any(ndim(arg) != 0 for arg in sig):
                raise ValueError(
                    f"all arguments in signature for ndreduce must be scalars: {signature}"
                )

        self.signature = signature

        super().__init__(func=func)

    @cache
    def gufunc(self, core_ndim, *, target):
        # compiling gufuncs has some significant overhead (~130ms per function
        # and number of dimensions to aggregate), so do this in a lazy fashion
        numba_sig: list[tuple] = []
        slices = (slice(None),) * core_ndim
        # This is pretty messy. We could inherit from this class for the `ddof` methods.
        # But probably we want to make it more abstract, and take advantage of
        # forthcoming numba features such as dynamic signatures.
        if self.supports_ddof:
            for input_sig in self.signature:
                values, labels, ddof, out = input_sig
                numba_sig += [(values[slices], labels[slices], ddof, out[:])]
            first_sig = numba_sig[0]
            gufunc_sig = f"{','.join(2 * [_gufunc_arg_str(first_sig[0])])},(),(z)"
        else:
            for input_sig in self.signature:
                values, labels, out = input_sig
                numba_sig += [(values[slices], labels[slices], out[:])]
            first_sig = numba_sig[0]
            gufunc_sig = f"{','.join(2 * [_gufunc_arg_str(first_sig[0])])},(z)"

        vectorize = numba.guvectorize(
            numba_sig,
            gufunc_sig,
            nopython=True,
            target=target,
            cache=self.cache,
        )
        return vectorize(self.func)

    def __call__(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        *,
        ddof=1,
        num_labels: int | None = None,
        axis: int | tuple[int, ...] | None = None,
    ):
        values = np.asarray(values)
        labels = np.asarray(labels)

        if labels.dtype.kind not in "i":
            raise TypeError(
                "labels must be an integer array; it's expected to have already been factorized with a function such as `pd.factorize`"
            )

        # TODO: I think we can remove this, now that every function supports ND...
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

        # We need to be careful that we don't overflow `counts` in the grouping
        # function. So the labels need to be a big enough integer type to hold the
        # maximum possible count, since we generate the counts array based on the labels
        # dtype. (We're over-estimating a bit here, because `values` might be over
        # multiple dimensions, we could refine it down; would need to consider for
        # axis being None or a tuple, though.)
        if np.iinfo(labels.dtype).max < values.size:
            dtype = np.min_scalar_type(values.size)
            logger.debug(
                f"values' size {values.size} is greater than the max of {labels.dtype}. "
                f"We're casting the labels array to a larger dtype {dtype} to avoid the risk of overflow. "
                "It would be possible to implement this differently, so if the copy is a memory or "
                "performance issue, please raise an issue in numbagg, and we can "
                "consider approaches to avoid this."
            )
            labels = labels.astype(dtype)

        if values.dtype == np.bool_:
            values = values.astype(np.int32)

        if num_labels is None:
            num_labels = np.max(labels) + 1

        target = self.target

        if axis is None:
            if values.shape != labels.shape:
                raise ValueError(
                    "axis required if values and labels have different "
                    f"shapes: {values.shape} vs {labels.shape}"
                )
            gufunc = self.gufunc(values.ndim, target=target)
        elif isinstance(axis, int):
            if labels.shape != (values.shape[axis],):
                raise ValueError(
                    "values must have same shape along axis as labels: "
                    f"{(values.shape[axis],)} vs {labels.shape}"
                )
            values = np.moveaxis(values, axis, -1)
            gufunc = self.gufunc(1, target=target)
        else:
            values_shape = tuple(values.shape[ax] for ax in axis)
            if labels.shape != values_shape:
                raise ValueError(
                    "values must have same shape along axis as labels: "
                    f"{values_shape} vs {labels.shape}"
                )
            values = np.moveaxis(values, axis, range(-len(axis), 0, 1))
            gufunc = self.gufunc(len(axis), target=target)

        broadcast_ndim = values.ndim - labels.ndim
        broadcast_shape = values.shape[:broadcast_ndim]
        # Different functions initialize with different values — e.g. `sum` uses 0,
        # while `prod` uses 1. So we don't initialize with a value here, and instead
        # rely on the function to do so.
        result = np.empty(broadcast_shape + (num_labels,), values.dtype)
        args: tuple = (values, labels)

        if self.supports_ddof:
            args += (ddof,)
        args += (result,)
        print(args)

        gufunc(*args)
        return result


def move_axes(arr: np.ndarray, axes: tuple[int, ...]):
    """
    Move & reshape a tuple of axes to an array's final axis.
    """
    moved_arr = np.moveaxis(arr, axes, range(arr.ndim - len(axes), arr.ndim))
    new_shape = moved_arr.shape[: -len(axes)] + (-1,)
    return moved_arr.reshape(new_shape)


class ndquantile(NumbaBase):
    def __init__(
        self,
        func: Callable,
        signature: tuple[list[tuple], str],
        **kwargs,
    ):
        self.signature = signature
        super().__init__(func, **kwargs)

    def __call__(
        self,
        a: np.ndarray,
        quantiles: float | Iterable[float],
        axis: int | tuple[int, ...] | None = None,
        **kwargs,
    ):
        # Gufunc doesn't support a 0-len dimension for quantiles, so we need to make and
        # then remove a dummy axis.
        if not isinstance(quantiles, Iterable):
            squeeze = True
            quantiles = [quantiles]
        else:
            squeeze = False
        quantiles = np.asarray(quantiles)

        if axis is None:
            axis = tuple(range(a.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)

        a = move_axes(a, axis)

        # - 1st array is our input; we've moved the axes to the final axis.
        # - 2nd is the quantiles array, and is always only a single axis.
        # - 3rd array is the result array, and returns a final axis for quantiles.
        axes = [-1, -1, -1]

        gufunc = self.gufunc(target=self.target)
        # TODO: `nanquantile` raises a warning here for the default test
        # fixture; I can't figure out where it's coming from, and can't reproduce it
        # locally. So I'm ignoring so that we can still raise errors on other
        # warnings.
        if self.func.__name__ in ["nanquantile"]:
            warn: Literal["ignore", "warn"] = "ignore"
        else:
            warn = "warn"

        with np.errstate(invalid=warn):
            result = gufunc(a, quantiles, axes=axes, **kwargs)

        # numpy returns quantiles as the first axis, so we move ours to that position too
        result = np.moveaxis(result, -1, 0)
        if squeeze:
            result = result.squeeze(axis=0)

        return result

    @cache
    def gufunc(self, *, target):
        # We don't use `NumbaBaseSimple`'s here, because we need to specify different
        # core axes for the two inputs, which it doesn't support.

        vectorize = numba.guvectorize(
            # For nanquantile, `self.signature` is a tuple of both the "`float64`" and
            # the "`(n),(m)->(m)`" parts, because it has different core dims for its
            # operands, so doesn't work with the standard `gufunc_string_signature`
            # function.
            *self.signature,
            nopython=True,
            target=target,
            cache=self.cache,
        )
        return vectorize(self.func)


def _is_in_unsafe_thread_pool() -> bool:
    current_thread = threading.current_thread()
    # ThreadPoolExecutor threads typically have names like 'ThreadPoolExecutor-0_1'
    return current_thread.name.startswith(
        "ThreadPoolExecutor"
    ) and _thread_backend() in ["workqueue" or None]


@cache
def _thread_backend() -> str | None:
    # Note that `importlib.util.find_spec` doesn't work for these; it will falsely
    # return True

    try:
        from numba.np.ufunc import tbbpool  # noqa

        return "tbb"
    except ImportError:
        pass

    try:
        from numba.np.ufunc import omppool  # noqa

        return "omp"
    except ImportError:
        pass

    return "workqueue"
