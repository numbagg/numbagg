from __future__ import annotations

import abc
import functools
import logging
import os
import threading
import warnings
from collections.abc import Callable, Iterable
from functools import cache, cached_property
from typing import Any, Literal, TypeVar

import numba
import numpy as np
from numba.core.types import Type
from numba.np.ufunc.gufunc import GUFunc
from numpy.typing import NDArray

from numbagg.utils import (
    FloatArray,
    NumbaTypes,
    Targets,
    move_axes,
)

from .transform import rewrite_ndreduce

logger = logging.getLogger(__name__)


def _set_fast_math() -> set[str] | bool:
    """
    If "NUMBAGG_FASTMATH" is set to True, enable fastmath optimizations.\n
    We exclude the "no nans" and "no infs" flags.\n
    see https://llvm.org/docs/LangRef.html#fast-math-flags
    """
    if os.getenv("NUMBAGG_FASTMATH", "False").lower() in ("true", "1", "t"):
        warnings.warn(
            "Fastmath optimizations are enabled in numbagg. "
            "This may result in different results than numpy due to reduced precision.",
            UserWarning,
        )
        return {"nsz", "arcp", "contract", "afn", "reassoc"}
    else:
        return False


def _set_cache() -> bool:
    """https://github.com/numba/numba/issues/4807"""
    if os.getenv("NUMBAGG_CACHE", "False").lower() in ("true", "1", "t"):
        warnings.warn(
            "Numba caching is enabled in numbagg. "
            "This will likely cause segfaults when used with multiprocessing. "
            "See https://github.com/numba/numba/issues/4807",
            UserWarning,
        )
        return True
    else:
        return False


_ALPHABET = "abcdefghijkmnopqrstuvwxyz"
_FASTMATH = _set_fast_math()
_ENABLE_CACHE = _set_cache()


def ndim(arg: Type) -> int:
    return getattr(arg, "ndim", 0)


def _gufunc_arg_str(arg: Type) -> str:
    return f"({','.join(_ALPHABET[: ndim(arg)])})"


def gufunc_string_signature(
    numba_args: NumbaTypes, *, returns_scalar: bool = False
) -> str:
    """Convert a tuple of numba types into a numpy gufunc signature.

    The last type is used as output argument.

    Example:

    >>> from numba import float64
    >>> gufunc_string_signature((float64[:], float64))
    '(a)->()'
    """
    return (
        ",".join(map(_gufunc_arg_str, numba_args[:-1]))
        + "->"
        + ("()" if returns_scalar else _gufunc_arg_str(numba_args[-1]))
    )


T = TypeVar("T", bound="NumbaBase")
A = TypeVar("A", bound=FloatArray)


class NumbaBase:
    func: Callable[..., Any]
    signature: Any

    def __init__(
        self, func: Callable[..., Any], supports_parallel: bool = True
    ) -> None:
        self.func = func

        self.cache: bool = _ENABLE_CACHE
        self.supports_parallel: bool = supports_parallel
        self._target_cpu: bool = not supports_parallel
        functools.wraps(func)(self)

    def __repr__(self) -> str:
        return f"numbagg.{self.__name__}"  # type: ignore[attr-defined]

    @classmethod
    def wrap(cls: type[T], *args, **kwargs) -> Callable[..., T]:
        """
        Decorate a function
        """
        return lambda func: cls(func, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def target(self) -> Targets:
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
    def gufunc(self, *, target: Targets) -> GUFunc:
        gufunc_sig: str = gufunc_string_signature(self.signature[0])
        vectorize: Callable[..., GUFunc | Any] = numba.guvectorize(
            self.signature,
            gufunc_sig,
            nopython=True,
            target=target,
            cache=self.cache,
            fastmath=_FASTMATH,
        )
        return vectorize(self.func)


class NumbaBaseSimple(NumbaBase, metaclass=abc.ABCMeta):
    """
    Decorators which don't do any rewriting and all operands share core dims (all except
    the reduction functions + quantiles)
    """

    signature: list[NumbaTypes]

    def __init__(
        self,
        func: Callable[..., Any],
        signature: list[NumbaTypes],
        supports_parallel: bool = True,
    ) -> None:
        for sig in signature:
            if not isinstance(sig, tuple):
                raise TypeError(
                    f"signatures for {self.__class__} must be tuples: {signature}"
                )
        self.signature = signature
        super().__init__(func=func, supports_parallel=supports_parallel)


class ndaggregate(NumbaBaseSimple):
    """
    The decorator for simple aggregations.

    This is the "new" form of `ndreduce`.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        signature: list[NumbaTypes],
        supports_parallel: bool = True,
        supports_ddof: bool = False,
    ) -> None:
        self.supports_ddof: bool = supports_ddof
        super().__init__(func, signature, supports_parallel)

    def _optimize_axis_order(
        self, arr: np.ndarray, axes: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Optimize the order of axes to reduce for better memory access patterns.

        Uses stride-based ordering which works for all array layouts:
        C-contiguous, F-contiguous, transposed, sliced, or any view.
        """
        if len(axes) <= 1:
            return axes

        # Get strides for the specified axes
        axis_strides = [arr.strides[ax] for ax in axes]

        # Sort axes by stride size (largest first)
        # Larger strides = bigger jumps in memory = process first for better cache usage
        sorted_indices = np.argsort(axis_strides)[::-1]
        return tuple(axes[i] for i in sorted_indices)

    def __call__(
        self,
        *arrays: FloatArray,
        ddof: int = 1,
        axis: int | tuple[int, ...] | None = None,
    ):
        if axis is None:
            axis = tuple(range(arrays[0].ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)

        # Optimize axis order based on memory layout for better performance
        axis = self._optimize_axis_order(arrays[0], axis)

        if not all(isinstance(a, np.ndarray) for a in arrays):
            raise TypeError(
                f"All positional arguments to {self} must be arrays: {arrays}"
            )

        arrays = tuple(move_axes(a, axis) for a in arrays)

        if self.supports_ddof:
            return self.gufunc(target=self.target)(*arrays, ddof, axis=-1)
        else:
            return self.gufunc(target=self.target)(*arrays, axis=-1)

    @cache
    def gufunc(self, *, target: Targets):
        # The difference from the parent is `returns_scalar=True`. This is not elegant,
        # but we'll move to dynamic signatures once numba supports them.
        gufunc_sig: str = gufunc_string_signature(
            self.signature[0], returns_scalar=True
        )
        vectorize: Callable[..., GUFunc | Any] = numba.guvectorize(
            self.signature,
            gufunc_sig,
            nopython=True,
            target=target,
            cache=self.cache,
            fastmath=_FASTMATH,
        )
        return vectorize(self.func)


class ndmove(NumbaBaseSimple):
    """Create an N-dimensional moving window function along one dimension.

    Functions should accept arguments for the input array, a window
    size and the output array.

    For example, to write a simplified (and naively implemented) moving window
    sum::

        @ndmove([
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
        func: Callable[..., Any],
        signature: list[NumbaTypes] = [
            (numba.float32[:], numba.int32, numba.float32[:]),
            (numba.float64[:], numba.int64, numba.float64[:]),
        ],
        **kwargs: Any,
    ):
        super().__init__(func, signature, **kwargs)

    def __call__(
        self,
        *arr: FloatArray,
        window: int,
        min_count: int | None = None,
        axis: int | tuple[int, ...] = -1,
        **kwargs,
    ) -> FloatArray:
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


class ndmoveexp(NumbaBaseSimple):
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
        func: Callable[..., Any],
        signature: list[NumbaTypes] = [
            (numba.float64[:], numba.int64, numba.float64[:]),
            (numba.float32[:], numba.int32, numba.float32[:]),
        ],
        **kwargs: Any,
    ):
        super().__init__(func, signature, **kwargs)

    def __call__(
        self,
        *arr: FloatArray,
        alpha: float | FloatArray,
        min_weight: float = 0,
        axis: int = -1,
        **kwargs,
    ) -> FloatArray:
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
        func: Callable[..., Any],
        signature: list[NumbaTypes] = [
            (numba.float32[:], numba.int32, numba.float32[:]),
            (numba.float64[:], numba.int64, numba.float64[:]),
        ],
        **kwargs,
    ) -> None:
        super().__init__(func, signature, **kwargs)

    def __call__(
        self,
        arr: A,
        *,
        limit: None | int = None,
        axis: int = -1,
        **kwargs,
    ) -> A:
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
        func: Callable[..., Any],
        *,
        supports_ddof: bool = False,
        supports_bool: bool = True,
        supports_ints: bool = True,
    ) -> None:
        self.supports_bool: bool = supports_bool
        self.supports_ints: bool = supports_ints
        self.supports_ddof: bool = supports_ddof
        self.func = func
        super().__init__(func=func)

    @cache
    def gufunc(self, core_ndim, values_dtype, labels_dtype, *, target: Targets):
        values_type = numba.from_dtype(values_dtype)
        labels_type = numba.from_dtype(labels_dtype)

        slices = (slice(None),) * core_ndim
        if self.supports_ddof:
            numba_sig: list[NumbaTypes] = [
                (values_type[slices], labels_type[slices], numba.int64, values_type[:])
            ]
            gufunc_sig = f"{','.join(2 * [_gufunc_arg_str(numba_sig[0][0])])},(),(z)"
        else:
            numba_sig = [(values_type[slices], labels_type[slices], values_type[:])]
            gufunc_sig = f"{','.join(2 * [_gufunc_arg_str(numba_sig[0][0])])},(z)"

        vectorize: Callable[..., GUFunc | Any] = numba.guvectorize(
            numba_sig,
            gufunc_sig,
            nopython=True,
            target=target,
            cache=self.cache,
            fastmath=_FASTMATH,
        )
        return vectorize(self.func)

    def __call__(
        self,
        values: NDArray[Any],
        labels: NDArray[Any],
        *,
        ddof: int = 1,
        num_labels: int | None = None,
        axis: int | tuple[int, ...] | None = None,
    ):
        values = np.asarray(values)
        labels = np.asarray(labels)

        if labels.dtype.kind not in "i":
            raise TypeError(
                "labels must be an integer array; it's expected to have already been factorized with a function such as `pd.factorize`"
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

        # Use a float type. But TODO: I'm not confident when exactly numba will coerce
        # vs. raise an error. If this is important we should decide + add tests
        # (currently tests skip these cases, and IIUC the behavior changed when we added
        # our own pre-type caching).
        if (not self.supports_ints and np.issubdtype(values.dtype, np.integer)) or (
            not self.supports_bool and values.dtype == np.bool_
        ):
            values_dtype = values.dtype
            result_dtype: np.dtype = np.dtype(np.float64)
        else:
            values_dtype = values.dtype
            result_dtype = values.dtype

        if axis is None:
            if values.shape != labels.shape:
                raise ValueError(
                    "axis required if values and labels have different "
                    f"shapes: {values.shape} vs {labels.shape}"
                )
            gufunc = self.gufunc(
                core_ndim=values.ndim,
                values_dtype=values_dtype,
                labels_dtype=labels.dtype,
                target=target,
            )
        elif isinstance(axis, int):
            if labels.shape != (values.shape[axis],):
                raise ValueError(
                    "values must have same shape along axis as labels: "
                    f"{(values.shape[axis],)} vs {labels.shape}"
                )
            values = np.moveaxis(values, axis, -1)
            gufunc = self.gufunc(
                core_ndim=1,
                values_dtype=values_dtype,
                labels_dtype=labels.dtype,
                target=target,
            )
        else:
            values_shape = tuple(values.shape[ax] for ax in axis)
            if labels.shape != values_shape:
                raise ValueError(
                    "values must have same shape along axis as labels: "
                    f"{values_shape} vs {labels.shape}"
                )
            values = np.moveaxis(values, axis, range(-len(axis), 0, 1))
            gufunc = self.gufunc(
                core_ndim=len(axis),
                values_dtype=values_dtype,
                labels_dtype=labels.dtype,
                target=target,
            )

        broadcast_ndim: int = values.ndim - labels.ndim
        broadcast_shape: tuple[int, ...] = values.shape[:broadcast_ndim]
        # Different functions initialize with different values — e.g. `sum` uses 0,
        # while `prod` uses 1. So we don't initialize with a value here, and instead
        # rely on the function to do so.
        result = np.empty(broadcast_shape + (num_labels,), result_dtype)
        args: tuple = (values, labels)

        if self.supports_ddof:
            args += (ddof,)
        args += (result,)

        gufunc(*args)
        return result


class ndmatrix(NumbaBase):
    """
    Decorator for functions that produce matrix outputs.

    These functions take an array and produce a square matrix output
    (e.g., correlation matrix, covariance matrix).

    Broadcasting and Dimension Conventions:
    - Core dimensions: `(n, m) -> (n, n)` where n=variables, m=observations
    - Conceptual: The observations dimension (last) gets reduced through aggregation,
      and an additional variables dimension is added at the end to form the n×n matrix
    - Broadcasting: Works with arbitrary leading dimensions

    Examples:
    - 2D input `(3, 100)` -> output `(3, 3)`
    - 3D input `(batch=2, vars=3, obs=100)` -> output `(2, 3, 3)`
    - 4D input `(2, 5, 3, 100)` -> output `(2, 5, 3, 3)`

    This provides an advantage over NumPy's `corrcoef`/`cov` which only
    support 2D input. NumBagg functions broadcast over arbitrary leading dimensions,
    allowing efficient computation on batched data.
    """

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
        **kwargs,
    ):
        # Require at least 2D input
        if a.ndim < 2:
            raise ValueError(
                f"{self.func.__name__} requires at least a 2D array with shape (..., vars, obs). "
                "For 1D arrays, use nanvar for variance calculations."
            )

        # Static matrix functions use fixed convention: (..., vars, obs) -> (..., vars, vars)
        # No axis parameter - dimensions are fixed for consistency
        # vars_axis=-2, obs_axis=-1 (obs gets reduced by gufunc)

        gufunc = self.gufunc(target=self.target)
        # axes specifies which axes contain the core dimensions
        # For our signature "(vars,obs)->(vars,vars)":
        # - Input has 2 core dims: second-to-last (vars) and last (obs)
        # - Output has 2 core dims: last two dimensions (vars,vars)
        result = gufunc(a, axes=[(-2, -1), (-2, -1)], **kwargs)

        return result

    @cache
    def gufunc(self, *, target):
        vectorize = numba.guvectorize(
            *self.signature,
            nopython=True,
            target=target,
            cache=self.cache,
            fastmath=_FASTMATH,
        )
        return vectorize(self.func)


class ndmovematrix(NumbaBase):
    """Create moving window matrix functions.

    These functions take a 2D array and produce a 3D array of matrices
    for each window position (e.g., moving correlation/covariance matrices).

    Broadcasting and Dimension Conventions:
    - Core dimensions: `(n, m), (), () -> (m, n, n)` where n=variables, m=observations
    - Conceptual: The observations dimension is preserved and becomes the time axis,
      with n×n variable matrices added at the end for each time point
    - Broadcasting: Works with arbitrary leading dimensions

    Examples:
    - 2D input `(3, 100)` -> output `(100, 3, 3)` - matrix at each time
    - 3D input `(batch=2, vars=3, obs=100)` -> output `(2, 100, 3, 3)`
    - 4D input `(2, 5, 3, 100)` -> output `(2, 5, 100, 3, 3)`

    Each time step contains a matrix computed from the rolling window ending at that time.
    """

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
        window: int,
        min_count: int | None = None,
        **kwargs,
    ):
        a = np.asarray(a)

        if a.ndim < 2:
            raise ValueError(
                f"{self.func.__name__} requires at least a 2D array with shape (..., obs, vars)."
            )

        if min_count is None:
            min_count = window
        elif min_count < 0:
            raise ValueError(f"min_count must be positive: {min_count}")

        # Moving matrix functions use fixed convention: (..., obs, vars) -> (..., obs, vars, vars)
        # No axis parameter - dimensions are fixed for consistency
        # obs_axis=-2 (preserved as time dimension), vars_axis=-1 (duplicated to matrix dims)

        # Check window size against observations dimension (second-to-last)
        if not 0 < window <= a.shape[-2]:
            raise ValueError(f"window not in valid range: {window}")

        gufunc = self.gufunc(target=self.target)
        return gufunc(a, window, min_count, **kwargs)

    @cache
    def gufunc(self, *, target):
        vectorize = numba.guvectorize(
            *self.signature,
            nopython=True,
            target=target,
            cache=self.cache,
            fastmath=_FASTMATH,
        )
        return vectorize(self.func)


class ndquantile(NumbaBase):
    def __init__(
        self,
        func: Callable[..., Any],
        signature: tuple[NumbaTypes, str],
        **kwargs,
    ) -> None:
        self.signature = signature
        super().__init__(func, **kwargs)

    def __call__(
        self,
        a: NDArray[np.float64],
        quantiles: float | Iterable[float],
        axis: int | tuple[int, ...] | None = None,
        **kwargs,
    ) -> NDArray[np.float64]:
        # Gufunc doesn't support a 0-len dimension for quantiles, so we need to make and
        # then remove a dummy axis.
        if not isinstance(quantiles, Iterable):
            squeeze = True
            quantiles = [quantiles]
        else:
            squeeze = False
        quantiles = np.asarray(quantiles)

        if any(quantiles < 0) or any(quantiles > 1):
            raise ValueError(
                f"quantiles must be in the range [0, 1], inclusive. Got {quantiles}."
            )

        if axis is None:
            axis = tuple(range(a.ndim))
        elif not isinstance(axis, Iterable):
            axis = (axis,)

        a = move_axes(a, axis)

        # - 1st array is our input; we've moved the axes to the final axis.
        # - 2nd is the quantiles array, and is always only a single axis.
        # - 3rd array is the result array, and returns a final axis for quantiles.
        axes: list[int] = [-1, -1, -1]

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
    def gufunc(self, *, target: Targets):
        # We don't use `NumbaBaseSimple`'s here, because we need to specify different
        # core axes for the two inputs, which it doesn't support.

        vectorize: Callable[..., GUFunc | Any] = numba.guvectorize(
            # For nanquantile, `self.signature` is a tuple of both the "`float64`" and
            # the "`(n),(m)->(m)`" parts, because it has different core dims for its
            # operands, so doesn't work with the standard `gufunc_string_signature`
            # function.
            *self.signature,
            nopython=True,
            target=target,
            cache=self.cache,
            fastmath=_FASTMATH,
        )
        return vectorize(self.func)


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

    This is an "old" decorator, which has the advantage of never copying the array, even
    when the data is not contiguous. But it adds lots more complication, and the current
    implementation restricts any additional scalar parameters, such as `ddof`. The "new"
    version of this is `ndaggregate`. More details at
    https://github.com/numbagg/numbagg/issues/218.
    """

    def __init__(self, func: Callable[..., Any], signature, **kwargs) -> None:
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
    def gufunc(self, core_ndim, *, target: Targets):
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
                (
                    first_sig.args[0][(slice(None),) * core_ndim]
                    if core_ndim
                    else first_sig.args[0]
                ),
            )
            + first_sig.args[1:]
            + (first_sig.return_type,)
        )

        # Can't use `cache=True` because of the dynamic ast transformation
        vectorize = numba.guvectorize(
            numba_sig,
            gufunc_sig,
            nopython=True,
            target=target,
            fastmath=_FASTMATH,
        )
        return vectorize(self.transformed_func)

    def __call__(
        self, arr: NDArray[Any], *args, axis: tuple[int, ...] | int | None = None
    ):
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


class ndmoveexpmatrix(NumbaBase):
    """Create exponential moving window matrix functions.

    These functions take a 2D array and produce a 3D array of matrices
    for each time position using exponential decay (e.g., moving correlation/covariance matrices).

    Broadcasting and Dimension Conventions:
    - Core dimensions: `(n, m), (m), () -> (m, n, n)` where n=variables, m=observations
    - Conceptual: The observations dimension is preserved and becomes the time axis,
      with n×n variable matrices added at the end for each time point
    - Broadcasting: Works with arbitrary leading dimensions
    - Alpha parameter: Supports scalar or array broadcasting

    Examples:
    - 2D input `(3, 100)` -> output `(100, 3, 3)` - matrix at each time
    - 3D input `(batch=2, vars=3, obs=100)` -> output `(2, 100, 3, 3)`
    - 4D input `(2, 5, 3, 100)` -> output `(2, 5, 100, 3, 3)`

    Each time step contains a matrix computed using exponentially weighted observations
    up to that time, with more recent observations having higher weight.
    """

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
        alpha: float | np.ndarray,
        min_weight: float = 0,
        **kwargs,
    ):
        a = np.asarray(a)

        if a.ndim < 2:
            raise ValueError(
                f"{self.func.__name__} requires at least a 2D array with shape (..., obs, vars)."
            )

        # Exponential moving matrix functions use fixed convention: (..., obs, vars) -> (..., obs, vars, vars)
        # No axis parameter - dimensions are fixed for consistency
        # obs_axis=-2 (preserved as time dimension), vars_axis=-1 (duplicated to matrix dims)

        # Handle alpha parameter - broadcast to observations dimension (second-to-last)
        if not isinstance(alpha, np.ndarray):
            alpha = np.broadcast_to(alpha, a.shape[-2])  # type: ignore[assignment,unused-ignore]

        gufunc = self.gufunc(target=self.target)
        with np.errstate(invalid="ignore", divide="ignore"):
            return gufunc(a, alpha, min_weight, **kwargs)

    @cache
    def gufunc(self, *, target):
        vectorize = numba.guvectorize(
            *self.signature,
            nopython=True,
            target=target,
            cache=self.cache,
            fastmath=_FASTMATH,
        )
        return vectorize(self.func)


def _is_in_unsafe_thread_pool() -> bool:
    current_thread = threading.current_thread()
    # ThreadPoolExecutor threads typically have names like 'ThreadPoolExecutor-0_1'
    return current_thread.name.startswith(
        "ThreadPoolExecutor"
    ) and _thread_backend() in {"workqueue", None}


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
