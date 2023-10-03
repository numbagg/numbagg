import numpy as np
from numba import float32, float64, int32, int64

from .decorators import groupndreduce

dtypes = [
    (int32, int32, int32),
    (int32, int64, int32),
    (int64, int32, int64),
    (int64, int64, int64),
    (float32, int32, float32),
    (float32, int64, float32),
    (float64, int32, float64),
    (float64, int64, float64),
]


@groupndreduce(dtypes)
def group_nanmean(values, labels, out):
    counts = np.zeros(out.shape, dtype=labels.dtype)
    out[:] = 0.0

    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue

        value = values[indices]
        if not np.isnan(value):
            counts[label] += 1
            out[label] += value

    for label in range(len(out)):
        count = counts[label]
        if count == 0:
            out[label] = np.nan
        else:
            out[label] /= count


@groupndreduce(dtypes)
def group_nansum(values, labels, out):
    out[:] = 0
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue

        value = values[indices]
        if not np.isnan(value):
            out[label] += value


@groupndreduce(dtypes)
def group_nancount(values, labels, out):
    out[:] = 0
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]):
            out[label] += 1


@groupndreduce(dtypes)
def group_nanargmax(values, labels, out):
    max_values = np.full(out.shape, np.nan)
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        # Check for nan in the max_values to make sure we're updating it for the first time
        if not np.isnan(values[indices]) and (
            np.isnan(max_values[label]) or values[indices] > max_values[label]
        ):
            max_values[label] = values[indices]
            out[label] = indices[0]
    # If the max value for any label is still NaN (no valid data points), set it to NaN
    # We could instead set the array at the start to be `NaN` — would need to benchmark
    # which is faster.
    out[np.isnan(max_values)] = np.nan


@groupndreduce(dtypes)
def group_nanargmin(values, labels, out):
    min_values = np.full(out.shape, np.nan)
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        # Check for nan in the min_values to make sure we're updating it for the first time
        if not np.isnan(values[indices]) and (
            np.isnan(min_values[label]) or values[indices] < min_values[label]
        ):
            min_values[label] = values[indices]
            out[label] = indices[-1]
    # If the min value for any label is still NaN (no valid data points), set it to NaN
    out[np.isnan(min_values)] = np.nan


@groupndreduce(dtypes)
def group_nanfirst(values, labels, out):
    # Slightly inefficient for floats, which we could fill with NaNs at the start. We
    # could write separate routines.
    out[:] = -1
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if out[label] == -1 and not np.isnan(values[indices]):
            out[label] = values[indices]
    if out.dtype.kind == "f":
        out[out == -1] = np.nan


@groupndreduce(dtypes)
def group_nanlast(values, labels, out):
    out[:] = np.nan
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]):
            out[label] = values[indices]


@groupndreduce(dtypes)
def group_nanprod(values, labels, out):
    out[:] = 1
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]):
            out[label] *= values[indices]


@groupndreduce(dtypes)
def group_nansum_of_squares(values, labels, out):
    out[:] = 0
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]):
            out[label] += values[indices] ** 2


@groupndreduce(dtypes)
def group_nanstd(values, labels, out):
    sums = np.zeros(out.shape, dtype=values.dtype)
    sums_of_squares = np.zeros(out.shape, dtype=values.dtype)
    counts = np.zeros(out.shape, dtype=labels.dtype)
    out[:] = np.nan

    # Calculate sums, sum of squares, and counts
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue

        value = values[indices]
        if not np.isnan(value):
            counts[label] += 1
            sums[label] += value
            sums_of_squares[label] += value**2

    # Calculate standard deviation for each group
    for label in range(len(out)):
        count = counts[label]
        if count < 2:  # not enough data for std deviation
            out[label] = np.nan
        else:
            out[label] = np.sqrt(
                (sums_of_squares[label] - (sums[label] ** 2 / count)) / (count - 1)
            )


@groupndreduce(dtypes)
def group_nanmin(values, labels, out):
    # Floats could save an allocation by writing directly to `out`
    # Though weirdly it works OK for `nanmax`? Copying exactly the same function and
    # changing the sign causes a failure for int32s
    min_values = np.full(out.shape, np.nan)

    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        value = values[indices]
        if not np.isnan(value) and (
            np.isnan(min_values[label]) or value < min_values[label]
        ):
            min_values[label] = value

    out[:] = min_values


@groupndreduce(dtypes)
def group_nanmax(values, labels, out):
    out[:] = np.nan

    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        value = values[indices]
        if not np.isnan(value) and (np.isnan(out[label]) or value > out[label]):
            out[label] = value
