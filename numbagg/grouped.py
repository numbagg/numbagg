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
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue

        value = values[indices]
        if not np.isnan(value):
            out[label] += value


@groupndreduce(dtypes)
def group_nancount(values, labels, out):
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
    first_indices = np.full(out.shape, -1)
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]) and first_indices[label] == -1:
            out[label] = values[indices]
            first_indices[label] = 0
    # If no non-NaN value is found for any group, set its result to NaN
    out[first_indices == -1] = np.nan


@groupndreduce(dtypes)
def group_nanlast(values, labels, out):
    last_indices = np.full(out.shape, -1)
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]):
            out[label] = values[indices]
            last_indices[label] = 0
    # If no non-NaN value is found for any group, set its result to NaN
    out[last_indices == -1] = np.nan


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
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]):
            out[label] += values[indices] ** 2
