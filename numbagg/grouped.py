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
