from numba import float64, float32, int64, int32
import numpy as np


from .decorators import groupndreduce


@groupndreduce([
    (float64, int64, float64),
    (float64, int32, float64),
    (float32, int64, float32),
    (float32, int32, float32),
])
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
