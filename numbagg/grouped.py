import numpy as np

from .decorators import groupndreduce


@groupndreduce.wrap()
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


@groupndreduce.wrap()
def group_nansum(values, labels, out):
    out[:] = 0
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue

        value = values[indices]
        if not np.isnan(value):
            out[label] += value


@groupndreduce.wrap()
def group_nancount(values, labels, out):
    out[:] = 0
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]):
            out[label] += 1


@groupndreduce.wrap()
def group_nanargmax(values, labels, out):
    max_values = np.full(out.shape, np.nan)
    for i in range(len(values)):
        value = values[i]
        label = labels[i]
        if label < 0:
            continue
        # Check for nan in the max_values to make sure we're updating it for the first time
        if not np.isnan(value) and (
            np.isnan(max_values[label]) or value > max_values[label]
        ):
            max_values[label] = value
            out[label] = i
    # If the max value for any label is still NaN (no valid data points), set it to NaN
    # We could instead set the array at the start to be `NaN` — would need to benchmark
    # which is faster.

    # I'm quite confused why, but this raises a warning, so we do the full_loop instead.
    #
    #   out[np.isnan(max_values)] = np.nan
    for i in range(len(out)):
        if np.isnan(max_values[i]):
            out[i] = np.nan


@groupndreduce.wrap()
def group_nanargmin(values, labels, out):
    # Comments from `group_nanargmax` apply here too
    min_values = np.full(out.shape, np.nan)
    for i in range(len(values.flat)):
        value = values[i]
        label = labels[i]
        if label < 0:
            continue
        if not np.isnan(value) and (
            np.isnan(min_values[label]) or value < min_values[label]
        ):
            min_values[label] = value
            out[label] = i
    for idx in np.ndindex(out.shape):
        if np.isnan(min_values[idx]):
            out[idx] = np.nan


@groupndreduce.wrap()
def group_nanfirst(values, labels, out):
    # Slightly inefficient for floats, for which we could avoid allocating the
    # `have_seen_values` array, and instead use an array with NaNs from the start. We
    # could write separate routines, though I don't think we can use `@overload` with
    # out own gufuncs.
    have_seen_value = np.zeros(out.shape, dtype=np.bool_)
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not have_seen_value[label] and not np.isnan(values[indices]):
            have_seen_value[label] = True
            out[label] = values[indices]
    if out.dtype.kind == "f":
        out[~have_seen_value] = np.nan


@groupndreduce.wrap()
def group_nanlast(values, labels, out):
    out[:] = np.nan
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]):
            out[label] = values[indices]


@groupndreduce.wrap()
def group_nanprod(values, labels, out):
    out[:] = 1
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]):
            out[label] *= values[indices]


@groupndreduce.wrap()
def group_nansum_of_squares(values, labels, out):
    out[:] = 0
    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        if not np.isnan(values[indices]):
            out[label] += values[indices] ** 2


@groupndreduce.wrap(supports_bool=False, supports_ints=False, supports_ddof=True)
def group_nanvar(values, labels, ddof, out):
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

    # Calculate for each group
    for label in range(len(out)):
        count = counts[label]
        denom = count - ddof
        if denom <= 0:
            out[label] = np.nan
        else:
            out[label] = (sums_of_squares[label] - (sums[label] ** 2 / count)) / denom


@groupndreduce.wrap(supports_bool=False, supports_ints=False, supports_ddof=True)
def group_nanstd(values, labels, ddof, out):
    # Copy-pasted from `group_nanvar`
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

    for label in range(len(out)):
        count = counts[label]
        denom = count - ddof
        if denom <= 0:
            out[label] = np.nan
        else:
            out[label] = np.sqrt(
                (sums_of_squares[label] - (sums[label] ** 2 / count)) / denom
            )


@groupndreduce.wrap()
def group_nanmin(values, labels, out):
    # Floats could save an allocation by writing directly to `out`
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


@groupndreduce.wrap()
def group_nanmax(values, labels, out):
    # Floats could save an allocation by writing directly to `out`
    max_values = np.full(out.shape, np.nan)

    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        value = values[indices]
        if not np.isnan(value) and (
            np.isnan(max_values[label]) or value > max_values[label]
        ):
            max_values[label] = value

    out[:] = max_values


@groupndreduce.wrap()
def group_nanany(values, labels, out):
    out[:] = 0  # assuming 0 is 'False' for the given dtype

    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        value = values[indices]
        if not np.isnan(value) and value:
            out[label] = 1


@groupndreduce.wrap()
def group_nanall(values, labels, out):
    out[:] = 1  # assuming 1 is 'True' for the given dtype

    for indices in np.ndindex(values.shape):
        label = labels[indices]
        if label < 0:
            continue
        value = values[indices]
        if not np.isnan(value) and not value:
            out[label] = 0
