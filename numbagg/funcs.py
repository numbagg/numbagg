import numpy as np

from .decorators import ndreduce


@ndreduce
def nansum(a):
    asum = 0.0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
    return asum


@ndreduce
def nanmean(a):
    asum = 0.0
    count = 0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > 0:
        return asum / count
    else:
        return np.nan


@ndreduce
def nanmin(a):
    amin = np.infty
    allnan = 1
    for ai in a.flat:
        if ai <= amin:
            amin = ai
            allnan = 0
    if allnan:
        amin = np.nan
    return amin


@ndreduce(['float32->int64', 'float64->int64'])
def count(a):
    non_missing = 0
    for ai in a.flat:
        if not np.isnan(ai):
            non_missing += 1
    return non_missing
