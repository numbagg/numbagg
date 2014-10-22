import numpy as np

from .decorators import ndreduce


@ndreduce(['float32,bool_', 'float64,bool_'])
def allnan(a):
    f = 1
    for ai in a.flat:
        if not np.isnan(ai):
            f = 0
            break
    return f


@ndreduce(['float32,bool_', 'float64,bool_'])
def anynan(a):
    f = 0
    for ai in a.flat:
        if np.isnan(ai):
            f = 1
            break
    return f


@ndreduce(['float32,int64', 'float64,int64'])
def count(a):
    non_missing = 0
    for ai in a.flat:
        if not np.isnan(ai):
            non_missing += 1
    return non_missing


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


@ndreduce(['float32,int64', 'float64,int64'])
def nanargmax(a):
    amax = -np.infty
    idx = 0
    for i, ai in enumerate(a.flat):
        if ai > amax:
            amax = ai
            idx = i
    return idx


@ndreduce(['float32,int64', 'float64,int64'])
def nanargmin(a):
    amin = np.infty
    idx = 0
    for i, ai in enumerate(a.flat):
        if ai < amin:
            amin = ai
            idx = i
    return idx


@ndreduce
def nanmax(a):
    amax = -np.infty
    allnan = 1
    for ai in a.flat:
        if ai >= amax:
            amax = ai
            allnan = 0
    if allnan:
        amax = np.nan
    return amax


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
