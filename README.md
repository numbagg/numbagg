# Numbagg: Fast N-dimensional aggregation functions with Numba

[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/numbagg/numbagg/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/numbagg/numbagg/actions/workflows/test.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/numbagg?style=for-the-badge)](https://pypi.python.org/pypi/numbagg/)

Fast, flexible N-dimensional array functions written with
[Numba](https://github.com/numba/numba) and NumPy's [generalized
ufuncs](http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html).

Currently accelerated functions:

- Array functions: `allnan`, `anynan`, `count`, `nanargmax`,
  `nanargmin`, `nanmax`, `nanmean`, `nanstd`, `nanvar`, `nanmin`,
  `nansum`, `nanquantile`, `ffill`, `bfill`.
- Grouped functions: `group_nanall`, `group_nanany`, `group_nanargmax`,
  `group_nanargmin`, `group_nancount`, `group_nanfirst`, `group_nanlast`,
  `group_nanmax`, `group_nanmean`, `group_nanmin`, `group_nanprod`,
  `group_nanstd`, `group_nansum`, `group_nansum_of_squares`, `group_nanvar`.
- Moving window functions listed below
- Exponentially weighted moving functions listed below

## Benchmarks

| func                | numbagg | pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------ | ------: | -----: | ---------: | -----------: | ---------------: |
| `bfill`             |    58ms |  559ms |       62ms |        9.70x |            1.08x |
| `ffill`             |    52ms |  565ms |       59ms |       10.94x |            1.13x |
| `move_corr`         |   150ms | 3001ms |        n/a |       20.06x |              n/a |
| `move_cov`          |   127ms | 2021ms |        n/a |       15.93x |              n/a |
| `move_mean`         |    93ms |  388ms |       95ms |        4.18x |            1.02x |
| `move_std`          |    73ms |  580ms |      103ms |        7.97x |            1.42x |
| `move_sum`          |    91ms |  384ms |       75ms |        4.22x |            0.82x |
| `move_var`          |    71ms |  548ms |      101ms |        7.74x |            1.42x |
| `move_exp_nancorr`  |   195ms | 1420ms |        n/a |        7.29x |              n/a |
| `move_exp_nancount` |   101ms |  233ms |        n/a |        2.31x |              n/a |
| `move_exp_nancov`   |   134ms |  872ms |        n/a |        6.50x |              n/a |
| `move_exp_nanmean`  |   101ms |  217ms |        n/a |        2.15x |              n/a |
| `move_exp_nanstd`   |   132ms |  287ms |        n/a |        2.17x |              n/a |
| `move_exp_nansum`   |    98ms |  198ms |        n/a |        2.02x |              n/a |
| `move_exp_nanvar`   |   129ms |  249ms |        n/a |        1.94x |              n/a |

<details>
<summary>Full benchmarks</summary>

| func                |     size | numbagg | pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------ | -------: | ------: | -----: | ---------: | -----------: | ---------------: |
| `bfill`             |     1000 |     0ms |    0ms |        0ms |       29.78x |            0.59x |
|                     |   100000 |     1ms |    7ms |        1ms |       10.39x |            0.92x |
|                     | 10000000 |    58ms |  559ms |       62ms |        9.70x |            1.08x |
| `ffill`             |     1000 |     0ms |    0ms |        0ms |       30.66x |            0.39x |
|                     |   100000 |     1ms |    7ms |        1ms |       11.48x |            0.90x |
|                     | 10000000 |    52ms |  565ms |       59ms |       10.94x |            1.13x |
| `move_corr`         |     1000 |     0ms |    1ms |        n/a |      101.29x |              n/a |
|                     |   100000 |     2ms |   31ms |        n/a |       17.88x |              n/a |
|                     | 10000000 |   150ms | 3001ms |        n/a |       20.06x |              n/a |
| `move_cov`          |     1000 |     0ms |    1ms |        n/a |      102.03x |              n/a |
|                     |   100000 |     2ms |   23ms |        n/a |       14.12x |              n/a |
|                     | 10000000 |   127ms | 2021ms |        n/a |       15.93x |              n/a |
| `move_mean`         |     1000 |     0ms |    0ms |        0ms |       29.03x |            0.71x |
|                     |   100000 |     1ms |    3ms |        1ms |        2.69x |            0.63x |
|                     | 10000000 |    93ms |  388ms |       95ms |        4.18x |            1.02x |
| `move_std`          |     1000 |     0ms |    0ms |        0ms |       23.84x |            1.13x |
|                     |   100000 |     1ms |    5ms |        1ms |        5.32x |            1.04x |
|                     | 10000000 |    73ms |  580ms |      103ms |        7.97x |            1.42x |
| `move_sum`          |     1000 |     0ms |    0ms |        0ms |       29.34x |            0.67x |
|                     |   100000 |     1ms |    3ms |        1ms |        2.30x |            0.55x |
|                     | 10000000 |    91ms |  384ms |       75ms |        4.22x |            0.82x |
| `move_var`          |     1000 |     0ms |    0ms |        0ms |       21.37x |            1.12x |
|                     |   100000 |     1ms |    5ms |        1ms |        4.21x |            0.86x |
|                     | 10000000 |    71ms |  548ms |      101ms |        7.74x |            1.42x |
| `move_exp_nancorr`  |     1000 |     0ms |    1ms |        n/a |       33.02x |              n/a |
|                     |   100000 |     2ms |   15ms |        n/a |        6.41x |              n/a |
|                     | 10000000 |   195ms | 1420ms |        n/a |        7.29x |              n/a |
| `move_exp_nancount` |     1000 |     0ms |    0ms |        n/a |       10.89x |              n/a |
|                     |   100000 |     1ms |    2ms |        n/a |        1.95x |              n/a |
|                     | 10000000 |   101ms |  233ms |        n/a |        2.31x |              n/a |
| `move_exp_nancov`   |     1000 |     0ms |    1ms |        n/a |       40.56x |              n/a |
|                     |   100000 |     2ms |   10ms |        n/a |        5.90x |              n/a |
|                     | 10000000 |   134ms |  872ms |        n/a |        6.50x |              n/a |
| `move_exp_nanmean`  |     1000 |     0ms |    0ms |        n/a |        8.35x |              n/a |
|                     |   100000 |     1ms |    2ms |        n/a |        1.86x |              n/a |
|                     | 10000000 |   101ms |  217ms |        n/a |        2.15x |              n/a |
| `move_exp_nanstd`   |     1000 |     0ms |    0ms |        n/a |        9.89x |              n/a |
|                     |   100000 |     2ms |    3ms |        n/a |        1.85x |              n/a |
|                     | 10000000 |   132ms |  287ms |        n/a |        2.17x |              n/a |
| `move_exp_nansum`   |     1000 |     0ms |    0ms |        n/a |        8.19x |              n/a |
|                     |   100000 |     1ms |    2ms |        n/a |        1.70x |              n/a |
|                     | 10000000 |    98ms |  198ms |        n/a |        2.02x |              n/a |
| `move_exp_nanvar`   |     1000 |     0ms |    0ms |        n/a |        7.10x |              n/a |
|                     |   100000 |     2ms |    3ms |        n/a |        1.66x |              n/a |
|                     | 10000000 |   129ms |  249ms |        n/a |        1.94x |              n/a |

[^1][^2][^3]

[^1]:
    Benchmarks were run on a Mac M1 in October 2023 on numbagg's HEAD and
    pandas 2.1.1.

[^2]:
    While we separate the setup and the running of the functions, pandas still
    needs to do some work to create its result dataframe. So we focus on the
    benchmarks for larger arrays in order to reduce that impact. Any
    contributions to improve the benchmarks for other libraries are more than
    welcome.

[^3]:
    Pandas doesn't have an equivalent `move_exp_nancount` function, so this is
    compared to a function which uses its `sum` function on an array of `1`s.

</details>

## Easy to extend

Numbagg makes it easy to write, in pure Python/NumPy, flexible aggregation
functions accelerated by Numba. All the hard work is done by Numba's JIT
compiler and NumPy's gufunc machinery (as wrapped by Numba).

For example, here is how we wrote `nansum`:

```python
import numpy as np
from numbagg.decorators import ndreduce

@ndreduce
def nansum(a):
    asum = 0.0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
    return asum
```

You are welcome to experiment with Numbagg's decorator functions, but these are
not public APIs (yet): we reserve the right to change them at any time.

We'd rather get your pull requests to add new functions into Numbagg directly!

## Advantages over Bottleneck

- Way less code. Easier to add new functions. No ad-hoc templating
  system. No Cython!
- Fast functions still work for >3 dimensions.
- `axis` argument handles tuples of integers.

Most of the functions in Numbagg (including our test suite) are adapted from
Bottleneck's battle-hardened implementations.

## Our approach

Numbagg includes somewhat awkward workarounds for features missing from
NumPy/Numba:

- It implements its own cache for functions wrapped by Numba's
  `guvectorize`, because that decorator is rather slow.
- It does its [own handling of array
  transposes](https://github.com/numbagg/numbagg/blob/main/numbagg/decorators.py#L69)
  to handle the `axis` argument, which we hope will [eventually be
  directly supported](https://github.com/numpy/numpy/issues/5197) by
  all NumPy gufuncs.
- It uses some [terrible
  hacks](https://github.com/numbagg/numbagg/blob/main/numbagg/transform.py) to
  hide the out-of-bound memory access necessary to write [gufuncs that handle
  scalar
  values](https://github.com/numba/numba/blob/main/numba/tests/test_guvectorize_scalar.py)
  with Numba.

I hope that the need for most of these will eventually go away. In the meantime,
expect Numbagg to be tightly coupled to Numba and NumPy release cycles.

## License

3-clause BSD. Includes portions of Bottleneck, which is distributed under a
Simplified BSD license.
