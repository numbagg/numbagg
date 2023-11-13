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

## Why use numba?

### Performance

- Much faster than pandas for almost every function — 2-20x
- About the same speed as bottleneck on a single calculation
- Much faster than bottleneck — 4-7x — when parallelizing with multiple cores — for
  example, calculating over each row on an array with 10 rows.
- ...though numbagg's functions are JIT compiled, so they're much slower on
  their first run

<!-- Disabled in #189, hopefully temporarily -->
<!-- The compilation is generally cached on disk[^4]. -->

### Versatility

- More functions (though bottleneck has some functions we don't have, and pandas' functions
  have many more parameters)
- Fast functions work for >3 dimensions. Functions take an arbitrary axis or
  tuple of axes to calculate over
- Written in numba — way less code, simple to inspect, simple to improve

## Benchmarks

### 2D

Array of shape `(1, 10000000)`, over the final axis

| func                | numbagg | pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------ | ------: | -----: | ---------: | -----------: | ---------------: |
| `bfill`             |    17ms |  504ms |       20ms |       29.10x |            1.13x |
| `ffill`             |    18ms |  489ms |       19ms |       27.88x |            1.06x |
| `move_corr`         |    48ms |  922ms |        n/a |       19.23x |              n/a |
| `move_cov`          |    42ms |  653ms |        n/a |       15.50x |              n/a |
| `move_mean`         |    32ms |  131ms |       27ms |        4.12x |            0.86x |
| `move_std`          |    24ms |  190ms |       38ms |        7.86x |            1.57x |
| `move_sum`          |    31ms |  118ms |       27ms |        3.83x |            0.88x |
| `move_var`          |    24ms |  177ms |       35ms |        7.41x |            1.48x |
| `move_exp_nancorr`  |    69ms |  455ms |        n/a |        6.63x |              n/a |
| `move_exp_nancount` |    32ms |   83ms |        n/a |        2.59x |              n/a |
| `move_exp_nancov`   |    51ms |  283ms |        n/a |        5.58x |              n/a |
| `move_exp_nanmean`  |    33ms |   72ms |        n/a |        2.17x |              n/a |
| `move_exp_nanstd`   |    48ms |   95ms |        n/a |        1.98x |              n/a |
| `move_exp_nansum`   |    32ms |   64ms |        n/a |        1.97x |              n/a |
| `move_exp_nanvar`   |    42ms |   82ms |        n/a |        1.97x |              n/a |
| `nanquantile`       |   218ms |  680ms |        n/a |        3.12x |              n/a |

### ND

Array of shape `(100, 1000, 1000)`, over the final axis

| func                | numbagg | pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------ | ------: | -----: | ---------: | -----------: | ---------------: |
| `bfill`             |    38ms |    n/a |      244ms |          n/a |            6.38x |
| `ffill`             |    50ms |    n/a |      221ms |          n/a |            4.44x |
| `move_corr`         |   130ms |    n/a |        n/a |          n/a |              n/a |
| `move_cov`          |    69ms |    n/a |        n/a |          n/a |              n/a |
| `move_mean`         |    51ms |    n/a |      308ms |          n/a |            6.06x |
| `move_std`          |   106ms |    n/a |      372ms |          n/a |            3.51x |
| `move_sum`          |    59ms |    n/a |      287ms |          n/a |            4.90x |
| `move_var`          |    44ms |    n/a |      370ms |          n/a |            8.50x |
| `move_exp_nancorr`  |   136ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancount` |   119ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancov`   |   124ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanmean`  |   158ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanstd`   |    94ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nansum`   |   215ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanvar`   |   160ms |    n/a |        n/a |          n/a |              n/a |
| `nanquantile`       |  2179ms |    n/a |        n/a |          n/a |              n/a |

[^1][^2][^3]

[^1]:
    Benchmarks were run on a Mac M1 laptop in October 2023 on numbagg's HEAD,
    pandas 2.1.1, bottleneck 1.3.7. They're also run in CI, though without
    demonstrating the benefits of parallelization given GHA's CPU count.

[^2]:
    While we separate the setup and the running of the functions, pandas still
    needs to do some work to create its result dataframe, and numbagg does some
    checks in python which bottleneck does in C or doesn't do. So we focus on
    the benchmarks for larger arrays in order to reduce that impact. Any
    contributions to improve the benchmarks are welcome.

[^3]:
    Pandas doesn't have an equivalent `move_exp_nancount` function, so this is
    compared to a function which uses its `sum` function on an array of `1`s.

<details>
<summary>Full benchmarks</summary>

### All

| func                |                  shape |      size | numbagg | pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------ | ---------------------: | --------: | ------: | -----: | ---------: | -----------: | ---------------: |
| `bfill`             |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        3.18x |            0.03x |
|                     |          (10, 1000000) |  10000000 |     4ms |   74ms |       20ms |       20.56x |            5.64x |
|                     |          (1, 10000000) |  10000000 |    17ms |  504ms |       20ms |       29.10x |            1.13x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |       21ms |          n/a |            5.35x |
|                     |      (100, 1000, 1000) | 100000000 |    38ms |    n/a |      244ms |          n/a |            6.38x |
| `ffill`             |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        2.52x |            0.01x |
|                     |          (10, 1000000) |  10000000 |     4ms |   73ms |       19ms |       17.64x |            4.50x |
|                     |          (1, 10000000) |  10000000 |    18ms |  489ms |       19ms |       27.88x |            1.06x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |       19ms |          n/a |            4.60x |
|                     |      (100, 1000, 1000) | 100000000 |    50ms |    n/a |      221ms |          n/a |            4.44x |
| `move_corr`         |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        5.04x |              n/a |
|                     |          (10, 1000000) |  10000000 |    10ms |  927ms |        n/a |       89.63x |              n/a |
|                     |          (1, 10000000) |  10000000 |    48ms |  922ms |        n/a |       19.23x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    10ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |   130ms |    n/a |        n/a |          n/a |              n/a |
| `move_cov`          |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        4.64x |              n/a |
|                     |          (10, 1000000) |  10000000 |     9ms |  694ms |        n/a |       76.55x |              n/a |
|                     |          (1, 10000000) |  10000000 |    42ms |  653ms |        n/a |       15.50x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     9ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |    69ms |    n/a |        n/a |          n/a |              n/a |
| `move_mean`         |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        1.43x |            0.02x |
|                     |          (10, 1000000) |  10000000 |     7ms |  134ms |       27ms |       19.67x |            3.94x |
|                     |          (1, 10000000) |  10000000 |    32ms |  131ms |       27ms |        4.12x |            0.86x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       28ms |          n/a |            5.32x |
|                     |      (100, 1000, 1000) | 100000000 |    51ms |    n/a |      308ms |          n/a |            6.06x |
| `move_std`          |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        1.69x |            0.05x |
|                     |          (10, 1000000) |  10000000 |     5ms |  185ms |       36ms |       33.95x |            6.56x |
|                     |          (1, 10000000) |  10000000 |    24ms |  190ms |       38ms |        7.86x |            1.57x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       37ms |          n/a |            8.06x |
|                     |      (100, 1000, 1000) | 100000000 |   106ms |    n/a |      372ms |          n/a |            3.51x |
| `move_sum`          |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        1.64x |            0.02x |
|                     |          (10, 1000000) |  10000000 |     7ms |  125ms |       26ms |       17.60x |            3.68x |
|                     |          (1, 10000000) |  10000000 |    31ms |  118ms |       27ms |        3.83x |            0.88x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |       26ms |          n/a |            4.29x |
|                     |      (100, 1000, 1000) | 100000000 |    59ms |    n/a |      287ms |          n/a |            4.90x |
| `move_var`          |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        1.55x |            0.05x |
|                     |          (10, 1000000) |  10000000 |     5ms |  187ms |       35ms |       39.13x |            7.37x |
|                     |          (1, 10000000) |  10000000 |    24ms |  177ms |       35ms |        7.41x |            1.48x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    20ms |    n/a |       37ms |          n/a |            1.90x |
|                     |      (100, 1000, 1000) | 100000000 |    44ms |    n/a |      370ms |          n/a |            8.50x |
| `move_exp_nancorr`  |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        6.90x |              n/a |
|                     |          (10, 1000000) |  10000000 |    13ms |  459ms |        n/a |       35.88x |              n/a |
|                     |          (1, 10000000) |  10000000 |    69ms |  455ms |        n/a |        6.63x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    14ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |   136ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancount` |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        1.43x |              n/a |
|                     |          (10, 1000000) |  10000000 |     7ms |   73ms |        n/a |        9.82x |              n/a |
|                     |          (1, 10000000) |  10000000 |    32ms |   83ms |        n/a |        2.59x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |   119ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancov`   |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        6.49x |              n/a |
|                     |          (10, 1000000) |  10000000 |    10ms |  319ms |        n/a |       31.23x |              n/a |
|                     |          (1, 10000000) |  10000000 |    51ms |  283ms |        n/a |        5.58x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    10ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |   124ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanmean`  |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        1.26x |              n/a |
|                     |          (10, 1000000) |  10000000 |     6ms |   78ms |        n/a |       12.63x |              n/a |
|                     |          (1, 10000000) |  10000000 |    33ms |   72ms |        n/a |        2.17x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     7ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |   158ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanstd`   |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        2.09x |              n/a |
|                     |          (10, 1000000) |  10000000 |    10ms |  101ms |        n/a |        9.65x |              n/a |
|                     |          (1, 10000000) |  10000000 |    48ms |   95ms |        n/a |        1.98x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    10ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |    94ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nansum`   |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        1.37x |              n/a |
|                     |          (10, 1000000) |  10000000 |     7ms |   66ms |        n/a |        9.57x |              n/a |
|                     |          (1, 10000000) |  10000000 |    32ms |   64ms |        n/a |        1.97x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |   215ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanvar`   |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        1.39x |              n/a |
|                     |          (10, 1000000) |  10000000 |     9ms |   91ms |        n/a |       10.55x |              n/a |
|                     |          (1, 10000000) |  10000000 |    42ms |   82ms |        n/a |        1.97x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     9ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |   160ms |    n/a |        n/a |          n/a |              n/a |
| `nanquantile`       |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        3.28x |              n/a |
|                     |          (10, 1000000) |  10000000 |   214ms |  257ms |        n/a |        1.20x |              n/a |
|                     |          (1, 10000000) |  10000000 |   218ms |  680ms |        n/a |        3.12x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |   218ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |  2179ms |    n/a |        n/a |          n/a |              n/a |

</details>

## Example implementation

Numbagg makes it easy to write, in pure Python/NumPy, flexible aggregation
functions accelerated by Numba. All the hard work is done by Numba's JIT
compiler and NumPy's gufunc machinery (as wrapped by Numba).

For example, here is how we wrote `nansum`:

```python
import numpy as np
from numbagg.decorators import ndreduce

@ndreduce.wrap()
def nansum(a):
    asum = 0.0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
    return asum
```

## Implementation details

Numbagg includes somewhat awkward workarounds for features missing from
NumPy/Numba:

- It implements its own cache for functions wrapped by Numba's
  `guvectorize`, because that decorator is rather slow.
- It does its [own handling of array
  transposes](https://github.com/numbagg/numbagg/blob/e166adae94b3be35497dcdc22772026df75af253/numbagg/decorators.py#L170-L181)
  to handle the `axis` argument in reduction functions.
- It [rewrites plain functions into
  gufuncs](https://github.com/numbagg/numbagg/blob/e166adae94b3be35497dcdc22772026df75af253/numbagg/transform.py),
  to allow writing a traditional function while retaining the multidimensional advantages of
  gufuncs.

Already some of the ideas here have flowed upstream to numba (for example, [an
axis parameter](https://github.com/numpy/numpy/issues/5197)), and we hope
that others will follow.

## License

3-clause BSD. Includes portions of Bottleneck, which is distributed under a
Simplified BSD license.
