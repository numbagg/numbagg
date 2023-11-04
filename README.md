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
| `bfill`             |    17ms |  487ms |       21ms |       28.12x |            1.24x |
| `ffill`             |    17ms |  502ms |       19ms |       29.32x |            1.14x |
| `move_corr`         |    51ms |  937ms |        n/a |       18.25x |              n/a |
| `move_cov`          |    41ms |  667ms |        n/a |       16.16x |              n/a |
| `move_mean`         |    31ms |  127ms |       29ms |        4.06x |            0.92x |
| `move_std`          |    25ms |  186ms |       37ms |        7.58x |            1.52x |
| `move_sum`          |    31ms |  122ms |       25ms |        3.97x |            0.81x |
| `move_var`          |    24ms |  173ms |       35ms |        7.23x |            1.47x |
| `move_exp_nancorr`  |    63ms |  464ms |        n/a |        7.37x |              n/a |
| `move_exp_nancount` |    35ms |   86ms |        n/a |        2.47x |              n/a |
| `move_exp_nancov`   |    43ms |  290ms |        n/a |        6.69x |              n/a |
| `move_exp_nanmean`  |    33ms |   73ms |        n/a |        2.17x |              n/a |
| `move_exp_nanstd`   |    45ms |  101ms |        n/a |        2.24x |              n/a |
| `move_exp_nansum`   |    31ms |   61ms |        n/a |        1.95x |              n/a |
| `move_exp_nanvar`   |    42ms |   84ms |        n/a |        1.98x |              n/a |

### ND

Array of shape `(100, 1000, 1000)`, over the final axis

| func                | numbagg | pandas | bottleneck | pandas ratio | bottleneck ratio |
| :------------------ | ------: | -----: | ---------: | -----------: | ---------------: |
| `bfill`             |    36ms |    n/a |      261ms |          n/a |            7.28x |
| `ffill`             |    62ms |    n/a |      214ms |          n/a |            3.44x |
| `move_corr`         |    80ms |    n/a |        n/a |          n/a |              n/a |
| `move_cov`          |    74ms |    n/a |        n/a |          n/a |              n/a |
| `move_mean`         |    73ms |    n/a |      400ms |          n/a |            5.46x |
| `move_std`          |    65ms |    n/a |      543ms |          n/a |            8.30x |
| `move_sum`          |    47ms |    n/a |      282ms |          n/a |            6.03x |
| `move_var`          |    62ms |    n/a |      366ms |          n/a |            5.94x |
| `move_exp_nancorr`  |   125ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancount` |    65ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancov`   |    72ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanmean`  |    51ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanstd`   |    84ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nansum`   |    58ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanvar`   |   124ms |    n/a |        n/a |          n/a |              n/a |

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
| `bfill`             |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        2.21x |            0.02x |
|                     |          (10, 1000000) |  10000000 |     5ms |   81ms |       22ms |       14.92x |            3.97x |
|                     |          (1, 10000000) |  10000000 |    17ms |  487ms |       21ms |       28.12x |            1.24x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       23ms |          n/a |            4.83x |
|                     |      (100, 1000, 1000) | 100000000 |    36ms |    n/a |      261ms |          n/a |            7.28x |
| `ffill`             |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        3.07x |            0.02x |
|                     |          (10, 1000000) |  10000000 |     4ms |   77ms |       19ms |       21.89x |            5.34x |
|                     |          (1, 10000000) |  10000000 |    17ms |  502ms |       19ms |       29.32x |            1.14x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |       20ms |          n/a |            4.41x |
|                     |      (100, 1000, 1000) | 100000000 |    62ms |    n/a |      214ms |          n/a |            3.44x |
| `move_corr`         |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        9.84x |              n/a |
|                     |          (10, 1000000) |  10000000 |     9ms |  964ms |        n/a |      105.22x |              n/a |
|                     |          (1, 10000000) |  10000000 |    51ms |  937ms |        n/a |       18.25x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    14ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |    80ms |    n/a |        n/a |          n/a |              n/a |
| `move_cov`          |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        7.23x |              n/a |
|                     |          (10, 1000000) |  10000000 |     9ms |  655ms |        n/a |       74.77x |              n/a |
|                     |          (1, 10000000) |  10000000 |    41ms |  667ms |        n/a |       16.16x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    10ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |    74ms |    n/a |        n/a |          n/a |              n/a |
| `move_mean`         |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        1.55x |            0.02x |
|                     |          (10, 1000000) |  10000000 |     6ms |  126ms |       28ms |       20.95x |            4.58x |
|                     |          (1, 10000000) |  10000000 |    31ms |  127ms |       29ms |        4.06x |            0.92x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |       27ms |          n/a |            4.62x |
|                     |      (100, 1000, 1000) | 100000000 |    73ms |    n/a |      400ms |          n/a |            5.46x |
| `move_std`          |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        1.57x |            0.05x |
|                     |          (10, 1000000) |  10000000 |     5ms |  187ms |       37ms |       39.33x |            7.75x |
|                     |          (1, 10000000) |  10000000 |    25ms |  186ms |       37ms |        7.58x |            1.52x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |       34ms |          n/a |            6.09x |
|                     |      (100, 1000, 1000) | 100000000 |    65ms |    n/a |      543ms |          n/a |            8.30x |
| `move_sum`          |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        1.74x |            0.02x |
|                     |          (10, 1000000) |  10000000 |     7ms |  119ms |       24ms |       16.41x |            3.37x |
|                     |          (1, 10000000) |  10000000 |    31ms |  122ms |       25ms |        3.97x |            0.81x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |       25ms |          n/a |            3.99x |
|                     |      (100, 1000, 1000) | 100000000 |    47ms |    n/a |      282ms |          n/a |            6.03x |
| `move_var`          |              (1, 1000) |      1000 |     0ms |    0ms |        0ms |        1.81x |            0.06x |
|                     |          (10, 1000000) |  10000000 |     5ms |  174ms |       35ms |       32.90x |            6.53x |
|                     |          (1, 10000000) |  10000000 |    24ms |  173ms |       35ms |        7.23x |            1.47x |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       34ms |          n/a |            6.40x |
|                     |      (100, 1000, 1000) | 100000000 |    62ms |    n/a |      366ms |          n/a |            5.94x |
| `move_exp_nancorr`  |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        6.67x |              n/a |
|                     |          (10, 1000000) |  10000000 |    11ms |  464ms |        n/a |       42.03x |              n/a |
|                     |          (1, 10000000) |  10000000 |    63ms |  464ms |        n/a |        7.37x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    15ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |   125ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancount` |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        1.86x |              n/a |
|                     |          (10, 1000000) |  10000000 |     6ms |   89ms |        n/a |       14.52x |              n/a |
|                     |          (1, 10000000) |  10000000 |    35ms |   86ms |        n/a |        2.47x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     9ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |    65ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancov`   |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        7.77x |              n/a |
|                     |          (10, 1000000) |  10000000 |    10ms |  307ms |        n/a |       31.26x |              n/a |
|                     |          (1, 10000000) |  10000000 |    43ms |  290ms |        n/a |        6.69x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    10ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |    72ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanmean`  |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        1.35x |              n/a |
|                     |          (10, 1000000) |  10000000 |     8ms |   88ms |        n/a |       10.44x |              n/a |
|                     |          (1, 10000000) |  10000000 |    33ms |   73ms |        n/a |        2.17x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |    51ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanstd`   |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        1.93x |              n/a |
|                     |          (10, 1000000) |  10000000 |    10ms |   96ms |        n/a |        9.64x |              n/a |
|                     |          (1, 10000000) |  10000000 |    45ms |  101ms |        n/a |        2.24x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    10ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |    84ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nansum`   |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        1.38x |              n/a |
|                     |          (10, 1000000) |  10000000 |     6ms |   66ms |        n/a |       11.12x |              n/a |
|                     |          (1, 10000000) |  10000000 |    31ms |   61ms |        n/a |        1.95x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |    58ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanvar`   |              (1, 1000) |      1000 |     0ms |    0ms |        n/a |        1.09x |              n/a |
|                     |          (10, 1000000) |  10000000 |     8ms |  100ms |        n/a |       12.14x |              n/a |
|                     |          (1, 10000000) |  10000000 |    42ms |   84ms |        n/a |        1.98x |              n/a |
|                     | (10, 10, 10, 10, 1000) |  10000000 |    11ms |    n/a |        n/a |          n/a |              n/a |
|                     |      (100, 1000, 1000) | 100000000 |   124ms |    n/a |        n/a |          n/a |              n/a |

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
