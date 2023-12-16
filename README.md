# Numbagg: Fast N-dimensional aggregation functions with Numba

[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/numbagg/numbagg/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/numbagg/numbagg/actions/workflows/test.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/numbagg?style=for-the-badge)](https://pypi.python.org/pypi/numbagg/)

Fast, flexible N-dimensional array functions written with
[Numba](https://github.com/numba/numba) and NumPy's [generalized
ufuncs](http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html).

Currently accelerated functions:

- Aggregation functions: `allnan`, `anynan`, `count`, `nanargmax`,
  `nanargmin`, `nanmax`, `nanmean`, `nanstd`, `nanvar`, `nanmin`,
  `nansum`.
- Array functions listed below
- Grouped functions listed below
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

We have two high-level benchmarks — with a 1D array with no parallelization, and
with a 2D array with the potential for parallelization. Numbagg's relative performance is
much higher where parallelization is possible.

### 1D

Array of shape `(10000000,)`, over the final axis

| func                      | numbagg | pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------------ | ------: | -----: | ---------: | -----------: | ---------------: |
| `bfill`                   |    17ms |   19ms |       20ms |        1.11x |            1.15x |
| `ffill`                   |    17ms |   20ms |       20ms |        1.16x |            1.15x |
| `group_nanall`            |    52ms |   71ms |        n/a |        1.38x |              n/a |
| `group_nanany`            |    57ms |   70ms |        n/a |        1.23x |              n/a |
| `group_nanargmax`         |    63ms |  169ms |        n/a |        2.69x |              n/a |
| `group_nanargmin`         |    60ms |  179ms |        n/a |        2.98x |              n/a |
| `group_nancount`          |    57ms |   59ms |        n/a |        1.03x |              n/a |
| `group_nanfirst`          |    47ms |   71ms |        n/a |        1.52x |              n/a |
| `group_nanlast`           |    62ms |   71ms |        n/a |        1.15x |              n/a |
| `group_nanmax`            |    59ms |   71ms |        n/a |        1.20x |              n/a |
| `group_nanmean`           |    60ms |   72ms |        n/a |        1.21x |              n/a |
| `group_nanmin`            |    66ms |   73ms |        n/a |        1.11x |              n/a |
| `group_nanprod`           |    58ms |   67ms |        n/a |        1.15x |              n/a |
| `group_nanstd`            |    60ms |   70ms |        n/a |        1.17x |              n/a |
| `group_nansum`            |    61ms |   72ms |        n/a |        1.18x |              n/a |
| `group_nanvar`            |    64ms |   80ms |        n/a |        1.25x |              n/a |
| `group_nansum_of_squares` |    66ms |   80ms |        n/a |        1.21x |              n/a |
| `move_corr`               |    51ms |  955ms |        n/a |       18.65x |              n/a |
| `move_cov`                |    44ms |  661ms |        n/a |       14.94x |              n/a |
| `move_mean`               |    33ms |  119ms |       26ms |        3.62x |            0.79x |
| `move_std`                |    29ms |  179ms |       33ms |        6.26x |            1.15x |
| `move_sum`                |    31ms |  114ms |       24ms |        3.64x |            0.78x |
| `move_var`                |    30ms |  174ms |       32ms |        5.88x |            1.07x |
| `move_exp_nancorr`        |    68ms |  472ms |        n/a |        6.98x |              n/a |
| `move_exp_nancount`       |    36ms |   83ms |        n/a |        2.33x |              n/a |
| `move_exp_nancov`         |    51ms |  298ms |        n/a |        5.85x |              n/a |
| `move_exp_nanmean`        |    33ms |   69ms |        n/a |        2.09x |              n/a |
| `move_exp_nanstd`         |    46ms |   94ms |        n/a |        2.05x |              n/a |
| `move_exp_nansum`         |    31ms |   65ms |        n/a |        2.13x |              n/a |
| `move_exp_nanvar`         |    47ms |   82ms |        n/a |        1.75x |              n/a |
| `nanquantile`             |   212ms |  187ms |        n/a |        0.88x |              n/a |

### 2D

Array of shape `(100, 100000)`, over the final axis

| func                      | numbagg | pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------------ | ------: | -----: | ---------: | -----------: | ---------------: |
| `bfill`                   |     5ms |   57ms |       19ms |       12.48x |            4.23x |
| `ffill`                   |     5ms |   60ms |       20ms |       12.27x |            4.10x |
| `group_nanall`            |     2ms |   18ms |        n/a |       10.04x |              n/a |
| `group_nanany`            |     4ms |   19ms |        n/a |        5.20x |              n/a |
| `group_nanargmax`         |     n/a |   43ms |        n/a |          n/a |              n/a |
| `group_nanargmin`         |     n/a |   39ms |        n/a |          n/a |              n/a |
| `group_nancount`          |     4ms |   16ms |        n/a |        4.02x |              n/a |
| `group_nanfirst`          |     2ms |   16ms |        n/a |       10.64x |              n/a |
| `group_nanlast`           |     4ms |   16ms |        n/a |        4.08x |              n/a |
| `group_nanmax`            |     4ms |   18ms |        n/a |        4.17x |              n/a |
| `group_nanmean`           |     4ms |   19ms |        n/a |        5.44x |              n/a |
| `group_nanmin`            |     4ms |   18ms |        n/a |        4.10x |              n/a |
| `group_nanprod`           |     4ms |   16ms |        n/a |        4.25x |              n/a |
| `group_nanstd`            |     4ms |   20ms |        n/a |        5.51x |              n/a |
| `group_nansum`            |     4ms |   19ms |        n/a |        5.22x |              n/a |
| `group_nanvar`            |     4ms |   20ms |        n/a |        5.49x |              n/a |
| `group_nansum_of_squares` |     4ms |   28ms |        n/a |        7.47x |              n/a |
| `move_corr`               |    10ms |  919ms |        n/a |       90.69x |              n/a |
| `move_cov`                |     9ms |  622ms |        n/a |       69.62x |              n/a |
| `move_mean`               |     6ms |  122ms |       26ms |       18.98x |            4.07x |
| `move_std`                |     5ms |  172ms |       33ms |       32.21x |            6.18x |
| `move_sum`                |     5ms |  112ms |       25ms |       21.28x |            4.68x |
| `move_var`                |     6ms |  158ms |       32ms |       24.87x |            5.11x |
| `move_exp_nancorr`        |    15ms |  482ms |        n/a |       32.83x |              n/a |
| `move_exp_nancount`       |     7ms |   73ms |        n/a |        9.95x |              n/a |
| `move_exp_nancov`         |    10ms |  332ms |        n/a |       33.19x |              n/a |
| `move_exp_nanmean`        |     6ms |   78ms |        n/a |       12.12x |              n/a |
| `move_exp_nanstd`         |     9ms |   98ms |        n/a |       11.40x |              n/a |
| `move_exp_nansum`         |     7ms |   68ms |        n/a |        9.32x |              n/a |
| `move_exp_nanvar`         |     8ms |   88ms |        n/a |       10.74x |              n/a |
| `nanquantile`             |   213ms |  196ms |        n/a |        0.92x |              n/a |

[^1][^2][^3]

[^1]:
    Benchmarks were run on a Mac M1 laptop in December 2023 on numbagg's HEAD,
    pandas 2.1.1, bottleneck 1.3.7. They're also run in CI, though without
    demonstrating the full benefits of parallelization given GHA's CPU count.

[^2]:
    While we separate the setup and the running of the functions, pandas still
    needs to do some work to create its result dataframe, and numbagg does some
    checks in python which bottleneck does in C or doesn't do. So we focus on
    the benchmarks for larger arrays in order to reduce that impact. Any
    contributions to improve the benchmarks are welcome.

[^3]:
    Pandas doesn't have an equivalent `move_exp_nancount` function, so this is
    compared to a function which uses its `sum` function on an array of `1`s.
    Similarly for `group_nansum_of_squares`, this requires two separate
    operations in pandas.

<details>
<summary>Full benchmarks</summary>

### All

| func                      |                  shape |      size | numbagg | pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------------ | ---------------------: | --------: | ------: | -----: | ---------: | -----------: | ---------------: |
| `bfill`                   |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.62x |            0.01x |
|                           |            (10000000,) |  10000000 |    17ms |   19ms |       20ms |        1.11x |            1.15x |
|                           |          (100, 100000) |  10000000 |     5ms |   57ms |       19ms |       12.48x |            4.23x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       22ms |          n/a |            4.75x |
|                           |      (100, 1000, 1000) | 100000000 |    37ms |    n/a |      262ms |          n/a |            7.04x |
| `ffill`                   |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.60x |            0.01x |
|                           |            (10000000,) |  10000000 |    17ms |   20ms |       20ms |        1.16x |            1.15x |
|                           |          (100, 100000) |  10000000 |     5ms |   60ms |       20ms |       12.27x |            4.10x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |       19ms |          n/a |            4.59x |
|                           |      (100, 1000, 1000) | 100000000 |    38ms |    n/a |      225ms |          n/a |            5.86x |
| `group_nanall`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.73x |              n/a |
|                           |            (10000000,) |  10000000 |    52ms |   71ms |        n/a |        1.38x |              n/a |
|                           |          (100, 100000) |  10000000 |     2ms |   18ms |        n/a |       10.04x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     1ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanany`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.75x |              n/a |
|                           |            (10000000,) |  10000000 |    57ms |   70ms |        n/a |        1.23x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   19ms |        n/a |        5.20x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     3ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanargmax`         |                (1000,) |      1000 |     0ms |    1ms |        n/a |        7.85x |              n/a |
|                           |            (10000000,) |  10000000 |    63ms |  169ms |        n/a |        2.69x |              n/a |
|                           |          (100, 100000) |  10000000 |     n/a |   43ms |        n/a |          n/a |              n/a |
| `group_nanargmin`         |                (1000,) |      1000 |     0ms |    1ms |        n/a |        7.62x |              n/a |
|                           |            (10000000,) |  10000000 |    60ms |  179ms |        n/a |        2.98x |              n/a |
|                           |          (100, 100000) |  10000000 |     n/a |   39ms |        n/a |          n/a |              n/a |
| `group_nancount`          |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.73x |              n/a |
|                           |            (10000000,) |  10000000 |    57ms |   59ms |        n/a |        1.03x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   16ms |        n/a |        4.02x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     3ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanfirst`          |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.87x |              n/a |
|                           |            (10000000,) |  10000000 |    47ms |   71ms |        n/a |        1.52x |              n/a |
|                           |          (100, 100000) |  10000000 |     2ms |   16ms |        n/a |       10.64x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     1ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanlast`           |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.73x |              n/a |
|                           |            (10000000,) |  10000000 |    62ms |   71ms |        n/a |        1.15x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   16ms |        n/a |        4.08x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     3ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanmax`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.80x |              n/a |
|                           |            (10000000,) |  10000000 |    59ms |   71ms |        n/a |        1.20x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   18ms |        n/a |        4.17x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanmean`           |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.92x |              n/a |
|                           |            (10000000,) |  10000000 |    60ms |   72ms |        n/a |        1.21x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   19ms |        n/a |        5.44x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanmin`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.77x |              n/a |
|                           |            (10000000,) |  10000000 |    66ms |   73ms |        n/a |        1.11x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   18ms |        n/a |        4.10x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanprod`           |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.76x |              n/a |
|                           |            (10000000,) |  10000000 |    58ms |   67ms |        n/a |        1.15x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   16ms |        n/a |        4.25x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     3ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanstd`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.90x |              n/a |
|                           |            (10000000,) |  10000000 |    60ms |   70ms |        n/a |        1.17x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   20ms |        n/a |        5.51x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |          n/a |              n/a |
| `group_nansum`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.94x |              n/a |
|                           |            (10000000,) |  10000000 |    61ms |   72ms |        n/a |        1.18x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   19ms |        n/a |        5.22x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     3ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanvar`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.86x |              n/a |
|                           |            (10000000,) |  10000000 |    64ms |   80ms |        n/a |        1.25x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   20ms |        n/a |        5.49x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |          n/a |              n/a |
| `group_nansum_of_squares` |                (1000,) |      1000 |     0ms |    0ms |        n/a |        1.21x |              n/a |
|                           |            (10000000,) |  10000000 |    66ms |   80ms |        n/a |        1.21x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   28ms |        n/a |        7.47x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     3ms |    n/a |        n/a |          n/a |              n/a |
| `move_corr`               |                (1000,) |      1000 |     0ms |    0ms |        n/a |        7.70x |              n/a |
|                           |            (10000000,) |  10000000 |    51ms |  955ms |        n/a |       18.65x |              n/a |
|                           |          (100, 100000) |  10000000 |    10ms |  919ms |        n/a |       90.69x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     8ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |    98ms |    n/a |        n/a |          n/a |              n/a |
| `move_cov`                |                (1000,) |      1000 |     0ms |    0ms |        n/a |        7.13x |              n/a |
|                           |            (10000000,) |  10000000 |    44ms |  661ms |        n/a |       14.94x |              n/a |
|                           |          (100, 100000) |  10000000 |     9ms |  622ms |        n/a |       69.62x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     8ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |    83ms |    n/a |        n/a |          n/a |              n/a |
| `move_mean`               |                (1000,) |      1000 |     0ms |    0ms |        0ms |        1.63x |            0.02x |
|                           |            (10000000,) |  10000000 |    33ms |  119ms |       26ms |        3.62x |            0.79x |
|                           |          (100, 100000) |  10000000 |     6ms |  122ms |       26ms |       18.98x |            4.07x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |       28ms |          n/a |            4.24x |
|                           |      (100, 1000, 1000) | 100000000 |    49ms |    n/a |      294ms |          n/a |            6.01x |
| `move_std`                |                (1000,) |      1000 |     0ms |    0ms |        0ms |        1.79x |            0.05x |
|                           |            (10000000,) |  10000000 |    29ms |  179ms |       33ms |        6.26x |            1.15x |
|                           |          (100, 100000) |  10000000 |     5ms |  172ms |       33ms |       32.21x |            6.18x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       34ms |          n/a |            6.14x |
|                           |      (100, 1000, 1000) | 100000000 |    48ms |    n/a |      376ms |          n/a |            7.91x |
| `move_sum`                |                (1000,) |      1000 |     0ms |    0ms |        0ms |        1.64x |            0.02x |
|                           |            (10000000,) |  10000000 |    31ms |  114ms |       24ms |        3.64x |            0.78x |
|                           |          (100, 100000) |  10000000 |     5ms |  112ms |       25ms |       21.28x |            4.68x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     7ms |    n/a |       24ms |          n/a |            3.56x |
|                           |      (100, 1000, 1000) | 100000000 |    48ms |    n/a |      277ms |          n/a |            5.73x |
| `move_var`                |                (1000,) |      1000 |     0ms |    0ms |        0ms |        1.67x |            0.05x |
|                           |            (10000000,) |  10000000 |    30ms |  174ms |       32ms |        5.88x |            1.07x |
|                           |          (100, 100000) |  10000000 |     6ms |  158ms |       32ms |       24.87x |            5.11x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |       32ms |          n/a |            5.07x |
|                           |      (100, 1000, 1000) | 100000000 |    48ms |    n/a |      365ms |          n/a |            7.55x |
| `move_exp_nancorr`        |                (1000,) |      1000 |     0ms |    0ms |        n/a |        6.44x |              n/a |
|                           |            (10000000,) |  10000000 |    68ms |  472ms |        n/a |        6.98x |              n/a |
|                           |          (100, 100000) |  10000000 |    15ms |  482ms |        n/a |       32.83x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    12ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |   112ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancount`       |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.94x |              n/a |
|                           |            (10000000,) |  10000000 |    36ms |   83ms |        n/a |        2.33x |              n/a |
|                           |          (100, 100000) |  10000000 |     7ms |   73ms |        n/a |        9.95x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |    58ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancov`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |        5.41x |              n/a |
|                           |            (10000000,) |  10000000 |    51ms |  298ms |        n/a |        5.85x |              n/a |
|                           |          (100, 100000) |  10000000 |    10ms |  332ms |        n/a |       33.19x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    11ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |    81ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanmean`        |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.66x |              n/a |
|                           |            (10000000,) |  10000000 |    33ms |   69ms |        n/a |        2.09x |              n/a |
|                           |          (100, 100000) |  10000000 |     6ms |   78ms |        n/a |       12.12x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |    69ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanstd`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |        1.09x |              n/a |
|                           |            (10000000,) |  10000000 |    46ms |   94ms |        n/a |        2.05x |              n/a |
|                           |          (100, 100000) |  10000000 |     9ms |   98ms |        n/a |       11.40x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     8ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |    80ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nansum`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |        1.36x |              n/a |
|                           |            (10000000,) |  10000000 |    31ms |   65ms |        n/a |        2.13x |              n/a |
|                           |          (100, 100000) |  10000000 |     7ms |   68ms |        n/a |        9.32x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     7ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |    55ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanvar`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |        1.27x |              n/a |
|                           |            (10000000,) |  10000000 |    47ms |   82ms |        n/a |        1.75x |              n/a |
|                           |          (100, 100000) |  10000000 |     8ms |   88ms |        n/a |       10.74x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     8ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |    69ms |    n/a |        n/a |          n/a |              n/a |
| `nanquantile`             |                (1000,) |      1000 |     0ms |    0ms |        n/a |        1.01x |              n/a |
|                           |            (10000000,) |  10000000 |   212ms |  187ms |        n/a |        0.88x |              n/a |
|                           |          (100, 100000) |  10000000 |   213ms |  196ms |        n/a |        0.92x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |   214ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |  2045ms |    n/a |        n/a |          n/a |              n/a |

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
