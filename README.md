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

- Faster than pandas for most functions — 2-20x
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

### Summary

Two benchmarks summarize numbagg's performance — one with a 1D array with no
parallelization, and one with a 2D array with the potential for parallelization.
Numbagg's relative performance is much higher where parallelization is possible.

The values in the table are numbagg's performance as a multiple of other libraries for a
given shaped array, calculated over the final axis. (so 1.0x means equal to numbagg,
higher means slower than numbagg.)

| func                      | pandas, `(10000000,)` | bottleneck, `(10000000,)` | pandas, `(100, 100000)` | bottleneck, `(100, 100000)` |
| :------------------------ | --------------------: | ------------------------: | ----------------------: | --------------------------: |
| `bfill`                   |                 1.18x |                     1.21x |                  12.35x |                       4.28x |
| `ffill`                   |                 1.14x |                     1.16x |                  14.24x |                       4.53x |
| `group_nanall`            |                 1.56x |                       n/a |                   7.49x |                         n/a |
| `group_nanany`            |                 1.25x |                       n/a |                   4.61x |                         n/a |
| `group_nanargmax`         |                 3.03x |                       n/a |                   8.38x |                         n/a |
| `group_nanargmin`         |                 2.92x |                       n/a |                   9.28x |                         n/a |
| `group_nancount`          |                 1.06x |                       n/a |                   3.98x |                         n/a |
| `group_nanfirst`          |                 1.42x |                       n/a |                   8.35x |                         n/a |
| `group_nanlast`           |                 1.21x |                       n/a |                   4.61x |                         n/a |
| `group_nanmax`            |                 1.12x |                       n/a |                   3.81x |                         n/a |
| `group_nanmean`           |                 1.23x |                       n/a |                   4.90x |                         n/a |
| `group_nanmin`            |                 1.18x |                       n/a |                   3.88x |                         n/a |
| `group_nanprod`           |                 1.21x |                       n/a |                   4.18x |                         n/a |
| `group_nanstd`            |                 1.33x |                       n/a |                   4.25x |                         n/a |
| `group_nansum_of_squares` |                 1.42x |                       n/a |                   6.95x |                         n/a |
| `group_nansum`            |                 1.29x |                       n/a |                   4.86x |                         n/a |
| `group_nanvar`            |                 1.21x |                       n/a |                   4.52x |                         n/a |
| `move_corr`               |                17.37x |                       n/a |                  80.76x |                         n/a |
| `move_cov`                |                15.06x |                       n/a |                  71.25x |                         n/a |
| `move_exp_nancorr`        |                 6.68x |                       n/a |                  32.74x |                         n/a |
| `move_exp_nancount`       |                 2.59x |                       n/a |                   8.08x |                         n/a |
| `move_exp_nancov`         |                 6.03x |                       n/a |                  30.65x |                         n/a |
| `move_exp_nanmean`        |                 2.11x |                       n/a |                  11.79x |                         n/a |
| `move_exp_nanstd`         |                 1.86x |                       n/a |                   9.70x |                         n/a |
| `move_exp_nansum`         |                 2.03x |                       n/a |                   8.46x |                         n/a |
| `move_exp_nanvar`         |                 1.94x |                       n/a |                   9.67x |                         n/a |
| `move_mean`               |                 4.07x |                     0.89x |                  12.33x |                       3.21x |
| `move_std`                |                 6.11x |                     1.12x |                  22.54x |                       4.65x |
| `move_sum`                |                 4.22x |                     0.87x |                  15.50x |                       3.50x |
| `move_var`                |                 5.76x |                     1.07x |                  26.96x |                       5.32x |
| `nanargmax`               |                 2.42x |                     1.09x |                   2.13x |                       0.97x |
| `nanargmin`               |                 2.39x |                     1.06x |                   2.25x |                       1.01x |
| `nancount`                |                 1.63x |                       n/a |                   8.67x |                         n/a |
| `nanmax`                  |                 0.97x |                     0.91x |                   1.43x |                       0.99x |
| `nanmean`                 |                 2.54x |                     0.98x |                  12.29x |                       4.48x |
| `nanmin`                  |                 1.03x |                     1.03x |                   1.39x |                       1.00x |
| `nanquantile`             |                 0.91x |                       n/a |                   0.98x |                         n/a |
| `nanstd`                  |                 1.47x |                     1.51x |                   7.92x |                       6.91x |
| `nansum`                  |                 2.17x |                     0.98x |                  13.85x |                       4.19x |
| `nanvar`                  |                 1.63x |                     1.48x |                   7.34x |                       5.90x |

### Full benchmarks

<details>

| func                      |                  shape |      size | numbagg | pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------------ | ---------------------: | --------: | ------: | -----: | ---------: | -----------: | ---------------: |
| `bfill`                   |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.51x |            0.01x |
|                           |            (10000000,) |  10000000 |    18ms |   21ms |       22ms |        1.18x |            1.21x |
|                           |          (100, 100000) |  10000000 |     5ms |   64ms |       22ms |       12.35x |            4.28x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |       23ms |          n/a |            5.36x |
|                           |      (100, 1000, 1000) | 100000000 |    53ms |    n/a |      278ms |          n/a |            5.22x |
| `ffill`                   |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.51x |            0.01x |
|                           |            (10000000,) |  10000000 |    18ms |   21ms |       21ms |        1.14x |            1.16x |
|                           |          (100, 100000) |  10000000 |     4ms |   64ms |       20ms |       14.24x |            4.53x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       20ms |          n/a |            3.94x |
|                           |      (100, 1000, 1000) | 100000000 |    54ms |    n/a |      255ms |          n/a |            4.68x |
| `group_nanall`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.65x |              n/a |
|                           |            (10000000,) |  10000000 |    52ms |   81ms |        n/a |        1.56x |              n/a |
|                           |          (100, 100000) |  10000000 |     3ms |   19ms |        n/a |        7.49x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     1ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanany`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.63x |              n/a |
|                           |            (10000000,) |  10000000 |    62ms |   77ms |        n/a |        1.25x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   21ms |        n/a |        4.61x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     3ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanargmax`         |                (1000,) |      1000 |     0ms |    1ms |        n/a |        6.48x |              n/a |
|                           |            (10000000,) |  10000000 |    68ms |  205ms |        n/a |        3.03x |              n/a |
|                           |          (100, 100000) |  10000000 |     6ms |   48ms |        n/a |        8.38x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanargmin`         |                (1000,) |      1000 |     0ms |    1ms |        n/a |        6.68x |              n/a |
|                           |            (10000000,) |  10000000 |    64ms |  187ms |        n/a |        2.92x |              n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   46ms |        n/a |        9.28x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |        n/a |          n/a |              n/a |
| `group_nancount`          |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.61x |              n/a |
|                           |            (10000000,) |  10000000 |    63ms |   67ms |        n/a |        1.06x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   17ms |        n/a |        3.98x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     3ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanfirst`          |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.83x |              n/a |
|                           |            (10000000,) |  10000000 |    53ms |   75ms |        n/a |        1.42x |              n/a |
|                           |          (100, 100000) |  10000000 |     2ms |   17ms |        n/a |        8.35x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     2ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanlast`           |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.83x |              n/a |
|                           |            (10000000,) |  10000000 |    61ms |   74ms |        n/a |        1.21x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   17ms |        n/a |        4.61x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     3ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanmax`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.68x |              n/a |
|                           |            (10000000,) |  10000000 |    65ms |   72ms |        n/a |        1.12x |              n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   19ms |        n/a |        3.81x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanmean`           |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.79x |              n/a |
|                           |            (10000000,) |  10000000 |    64ms |   79ms |        n/a |        1.23x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   22ms |        n/a |        4.90x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanmin`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.84x |              n/a |
|                           |            (10000000,) |  10000000 |    66ms |   77ms |        n/a |        1.18x |              n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   19ms |        n/a |        3.88x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanprod`           |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.82x |              n/a |
|                           |            (10000000,) |  10000000 |    62ms |   76ms |        n/a |        1.21x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   18ms |        n/a |        4.18x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanstd`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.66x |              n/a |
|                           |            (10000000,) |  10000000 |    64ms |   85ms |        n/a |        1.33x |              n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   22ms |        n/a |        4.25x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |          n/a |              n/a |
| `group_nansum`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.81x |              n/a |
|                           |            (10000000,) |  10000000 |    63ms |   81ms |        n/a |        1.29x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   20ms |        n/a |        4.86x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |          n/a |              n/a |
| `group_nanvar`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.73x |              n/a |
|                           |            (10000000,) |  10000000 |    66ms |   79ms |        n/a |        1.21x |              n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   22ms |        n/a |        4.52x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |          n/a |              n/a |
| `group_nansum_of_squares` |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.86x |              n/a |
|                           |            (10000000,) |  10000000 |    61ms |   87ms |        n/a |        1.42x |              n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   31ms |        n/a |        6.95x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     3ms |    n/a |        n/a |          n/a |              n/a |
| `move_corr`               |                (1000,) |      1000 |     0ms |    1ms |        n/a |        3.54x |              n/a |
|                           |            (10000000,) |  10000000 |    58ms | 1013ms |        n/a |       17.37x |              n/a |
|                           |          (100, 100000) |  10000000 |    12ms |  948ms |        n/a |       80.76x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    10ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |   208ms |    n/a |        n/a |          n/a |              n/a |
| `move_cov`                |                (1000,) |      1000 |     0ms |    0ms |        n/a |        3.09x |              n/a |
|                           |            (10000000,) |  10000000 |    46ms |  700ms |        n/a |       15.06x |              n/a |
|                           |          (100, 100000) |  10000000 |     9ms |  634ms |        n/a |       71.25x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     9ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |   185ms |    n/a |        n/a |          n/a |              n/a |
| `move_mean`               |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.61x |            0.01x |
|                           |            (10000000,) |  10000000 |    32ms |  132ms |       29ms |        4.07x |            0.89x |
|                           |          (100, 100000) |  10000000 |    10ms |  118ms |       31ms |       12.33x |            3.21x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     7ms |    n/a |       28ms |          n/a |            4.14x |
|                           |      (100, 1000, 1000) | 100000000 |   107ms |    n/a |      355ms |          n/a |            3.33x |
| `move_std`                |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.71x |            0.02x |
|                           |            (10000000,) |  10000000 |    32ms |  198ms |       36ms |        6.11x |            1.12x |
|                           |          (100, 100000) |  10000000 |     8ms |  181ms |       37ms |       22.54x |            4.65x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     8ms |    n/a |       36ms |          n/a |            4.75x |
|                           |      (100, 1000, 1000) | 100000000 |    81ms |    n/a |      404ms |          n/a |            4.96x |
| `move_sum`                |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.78x |            0.01x |
|                           |            (10000000,) |  10000000 |    32ms |  135ms |       28ms |        4.22x |            0.87x |
|                           |          (100, 100000) |  10000000 |     8ms |  116ms |       26ms |       15.50x |            3.50x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     7ms |    n/a |       26ms |          n/a |            3.74x |
|                           |      (100, 1000, 1000) | 100000000 |    76ms |    n/a |      302ms |          n/a |            3.97x |
| `move_var`                |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.63x |            0.02x |
|                           |            (10000000,) |  10000000 |    32ms |  183ms |       34ms |        5.76x |            1.07x |
|                           |          (100, 100000) |  10000000 |     6ms |  175ms |       35ms |       26.96x |            5.32x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     7ms |    n/a |       34ms |          n/a |            5.15x |
|                           |      (100, 1000, 1000) | 100000000 |   107ms |    n/a |      432ms |          n/a |            4.03x |
| `move_exp_nancorr`        |                (1000,) |      1000 |     0ms |    0ms |        n/a |        2.82x |              n/a |
|                           |            (10000000,) |  10000000 |    74ms |  492ms |        n/a |        6.68x |              n/a |
|                           |          (100, 100000) |  10000000 |    15ms |  478ms |        n/a |       32.74x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    15ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |   145ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancount`       |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.83x |              n/a |
|                           |            (10000000,) |  10000000 |    35ms |   90ms |        n/a |        2.59x |              n/a |
|                           |          (100, 100000) |  10000000 |    10ms |   79ms |        n/a |        8.08x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     7ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |    85ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nancov`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |        2.49x |              n/a |
|                           |            (10000000,) |  10000000 |    55ms |  329ms |        n/a |        6.03x |              n/a |
|                           |          (100, 100000) |  10000000 |    11ms |  342ms |        n/a |       30.65x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    13ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |   202ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanmean`        |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.60x |              n/a |
|                           |            (10000000,) |  10000000 |    36ms |   76ms |        n/a |        2.11x |              n/a |
|                           |          (100, 100000) |  10000000 |     8ms |   88ms |        n/a |       11.79x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     9ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |   174ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanstd`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.87x |              n/a |
|                           |            (10000000,) |  10000000 |    50ms |   94ms |        n/a |        1.86x |              n/a |
|                           |          (100, 100000) |  10000000 |    10ms |  100ms |        n/a |        9.70x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    10ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |   117ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nansum`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.60x |              n/a |
|                           |            (10000000,) |  10000000 |    34ms |   69ms |        n/a |        2.03x |              n/a |
|                           |          (100, 100000) |  10000000 |     9ms |   73ms |        n/a |        8.46x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |   127ms |    n/a |        n/a |          n/a |              n/a |
| `move_exp_nanvar`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.49x |              n/a |
|                           |            (10000000,) |  10000000 |    46ms |   89ms |        n/a |        1.94x |              n/a |
|                           |          (100, 100000) |  10000000 |     9ms |   91ms |        n/a |        9.67x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     9ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |   113ms |    n/a |        n/a |          n/a |              n/a |
| `nanargmax`               |                (1000,) |      1000 |     0ms |    0ms |        0ms |       15.10x |            0.22x |
|                           |            (10000000,) |  10000000 |    13ms |   32ms |       14ms |        2.42x |            1.09x |
|                           |          (100, 100000) |  10000000 |    14ms |   30ms |       13ms |        2.13x |            0.97x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    13ms |    n/a |       15ms |          n/a |            1.11x |
|                           |      (100, 1000, 1000) | 100000000 |   138ms |    n/a |      147ms |          n/a |            1.06x |
| `nanargmin`               |                (1000,) |      1000 |     0ms |    0ms |        0ms |       13.05x |            0.20x |
|                           |            (10000000,) |  10000000 |    13ms |   32ms |       14ms |        2.39x |            1.06x |
|                           |          (100, 100000) |  10000000 |    13ms |   30ms |       14ms |        2.25x |            1.01x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    14ms |    n/a |       15ms |          n/a |            1.04x |
|                           |      (100, 1000, 1000) | 100000000 |   142ms |    n/a |      145ms |          n/a |            1.02x |
| `nancount`                |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.73x |              n/a |
|                           |            (10000000,) |  10000000 |     4ms |    6ms |        n/a |        1.63x |              n/a |
|                           |          (100, 100000) |  10000000 |     1ms |   10ms |        n/a |        8.67x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     1ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |     8ms |    n/a |        n/a |          n/a |              n/a |
| `nanmax`                  |                (1000,) |      1000 |     0ms |    0ms |        0ms |        8.06x |            0.20x |
|                           |            (10000000,) |  10000000 |    15ms |   14ms |       13ms |        0.97x |            0.91x |
|                           |          (100, 100000) |  10000000 |    14ms |   19ms |       14ms |        1.43x |            0.99x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    13ms |    n/a |       13ms |          n/a |            1.00x |
|                           |      (100, 1000, 1000) | 100000000 |   130ms |    n/a |      130ms |          n/a |            1.01x |
| `nanmean`                 |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.45x |            0.01x |
|                           |            (10000000,) |  10000000 |    10ms |   26ms |       10ms |        2.54x |            0.98x |
|                           |          (100, 100000) |  10000000 |     2ms |   30ms |       11ms |       12.29x |            4.48x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     2ms |    n/a |       11ms |          n/a |            5.15x |
|                           |      (100, 1000, 1000) | 100000000 |    27ms |    n/a |       95ms |          n/a |            3.51x |
| `nanmin`                  |                (1000,) |      1000 |     0ms |    0ms |        0ms |        7.99x |            0.22x |
|                           |            (10000000,) |  10000000 |    14ms |   14ms |       14ms |        1.03x |            1.03x |
|                           |          (100, 100000) |  10000000 |    14ms |   19ms |       14ms |        1.39x |            1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    13ms |    n/a |       12ms |          n/a |            0.97x |
|                           |      (100, 1000, 1000) | 100000000 |   130ms |    n/a |      133ms |          n/a |            1.03x |
| `nanquantile`             |                (1000,) |      1000 |     0ms |    0ms |        n/a |        0.93x |              n/a |
|                           |            (10000000,) |  10000000 |   228ms |  207ms |        n/a |        0.91x |              n/a |
|                           |          (100, 100000) |  10000000 |   224ms |  220ms |        n/a |        0.98x |              n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |   230ms |    n/a |        n/a |          n/a |              n/a |
|                           |      (100, 1000, 1000) | 100000000 |  2208ms |    n/a |        n/a |          n/a |              n/a |
| `nanstd`                  |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.37x |            0.02x |
|                           |            (10000000,) |  10000000 |    20ms |   29ms |       30ms |        1.47x |            1.51x |
|                           |          (100, 100000) |  10000000 |     4ms |   34ms |       30ms |        7.92x |            6.91x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       31ms |          n/a |            6.42x |
|                           |      (100, 1000, 1000) | 100000000 |    41ms |    n/a |      296ms |          n/a |            7.18x |
| `nansum`                  |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.85x |            0.01x |
|                           |            (10000000,) |  10000000 |    11ms |   24ms |       11ms |        2.17x |            0.98x |
|                           |          (100, 100000) |  10000000 |     2ms |   32ms |       10ms |       13.85x |            4.19x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     2ms |    n/a |       10ms |          n/a |            4.49x |
|                           |      (100, 1000, 1000) | 100000000 |    18ms |    n/a |      101ms |          n/a |            5.50x |
| `nanvar`                  |                (1000,) |      1000 |     0ms |    0ms |        0ms |        0.48x |            0.02x |
|                           |            (10000000,) |  10000000 |    20ms |   33ms |       30ms |        1.63x |            1.48x |
|                           |          (100, 100000) |  10000000 |     5ms |   36ms |       29ms |        7.34x |            5.90x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       29ms |          n/a |            6.00x |
|                           |      (100, 1000, 1000) | 100000000 |    44ms |    n/a |      295ms |          n/a |            6.70x |

</details>

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

[^4]:
    `anynan` & `allnan` are also functions in numbagg, but not listed here as they
    require a different benchmark setup.

[^5]:
    `nanmin`, `nanmax`, `nanargmin` & `nanargmax` are not currently parallelized,
    so exhibit worse performance on parallelizable arrays.

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
