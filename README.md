# Numbagg: Fast N-dimensional aggregation functions with Numba

[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/numbagg/numbagg/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/numbagg/numbagg/actions/workflows/test.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/numbagg?style=for-the-badge)](https://pypi.python.org/pypi/numbagg/)

Fast, flexible N-dimensional array functions written with
[Numba](https://github.com/numba/numba) and NumPy's [generalized
ufuncs](http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html).

## Why use numbagg?

### Performance

- Outperforms pandas
  - On a single core, 2-10x faster for moving window functions, 1-2x faster for
    aggregation and grouping functions
  - When parallelizing with multiple cores, 4-30x faster
- Outperforms bottleneck on multiple cores
  - On a single core, matches bottleneck
  - When parallelizing with multiple cores, 3-7x faster
- Outperforms numpy on multiple cores
  - On a single core, matches numpy
  - When parallelizing with multiple cores, 5-15x faster
- ...though numbagg's functions are JIT compiled, so the first run is much slower

### Versatility

- More functions (though bottleneck has some functions we don't have, and pandas' functions
  have many more parameters)
- Functions work for >3 dimensions. All functions take an arbitrary axis or
  tuple of axes to calculate over
- Written in numba — way less code, simple to inspect, simple to improve

## Functions & benchmarks

### Summary benchmark

Two benchmarks summarize numbagg's performance — the first with a 1D array without
parallelization, and a second with a 2D array with parallelization. Numbagg's relative
performance is much higher where parallelization is possible. A wider range of arrays is
listed in the full set of benchmarks below.

The values in the table are numbagg's performance as a multiple of other libraries for a
given shaped array calculated over the final axis. (so 1.00x means numbagg is equal,
higher means numbagg is faster.). A shape of `(100000000,)` means a 1D array with 100M
items.

| func                      | pandas<br>`(10000000,)` | bottleneck<br>`(10000000,)` | numpy<br>`(10000000,)` | pandas<br>`(100, 100000)` | bottleneck<br>`(100, 100000)` | numpy<br>`(100, 100000)` |
| :------------------------ | ----------------------: | --------------------------: | ---------------------: | ------------------------: | ----------------------------: | -----------------------: |
| `bfill`                   |                   1.16x |                       1.18x |                    n/a |                    11.31x |                         4.05x |                      n/a |
| `ffill`                   |                   1.21x |                       1.18x |                    n/a |                    12.60x |                         4.39x |                      n/a |
| `group_nanall`            |                   1.41x |                         n/a |                    n/a |                    10.70x |                           n/a |                      n/a |
| `group_nanany`            |                   1.16x |                         n/a |                    n/a |                     5.49x |                           n/a |                      n/a |
| `group_nanargmax`         |                   2.83x |                         n/a |                    n/a |                    11.44x |                           n/a |                      n/a |
| `group_nanargmin`         |                   2.85x |                         n/a |                    n/a |                    11.18x |                           n/a |                      n/a |
| `group_nancount`          |                   1.01x |                         n/a |                    n/a |                     4.62x |                           n/a |                      n/a |
| `group_nanfirst`          |                   1.38x |                         n/a |                    n/a |                    11.76x |                           n/a |                      n/a |
| `group_nanlast`           |                   1.12x |                         n/a |                    n/a |                     4.66x |                           n/a |                      n/a |
| `group_nanmax`            |                   1.12x |                         n/a |                    n/a |                     4.54x |                           n/a |                      n/a |
| `group_nanmean`           |                   1.16x |                         n/a |                    n/a |                     5.28x |                           n/a |                      n/a |
| `group_nanmin`            |                   1.13x |                         n/a |                    n/a |                     4.52x |                           n/a |                      n/a |
| `group_nanprod`           |                   1.09x |                         n/a |                    n/a |                     4.93x |                           n/a |                      n/a |
| `group_nanstd`            |                   1.21x |                         n/a |                    n/a |                     4.82x |                           n/a |                      n/a |
| `group_nansum_of_squares` |                   1.36x |                         n/a |                    n/a |                     2.62x |                           n/a |                      n/a |
| `group_nansum`            |                   1.22x |                         n/a |                    n/a |                     5.18x |                           n/a |                      n/a |
| `group_nanvar`            |                   1.19x |                         n/a |                    n/a |                     4.86x |                           n/a |                      n/a |
| `move_corr`               |                  19.56x |                         n/a |                    n/a |                    82.74x |                           n/a |                      n/a |
| `move_cov`                |                  14.46x |                         n/a |                    n/a |                    68.84x |                           n/a |                      n/a |
| `move_exp_nancorr`        |                   6.99x |                         n/a |                    n/a |                    30.75x |                           n/a |                      n/a |
| `move_exp_nancount`       |                   2.36x |                         n/a |                    n/a |                    10.49x |                           n/a |                      n/a |
| `move_exp_nancov`         |                   6.12x |                         n/a |                    n/a |                    30.65x |                           n/a |                      n/a |
| `move_exp_nanmean`        |                   2.17x |                         n/a |                    n/a |                    10.67x |                           n/a |                      n/a |
| `move_exp_nanstd`         |                   1.87x |                         n/a |                    n/a |                     9.98x |                           n/a |                      n/a |
| `move_exp_nansum`         |                   1.90x |                         n/a |                    n/a |                     8.17x |                           n/a |                      n/a |
| `move_exp_nanvar`         |                   1.81x |                         n/a |                    n/a |                     8.84x |                           n/a |                      n/a |
| `move_mean`               |                   3.85x |                       0.88x |                    n/a |                    17.69x |                         4.16x |                      n/a |
| `move_std`                |                   5.96x |                       1.32x |                    n/a |                    24.90x |                         6.61x |                      n/a |
| `move_sum`                |                   3.69x |                       0.85x |                    n/a |                    14.01x |                         3.33x |                      n/a |
| `move_var`                |                   5.66x |                       1.23x |                    n/a |                    26.11x |                         5.29x |                      n/a |
| `nanargmax`[^5]           |                   2.19x |                       0.99x |                    n/a |                     2.08x |                         1.00x |                      n/a |
| `nanargmin`[^5]           |                   2.52x |                       0.98x |                    n/a |                     2.14x |                         0.94x |                      n/a |
| `nancount`                |                   0.71x |                         n/a |                  0.37x |                     2.38x |                           n/a |                    0.72x |
| `nanmax`[^5]              |                   3.26x |                       1.00x |                  0.11x |                     3.63x |                         3.25x |                    0.11x |
| `nanmean`                 |                   2.35x |                       0.98x |                  2.80x |                    17.80x |                         6.03x |                   17.31x |
| `nanmin`[^5]              |                   3.25x |                       0.99x |                  0.10x |                     3.64x |                         3.26x |                    0.11x |
| `nanquantile`             |                   0.95x |                         n/a |                  0.79x |                     5.34x |                           n/a |                    4.94x |
| `nanstd`                  |                   1.49x |                       1.47x |                  2.72x |                    10.69x |                         9.17x |                   17.76x |
| `nansum`                  |                   2.13x |                       0.99x |                  3.34x |                    17.31x |                         5.81x |                   17.92x |
| `nanvar`                  |                   1.51x |                       1.48x |                  2.74x |                    10.76x |                         9.22x |                   17.15x |

### Full benchmarks

<details>

| func                      |                  shape |      size | pandas | bottleneck |  numpy | numbagg | pandas_ratio | bottleneck_ratio | numpy_ratio | numbagg_ratio |
| :------------------------ | ---------------------: | --------: | -----: | ---------: | -----: | ------: | -----------: | ---------------: | ----------: | ------------: |
| `bfill`                   |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        1.53x |            0.03x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   20ms |       21ms |    n/a |    18ms |        1.16x |            1.18x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   61ms |       22ms |    n/a |     5ms |       11.31x |            4.05x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       21ms |    n/a |     5ms |          n/a |            4.40x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      252ms |    n/a |    43ms |          n/a |            5.86x |         n/a |         1.00x |
| `ffill`                   |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        1.54x |            0.02x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   20ms |       20ms |    n/a |    17ms |        1.21x |            1.18x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   57ms |       20ms |    n/a |     5ms |       12.60x |            4.39x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       20ms |    n/a |     5ms |          n/a |            3.85x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      236ms |    n/a |    42ms |          n/a |            5.59x |         n/a |         1.00x |
| `group_nanall`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.85x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   67ms |        n/a |    n/a |    47ms |        1.41x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   17ms |        n/a |    n/a |     2ms |       10.70x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     1ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanany`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.84x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   67ms |        n/a |    n/a |    57ms |        1.16x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   17ms |        n/a |    n/a |     3ms |        5.49x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanargmax`         |                (1000,) |      1000 |    1ms |        n/a |    n/a |     0ms |       16.59x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  164ms |        n/a |    n/a |    58ms |        2.83x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   39ms |        n/a |    n/a |     3ms |       11.44x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanargmin`         |                (1000,) |      1000 |    1ms |        n/a |    n/a |     0ms |       17.93x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  166ms |        n/a |    n/a |    58ms |        2.85x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   39ms |        n/a |    n/a |     4ms |       11.18x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nancount`          |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.68x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   56ms |        n/a |    n/a |    56ms |        1.01x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   15ms |        n/a |    n/a |     3ms |        4.62x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanfirst`          |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.86x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   64ms |        n/a |    n/a |    46ms |        1.38x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   16ms |        n/a |    n/a |     1ms |       11.76x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     1ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanlast`           |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.85x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   65ms |        n/a |    n/a |    58ms |        1.12x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   15ms |        n/a |    n/a |     3ms |        4.66x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanmax`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.81x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   67ms |        n/a |    n/a |    60ms |        1.12x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   17ms |        n/a |    n/a |     4ms |        4.54x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanmean`           |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.43x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   70ms |        n/a |    n/a |    60ms |        1.16x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   19ms |        n/a |    n/a |     4ms |        5.28x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanmin`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.87x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   66ms |        n/a |    n/a |    58ms |        1.13x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   17ms |        n/a |    n/a |     4ms |        4.52x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanprod`           |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.09x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   64ms |        n/a |    n/a |    58ms |        1.09x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   16ms |        n/a |    n/a |     3ms |        4.93x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanstd`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.89x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   74ms |        n/a |    n/a |    61ms |        1.21x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   20ms |        n/a |    n/a |     4ms |        4.82x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nansum`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.87x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   69ms |        n/a |    n/a |    56ms |        1.22x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   19ms |        n/a |    n/a |     4ms |        5.18x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanvar`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.64x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   70ms |        n/a |    n/a |    59ms |        1.19x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   21ms |        n/a |    n/a |     4ms |        4.86x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nansum_of_squares` |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        0.52x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  125ms |        n/a |    n/a |    92ms |        1.36x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   33ms |        n/a |    n/a |    13ms |        2.62x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     7ms |          n/a |              n/a |         n/a |         1.00x |
| `move_corr`               |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |       10.60x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  938ms |        n/a |    n/a |    48ms |       19.56x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  874ms |        n/a |    n/a |    11ms |       82.74x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |    10ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    79ms |          n/a |              n/a |         n/a |         1.00x |
| `move_cov`                |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        9.43x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  624ms |        n/a |    n/a |    43ms |       14.46x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  631ms |        n/a |    n/a |     9ms |       68.84x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     9ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    76ms |          n/a |              n/a |         n/a |         1.00x |
| `move_mean`               |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        1.84x |            0.03x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  123ms |       28ms |    n/a |    32ms |        3.85x |            0.88x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  120ms |       28ms |    n/a |     7ms |       17.69x |            4.16x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       28ms |    n/a |     8ms |          n/a |            3.62x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      306ms |    n/a |    64ms |          n/a |            4.75x |         n/a |         1.00x |
| `move_std`                |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        2.17x |            0.08x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  179ms |       40ms |    n/a |    30ms |        5.96x |            1.32x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  162ms |       43ms |    n/a |     6ms |       24.90x |            6.61x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       40ms |    n/a |     8ms |          n/a |            5.23x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      427ms |    n/a |    68ms |          n/a |            6.25x |         n/a |         1.00x |
| `move_sum`                |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        1.80x |            0.02x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  120ms |       28ms |    n/a |    32ms |        3.69x |            0.85x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  113ms |       27ms |    n/a |     8ms |       14.01x |            3.33x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       27ms |    n/a |     8ms |          n/a |            3.40x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      290ms |    n/a |    62ms |          n/a |            4.66x |         n/a |         1.00x |
| `move_var`                |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        1.98x |            0.08x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  171ms |       37ms |    n/a |    30ms |        5.66x |            1.23x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  182ms |       37ms |    n/a |     7ms |       26.11x |            5.29x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       37ms |    n/a |     6ms |          n/a |            5.81x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      398ms |    n/a |    56ms |          n/a |            7.14x |         n/a |         1.00x |
| `move_exp_nancorr`        |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        6.91x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  485ms |        n/a |    n/a |    69ms |        6.99x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  483ms |        n/a |    n/a |    16ms |       30.75x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |    15ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |   133ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nancount`       |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        2.02x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   77ms |        n/a |    n/a |    33ms |        2.36x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   69ms |        n/a |    n/a |     7ms |       10.49x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     7ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    59ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nancov`         |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        6.78x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  322ms |        n/a |    n/a |    53ms |        6.12x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  370ms |        n/a |    n/a |    12ms |       30.65x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |    12ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |   113ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nanmean`        |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.41x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   72ms |        n/a |    n/a |    33ms |        2.17x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   76ms |        n/a |    n/a |     7ms |       10.67x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     8ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    60ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nanstd`         |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        2.30x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   87ms |        n/a |    n/a |    46ms |        1.87x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   94ms |        n/a |    n/a |     9ms |        9.98x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |    10ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    77ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nansum`         |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.27x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   64ms |        n/a |    n/a |    34ms |        1.90x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   69ms |        n/a |    n/a |     8ms |        8.17x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     6ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    61ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nanvar`         |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.41x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   78ms |        n/a |    n/a |    43ms |        1.81x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   88ms |        n/a |    n/a |    10ms |        8.84x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     9ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    73ms |          n/a |              n/a |         n/a |         1.00x |
| `nanargmax`[^5]           |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |       12.97x |            0.21x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   28ms |       13ms |    n/a |    13ms |        2.19x |            0.99x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   26ms |       12ms |    n/a |    12ms |        2.08x |            1.00x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       14ms |    n/a |    13ms |          n/a |            1.04x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      145ms |    n/a |   133ms |          n/a |            1.09x |         n/a |         1.00x |
| `nanargmin`[^5]           |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |       13.63x |            0.21x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   32ms |       12ms |    n/a |    13ms |        2.52x |            0.98x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   28ms |       12ms |    n/a |    13ms |        2.14x |            0.94x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       16ms |    n/a |    14ms |          n/a |            1.17x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      133ms |    n/a |   127ms |          n/a |            1.05x |         n/a |         1.00x |
| `nancount`                |                (1000,) |      1000 |    0ms |        n/a |    0ms |     1ms |        0.10x |              n/a |       0.00x |         1.00x |
|                           |            (10000000,) |  10000000 |    7ms |        n/a |    3ms |     9ms |        0.71x |              n/a |       0.37x |         1.00x |
|                           |          (100, 100000) |  10000000 |   11ms |        n/a |    3ms |     5ms |        2.38x |              n/a |       0.72x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    3ms |     1ms |          n/a |              n/a |       3.43x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |   41ms |    25ms |          n/a |              n/a |       1.60x |         1.00x |
| `nanmax`[^5]              |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        8.16x |            0.21x |       0.37x |         1.00x |
|                           |            (10000000,) |  10000000 |   41ms |       12ms |    1ms |    12ms |        3.26x |            1.00x |       0.11x |         1.00x |
|                           |          (100, 100000) |  10000000 |   45ms |       40ms |    1ms |    12ms |        3.63x |            3.25x |       0.11x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       40ms |    1ms |    12ms |          n/a |            3.30x |       0.12x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      401ms |   14ms |   121ms |          n/a |            3.31x |       0.12x |         1.00x |
| `nanmean`                 |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        1.34x |            0.02x |       0.20x |         1.00x |
|                           |            (10000000,) |  10000000 |   22ms |        9ms |   27ms |    10ms |        2.35x |            0.98x |       2.80x |         1.00x |
|                           |          (100, 100000) |  10000000 |   28ms |        9ms |   27ms |     2ms |       17.80x |            6.03x |      17.31x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        9ms |   26ms |     2ms |          n/a |            5.88x |      17.12x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |       90ms |  297ms |    14ms |          n/a |            6.60x |      21.79x |         1.00x |
| `nanmin`[^5]              |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        8.14x |            0.21x |       0.37x |         1.00x |
|                           |            (10000000,) |  10000000 |   41ms |       12ms |    1ms |    13ms |        3.25x |            0.99x |       0.10x |         1.00x |
|                           |          (100, 100000) |  10000000 |   45ms |       41ms |    1ms |    12ms |        3.64x |            3.26x |       0.11x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       40ms |    1ms |    12ms |          n/a |            3.30x |       0.12x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      406ms |   14ms |   122ms |          n/a |            3.33x |       0.11x |         1.00x |
| `nanquantile`             |                (1000,) |      1000 |    0ms |        n/a |    0ms |     0ms |        1.42x |              n/a |       0.55x |         1.00x |
|                           |            (10000000,) |  10000000 |  189ms |        n/a |  158ms |   199ms |        0.95x |              n/a |       0.79x |         1.00x |
|                           |          (100, 100000) |  10000000 |  197ms |        n/a |  182ms |    37ms |        5.34x |              n/a |       4.94x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |  426ms |    38ms |          n/a |              n/a |      11.27x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a | 4308ms |   345ms |          n/a |              n/a |      12.48x |         1.00x |
| `nanstd`                  |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        1.07x |            0.06x |       0.46x |         1.00x |
|                           |            (10000000,) |  10000000 |   28ms |       28ms |   52ms |    19ms |        1.49x |            1.47x |       2.72x |         1.00x |
|                           |          (100, 100000) |  10000000 |   33ms |       28ms |   55ms |     3ms |       10.69x |            9.17x |      17.76x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       28ms |   55ms |     3ms |          n/a |            8.48x |      16.69x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      279ms |  586ms |    27ms |          n/a |           10.40x |      21.87x |         1.00x |
| `nansum`                  |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        1.31x |            0.02x |       0.08x |         1.00x |
|                           |            (10000000,) |  10000000 |   20ms |        9ms |   32ms |    10ms |        2.13x |            0.99x |       3.34x |         1.00x |
|                           |          (100, 100000) |  10000000 |   28ms |        9ms |   29ms |     2ms |       17.31x |            5.81x |      17.92x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       10ms |   24ms |     8ms |          n/a |            1.15x |       2.86x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |       95ms |  265ms |    17ms |          n/a |            5.67x |      15.78x |         1.00x |
| `nanvar`                  |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        1.07x |            0.06x |       0.45x |         1.00x |
|                           |            (10000000,) |  10000000 |   28ms |       28ms |   52ms |    19ms |        1.51x |            1.48x |       2.74x |         1.00x |
|                           |          (100, 100000) |  10000000 |   33ms |       28ms |   52ms |     3ms |       10.76x |            9.22x |      17.15x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       28ms |   57ms |     3ms |          n/a |            8.40x |      17.02x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      278ms |  580ms |    26ms |          n/a |           10.78x |      22.46x |         1.00x |

</details>

[^1][^2][^3][^4]

[^1]:
    Benchmarks were run on a Mac M1 laptop in December 2023 on numbagg's HEAD,
    pandas 2.1.1, bottleneck 1.3.7. The run in CI, though without demonstrating
    the full benefits of parallelization given GHA's low CPU count.

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
    This function is not currently parallelized, so exhibits worse performance
    on parallelizable arrays.

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
