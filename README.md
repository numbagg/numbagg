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
| `bfill`                   |                   1.17x |                       1.18x |                    n/a |                    12.24x |                         4.36x |                      n/a |
| `ffill`                   |                   1.17x |                       1.12x |                    n/a |                    12.76x |                         4.34x |                      n/a |
| `group_nanall`            |                   1.44x |                         n/a |                    n/a |                    10.84x |                           n/a |                      n/a |
| `group_nanany`            |                   1.20x |                         n/a |                    n/a |                     5.25x |                           n/a |                      n/a |
| `group_nanargmax`         |                   2.88x |                         n/a |                    n/a |                     9.89x |                           n/a |                      n/a |
| `group_nanargmin`         |                   2.82x |                         n/a |                    n/a |                     9.96x |                           n/a |                      n/a |
| `group_nancount`          |                   1.01x |                         n/a |                    n/a |                     4.70x |                           n/a |                      n/a |
| `group_nanfirst`          |                   1.39x |                         n/a |                    n/a |                    11.80x |                           n/a |                      n/a |
| `group_nanlast`           |                   1.16x |                         n/a |                    n/a |                     5.36x |                           n/a |                      n/a |
| `group_nanmax`            |                   1.14x |                         n/a |                    n/a |                     5.22x |                           n/a |                      n/a |
| `group_nanmean`           |                   1.19x |                         n/a |                    n/a |                     5.64x |                           n/a |                      n/a |
| `group_nanmin`            |                   1.13x |                         n/a |                    n/a |                     5.26x |                           n/a |                      n/a |
| `group_nanprod`           |                   1.15x |                         n/a |                    n/a |                     4.95x |                           n/a |                      n/a |
| `group_nanstd`            |                   1.18x |                         n/a |                    n/a |                     5.03x |                           n/a |                      n/a |
| `group_nansum_of_squares` |                   1.35x |                         n/a |                    n/a |                     8.11x |                           n/a |                      n/a |
| `group_nansum`            |                   1.21x |                         n/a |                    n/a |                     5.95x |                           n/a |                      n/a |
| `group_nanvar`            |                   1.19x |                         n/a |                    n/a |                     5.65x |                           n/a |                      n/a |
| `move_corr`               |                  19.04x |                         n/a |                    n/a |                    92.48x |                           n/a |                      n/a |
| `move_cov`                |                  14.58x |                         n/a |                    n/a |                    71.61x |                           n/a |                      n/a |
| `move_exp_nancorr`        |                   6.73x |                         n/a |                    n/a |                    35.30x |                           n/a |                      n/a |
| `move_exp_nancount`       |                   2.35x |                         n/a |                    n/a |                    10.56x |                           n/a |                      n/a |
| `move_exp_nancov`         |                   5.77x |                         n/a |                    n/a |                    31.75x |                           n/a |                      n/a |
| `move_exp_nanmean`        |                   2.03x |                         n/a |                    n/a |                    11.07x |                           n/a |                      n/a |
| `move_exp_nanstd`         |                   1.89x |                         n/a |                    n/a |                    10.07x |                           n/a |                      n/a |
| `move_exp_nansum`         |                   1.88x |                         n/a |                    n/a |                     9.70x |                           n/a |                      n/a |
| `move_exp_nanvar`         |                   1.82x |                         n/a |                    n/a |                     9.71x |                           n/a |                      n/a |
| `move_mean`               |                   3.82x |                       0.87x |                    n/a |                    16.61x |                         4.01x |                      n/a |
| `move_std`                |                   5.96x |                       1.29x |                    n/a |                    24.52x |                         6.04x |                      n/a |
| `move_sum`                |                   3.80x |                       0.83x |                    n/a |                    15.95x |                         3.70x |                      n/a |
| `move_var`                |                   5.78x |                       1.27x |                    n/a |                    25.41x |                         5.85x |                      n/a |
| `nanargmax`[^5]           |                   2.45x |                       1.00x |                    n/a |                     2.16x |                         1.00x |                      n/a |
| `nanargmin`[^5]           |                   2.19x |                       1.01x |                    n/a |                     2.05x |                         1.02x |                      n/a |
| `nancount`                |                   1.40x |                         n/a |                  1.06x |                    11.00x |                           n/a |                    4.16x |
| `nanmax`[^5]              |                   3.26x |                       1.00x |                  0.11x |                     3.62x |                         3.24x |                    0.11x |
| `nanmean`                 |                   2.42x |                       0.98x |                  2.83x |                    13.58x |                         4.54x |                   13.13x |
| `nanmin`[^5]              |                   3.27x |                       1.00x |                  0.11x |                     3.62x |                         3.24x |                    0.11x |
| `nanquantile`             |                   0.94x |                         n/a |                  0.78x |                     5.45x |                           n/a |                    5.01x |
| `nanstd`                  |                   1.50x |                       1.51x |                  2.75x |                     8.29x |                         7.35x |                   13.27x |
| `nansum`                  |                   2.28x |                       0.97x |                  2.52x |                    17.71x |                         6.24x |                   16.05x |
| `nanvar`                  |                   1.50x |                       1.49x |                  2.81x |                     8.18x |                         6.97x |                   13.32x |

### Full benchmarks

<details>

| func                      |                  shape |      size | pandas | bottleneck |  numpy | numbagg | pandas_ratio | bottleneck_ratio | numpy_ratio | numbagg_ratio |
| :------------------------ | ---------------------: | --------: | -----: | ---------: | -----: | ------: | -----------: | ---------------: | ----------: | ------------: |
| `bfill`                   |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        1.59x |            0.03x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   20ms |       20ms |    n/a |    17ms |        1.17x |            1.18x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   57ms |       20ms |    n/a |     5ms |       12.24x |            4.36x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       21ms |    n/a |     5ms |          n/a |            4.40x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      248ms |    n/a |    44ms |          n/a |            5.70x |         n/a |         1.00x |
| `ffill`                   |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        1.53x |            0.02x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   20ms |       19ms |    n/a |    17ms |        1.17x |            1.12x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   56ms |       19ms |    n/a |     4ms |       12.76x |            4.34x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       19ms |    n/a |     4ms |          n/a |            4.33x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      219ms |    n/a |    42ms |          n/a |            5.25x |         n/a |         1.00x |
| `group_nanall`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.79x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   68ms |        n/a |    n/a |    47ms |        1.44x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   17ms |        n/a |    n/a |     2ms |       10.84x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     1ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanany`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.78x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   68ms |        n/a |    n/a |    56ms |        1.20x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   18ms |        n/a |    n/a |     3ms |        5.25x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanargmax`         |                (1000,) |      1000 |    1ms |        n/a |    n/a |     0ms |       17.60x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  171ms |        n/a |    n/a |    59ms |        2.88x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   40ms |        n/a |    n/a |     4ms |        9.89x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanargmin`         |                (1000,) |      1000 |    1ms |        n/a |    n/a |     0ms |       17.56x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  166ms |        n/a |    n/a |    59ms |        2.82x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   41ms |        n/a |    n/a |     4ms |        9.96x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nancount`          |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.68x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   56ms |        n/a |    n/a |    55ms |        1.01x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   15ms |        n/a |    n/a |     3ms |        4.70x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanfirst`          |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.88x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   63ms |        n/a |    n/a |    45ms |        1.39x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   15ms |        n/a |    n/a |     1ms |       11.80x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     1ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanlast`           |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.87x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   62ms |        n/a |    n/a |    53ms |        1.16x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   15ms |        n/a |    n/a |     3ms |        5.36x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanmax`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.89x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   66ms |        n/a |    n/a |    57ms |        1.14x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   17ms |        n/a |    n/a |     3ms |        5.22x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanmean`           |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.81x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   67ms |        n/a |    n/a |    57ms |        1.19x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   19ms |        n/a |    n/a |     3ms |        5.64x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanmin`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.84x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   66ms |        n/a |    n/a |    58ms |        1.13x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   17ms |        n/a |    n/a |     3ms |        5.26x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanprod`           |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.86x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   63ms |        n/a |    n/a |    55ms |        1.15x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   16ms |        n/a |    n/a |     3ms |        4.95x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanstd`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.73x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   70ms |        n/a |    n/a |    59ms |        1.18x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   20ms |        n/a |    n/a |     4ms |        5.03x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nansum`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.89x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   67ms |        n/a |    n/a |    56ms |        1.21x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   19ms |        n/a |    n/a |     3ms |        5.95x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanvar`            |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.71x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   69ms |        n/a |    n/a |    58ms |        1.19x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   20ms |        n/a |    n/a |     4ms |        5.65x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nansum_of_squares` |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        2.36x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   75ms |        n/a |    n/a |    55ms |        1.35x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   26ms |        n/a |    n/a |     3ms |        8.11x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     3ms |          n/a |              n/a |         n/a |         1.00x |
| `move_corr`               |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |       10.85x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  909ms |        n/a |    n/a |    48ms |       19.04x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  869ms |        n/a |    n/a |     9ms |       92.48x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     9ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    79ms |          n/a |              n/a |         n/a |         1.00x |
| `move_cov`                |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |       10.05x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  623ms |        n/a |    n/a |    43ms |       14.58x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  603ms |        n/a |    n/a |     8ms |       71.61x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     8ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    72ms |          n/a |              n/a |         n/a |         1.00x |
| `move_mean`               |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        1.84x |            0.03x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  120ms |       27ms |    n/a |    31ms |        3.82x |            0.87x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  113ms |       27ms |    n/a |     7ms |       16.61x |            4.01x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       27ms |    n/a |     7ms |          n/a |            3.96x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      296ms |    n/a |    58ms |          n/a |            5.08x |         n/a |         1.00x |
| `move_std`                |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        2.21x |            0.08x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  178ms |       39ms |    n/a |    30ms |        5.96x |            1.29x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  157ms |       39ms |    n/a |     6ms |       24.52x |            6.04x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       39ms |    n/a |     7ms |          n/a |            5.88x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      411ms |    n/a |    58ms |          n/a |            7.13x |         n/a |         1.00x |
| `move_sum`                |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        1.81x |            0.02x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  121ms |       26ms |    n/a |    32ms |        3.80x |            0.83x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  113ms |       26ms |    n/a |     7ms |       15.95x |            3.70x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       26ms |    n/a |     7ms |          n/a |            3.59x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      281ms |    n/a |    59ms |          n/a |            4.77x |         n/a |         1.00x |
| `move_var`                |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |        2.04x |            0.08x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  168ms |       37ms |    n/a |    29ms |        5.78x |            1.27x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  161ms |       37ms |    n/a |     6ms |       25.41x |            5.85x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       37ms |    n/a |     6ms |          n/a |            5.85x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      398ms |    n/a |    56ms |          n/a |            7.07x |         n/a |         1.00x |
| `move_exp_nancorr`        |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        7.27x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  464ms |        n/a |    n/a |    69ms |        6.73x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  471ms |        n/a |    n/a |    13ms |       35.30x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |    13ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |   111ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nancount`       |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        2.04x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   77ms |        n/a |    n/a |    33ms |        2.35x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   69ms |        n/a |    n/a |     7ms |       10.56x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     6ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    59ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nancov`         |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        7.07x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |  298ms |        n/a |    n/a |    52ms |        5.77x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |  333ms |        n/a |    n/a |    10ms |       31.75x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |    10ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    87ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nanmean`        |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.40x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   67ms |        n/a |    n/a |    33ms |        2.03x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   74ms |        n/a |    n/a |     7ms |       11.07x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     7ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    60ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nanstd`         |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        2.33x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   88ms |        n/a |    n/a |    46ms |        1.89x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   95ms |        n/a |    n/a |     9ms |       10.07x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     9ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    78ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nansum`         |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.36x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   62ms |        n/a |    n/a |    33ms |        1.88x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   71ms |        n/a |    n/a |     7ms |        9.70x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     6ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    60ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nanvar`         |                (1000,) |      1000 |    0ms |        n/a |    n/a |     0ms |        1.40x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   77ms |        n/a |    n/a |    42ms |        1.82x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   84ms |        n/a |    n/a |     9ms |        9.71x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    n/a |     9ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |    n/a |    73ms |          n/a |              n/a |         n/a |         1.00x |
| `nanargmax`[^5]           |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |       13.07x |            0.21x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   31ms |       12ms |    n/a |    12ms |        2.45x |            1.00x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   28ms |       13ms |    n/a |    13ms |        2.16x |            1.00x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       13ms |    n/a |    13ms |          n/a |            1.05x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      133ms |    n/a |   127ms |          n/a |            1.05x |         n/a |         1.00x |
| `nanargmin`[^5]           |                (1000,) |      1000 |    0ms |        0ms |    n/a |     0ms |       12.72x |            0.21x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |   27ms |       13ms |    n/a |    12ms |        2.19x |            1.01x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |   26ms |       13ms |    n/a |    12ms |        2.05x |            1.02x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       13ms |    n/a |    13ms |          n/a |            1.05x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      135ms |    n/a |   129ms |          n/a |            1.05x |         n/a |         1.00x |
| `nancount`                |                (1000,) |      1000 |    0ms |        n/a |    0ms |     0ms |        2.24x |              n/a |       0.05x |         1.00x |
|                           |            (10000000,) |  10000000 |    5ms |        n/a |    4ms |     3ms |        1.40x |              n/a |       1.06x |         1.00x |
|                           |          (100, 100000) |  10000000 |    9ms |        n/a |    3ms |     1ms |       11.00x |              n/a |       4.16x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |    4ms |     1ms |          n/a |              n/a |       3.58x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a |   45ms |     7ms |          n/a |              n/a |       6.74x |         1.00x |
| `nanmax`[^5]              |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        8.21x |            0.21x |       0.38x |         1.00x |
|                           |            (10000000,) |  10000000 |   41ms |       12ms |    1ms |    13ms |        3.26x |            1.00x |       0.11x |         1.00x |
|                           |          (100, 100000) |  10000000 |   45ms |       41ms |    1ms |    13ms |        3.62x |            3.24x |       0.11x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       40ms |    1ms |    12ms |          n/a |            3.31x |       0.12x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      402ms |   15ms |   121ms |          n/a |            3.31x |       0.12x |         1.00x |
| `nanmean`                 |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        1.32x |            0.02x |       0.20x |         1.00x |
|                           |            (10000000,) |  10000000 |   23ms |        9ms |   27ms |    10ms |        2.42x |            0.98x |       2.83x |         1.00x |
|                           |          (100, 100000) |  10000000 |   28ms |        9ms |   27ms |     2ms |       13.58x |            4.54x |      13.13x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        9ms |   27ms |     2ms |          n/a |            4.56x |      13.69x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |       91ms |  310ms |    17ms |          n/a |            5.39x |      18.39x |         1.00x |
| `nanmin`[^5]              |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        8.09x |            0.21x |       0.38x |         1.00x |
|                           |            (10000000,) |  10000000 |   41ms |       12ms |    1ms |    13ms |        3.27x |            1.00x |       0.11x |         1.00x |
|                           |          (100, 100000) |  10000000 |   45ms |       41ms |    1ms |    13ms |        3.62x |            3.24x |       0.11x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       40ms |    1ms |    12ms |          n/a |            3.28x |       0.12x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      401ms |   15ms |   122ms |          n/a |            3.30x |       0.12x |         1.00x |
| `nanquantile`             |                (1000,) |      1000 |    0ms |        n/a |    0ms |     0ms |        1.46x |              n/a |       0.57x |         1.00x |
|                           |            (10000000,) |  10000000 |  186ms |        n/a |  155ms |   198ms |        0.94x |              n/a |       0.78x |         1.00x |
|                           |          (100, 100000) |  10000000 |  197ms |        n/a |  181ms |    36ms |        5.45x |              n/a |       5.01x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        n/a |  425ms |    34ms |          n/a |              n/a |      12.50x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |        n/a | 4254ms |   331ms |          n/a |              n/a |      12.85x |         1.00x |
| `nanstd`                  |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        1.06x |            0.06x |       0.46x |         1.00x |
|                           |            (10000000,) |  10000000 |   29ms |       29ms |   53ms |    19ms |        1.50x |            1.51x |       2.75x |         1.00x |
|                           |          (100, 100000) |  10000000 |   33ms |       29ms |   53ms |     4ms |        8.29x |            7.35x |      13.27x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       28ms |   55ms |     4ms |          n/a |            7.25x |      14.43x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      294ms |  600ms |    37ms |          n/a |            8.02x |      16.35x |         1.00x |
| `nansum`                  |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        1.28x |            0.02x |       0.08x |         1.00x |
|                           |            (10000000,) |  10000000 |   22ms |        9ms |   24ms |    10ms |        2.28x |            0.97x |       2.52x |         1.00x |
|                           |          (100, 100000) |  10000000 |   27ms |        9ms |   24ms |     2ms |       17.71x |            6.24x |      16.05x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |        9ms |   25ms |     1ms |          n/a |            6.05x |      16.66x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |       90ms |  282ms |    13ms |          n/a |            6.71x |      21.07x |         1.00x |
| `nanvar`                  |                (1000,) |      1000 |    0ms |        0ms |    0ms |     0ms |        1.08x |            0.06x |       0.45x |         1.00x |
|                           |            (10000000,) |  10000000 |   28ms |       28ms |   53ms |    19ms |        1.50x |            1.49x |       2.81x |         1.00x |
|                           |          (100, 100000) |  10000000 |   33ms |       28ms |   54ms |     4ms |        8.18x |            6.97x |      13.32x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    n/a |       28ms |   56ms |     4ms |          n/a |            7.13x |      14.28x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    n/a |      281ms |  601ms |    32ms |          n/a |            8.71x |      18.65x |         1.00x |

</details>

[^1][^2][^3][^4]

[^1]:
    Benchmarks were run on a Mac M1 laptop in December 2023 on numbagg's HEAD,
    pandas 2.1.1, bottleneck 1.3.7, numpy 1.25.2, with `python
numbagg/test/run_benchmarks.py -- --benchmark-max-time=10`. They run in CI,
    though GHA's low CPU count means we don't see the full benefits of
    parallelization.

[^2]:
    While we separate the setup and the running of the functions, pandas still
    needs to do some work to create its result dataframe, and numbagg does some
    checks in python which bottleneck does in C or doesn't do. So use benchmarks
    on larger arrays for our summary so we can focus on the computational speed,
    which doesn't asymptote away. Any contributions to improve the benchmarks are
    welcome.

[^3]:
    In some instances, a library won't have the exact function — for example,
    pandas doesn't have an equivalent `move_exp_nancount` function, so we use
    its `sum` function on an array of `1`s. Similarly for
    `group_nansum_of_squares`, we use two separate operations.

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
