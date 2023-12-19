# Numbagg: Fast N-dimensional aggregation functions with Numba

[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/numbagg/numbagg/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/numbagg/numbagg/actions/workflows/test.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/numbagg?style=for-the-badge)](https://pypi.python.org/pypi/numbagg/)

Fast, flexible N-dimensional array functions written with
[Numba](https://github.com/numba/numba) and NumPy's [generalized
ufuncs](http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html).

## Why use numbagg?

### Performance

- Outperforms pandas
  - On a single core, 2-20x faster for moving window functions and 1-2x for
    aggregation and grouping functions
  - When parallelizing with multiple cores, 4-30x faster
- Outperforms bottleneck on multiple cores
  - On a single core, matches bottleneck
  - When parallelizing with multiple cores, 4-7x faster
- ...though numbagg's functions are JIT compiled, so the first run is much slower

### Versatility

- More functions (though bottleneck has some functions we don't have, and pandas' functions
  have many more parameters)
- Functions work for >3 dimensions. All functions take an arbitrary axis or
  tuple of axes to calculate over
- Written in numba — way less code, simple to inspect, simple to improve

## Functions & benchmarks

### Summary benchmark

Two benchmarks summarize numbagg's performance — one with a 1D array with no
parallelization, and one with a 2D array with the potential for parallelization.
Numbagg's relative performance is much higher where parallelization is possible.

The values in the table are numbagg's performance as a multiple of other libraries for a
given shaped array, calculated over the final axis. (so 1.00x means numbagg is equal,
higher means numbagg is faster.)

| func                      | pandas<br>`(10000000,)` | bottleneck<br>`(10000000,)` | pandas<br>`(100, 100000)` | bottleneck<br>`(100, 100000)` |
| :------------------------ | ----------------------: | --------------------------: | ------------------------: | ----------------------------: |
| `bfill`                   |                   1.16x |                       1.17x |                    10.92x |                         3.95x |
| `ffill`                   |                   1.19x |                       1.19x |                    10.93x |                         3.89x |
| `group_nanall`            |                   1.41x |                         n/a |                     9.65x |                           n/a |
| `group_nanany`            |                   1.21x |                         n/a |                     4.26x |                           n/a |
| `group_nanargmax`         |                   3.12x |                         n/a |                     8.92x |                           n/a |
| `group_nanargmin`         |                   2.92x |                         n/a |                     8.38x |                           n/a |
| `group_nancount`          |                   1.06x |                         n/a |                     4.58x |                           n/a |
| `group_nanfirst`          |                   1.39x |                         n/a |                     8.98x |                           n/a |
| `group_nanlast`           |                   1.13x |                         n/a |                     3.64x |                           n/a |
| `group_nanmax`            |                   1.16x |                         n/a |                     4.69x |                           n/a |
| `group_nanmean`           |                   1.24x |                         n/a |                     4.46x |                           n/a |
| `group_nanmin`            |                   1.16x |                         n/a |                     3.72x |                           n/a |
| `group_nanprod`           |                   1.14x |                         n/a |                     3.77x |                           n/a |
| `group_nanstd`            |                   1.46x |                         n/a |                     3.10x |                           n/a |
| `group_nansum_of_squares` |                   1.31x |                         n/a |                     5.55x |                           n/a |
| `group_nansum`            |                   1.19x |                         n/a |                     4.43x |                           n/a |
| `group_nanvar`            |                   1.40x |                         n/a |                     4.40x |                           n/a |
| `move_corr`               |                  19.33x |                         n/a |                    74.36x |                           n/a |
| `move_cov`                |                  14.62x |                         n/a |                    56.92x |                           n/a |
| `move_exp_nancorr`        |                   5.74x |                         n/a |                    30.59x |                           n/a |
| `move_exp_nancount`       |                   2.70x |                         n/a |                     9.05x |                           n/a |
| `move_exp_nancov`         |                   5.92x |                         n/a |                    28.43x |                           n/a |
| `move_exp_nanmean`        |                   2.12x |                         n/a |                     9.27x |                           n/a |
| `move_exp_nanstd`         |                   1.91x |                         n/a |                     8.32x |                           n/a |
| `move_exp_nansum`         |                   2.01x |                         n/a |                     8.46x |                           n/a |
| `move_exp_nanvar`         |                   1.99x |                         n/a |                     9.01x |                           n/a |
| `move_mean`               |                   4.10x |                       0.87x |                    15.80x |                         4.04x |
| `move_std`                |                   6.08x |                       1.06x |                    24.27x |                         4.93x |
| `move_sum`                |                   3.62x |                       0.84x |                    14.81x |                         3.79x |
| `move_var`                |                   6.02x |                       1.17x |                    24.40x |                         5.14x |
| `nanargmax`               |                   2.32x |                       0.97x |                     2.11x |                         0.99x |
| `nanargmin`               |                   2.46x |                       0.99x |                     2.40x |                         0.98x |
| `nancount`                |                   1.44x |                         n/a |                     9.87x |                           n/a |
| `nanmax`                  |                   1.04x |                       1.03x |                     1.37x |                         1.01x |
| `nanmean`                 |                   2.49x |                       0.95x |                    11.44x |                         3.83x |
| `nanmin`                  |                   0.93x |                       0.92x |                     1.39x |                         0.99x |
| `nanquantile`             |                   0.88x |                         n/a |                     0.92x |                           n/a |
| `nanstd`                  |                   1.52x |                       1.58x |                     8.80x |                         7.27x |
| `nansum`                  |                   2.33x |                       1.00x |                    10.94x |                         3.48x |
| `nanvar`                  |                   1.43x |                       1.52x |                     7.36x |                         6.58x |

### Full benchmarks

<details>

| func                      |                  shape |      size | numbagg | pandas | bottleneck |  numpy | numbagg_ratio | pandas_ratio | bottleneck_ratio | numpy_ratio |
| :------------------------ | ---------------------: | --------: | ------: | -----: | ---------: | -----: | ------------: | -----------: | ---------------: | ----------: |
| `bfill`                   |                (1000,) |      1000 |     0ms |    0ms |        0ms |    n/a |         1.00x |        0.74x |            0.01x |         n/a |
|                           |            (10000000,) |  10000000 |    18ms |   21ms |       21ms |    n/a |         1.00x |        1.16x |            1.17x |         n/a |
|                           |          (100, 100000) |  10000000 |     6ms |   62ms |       22ms |    n/a |         1.00x |       10.92x |            3.95x |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       22ms |    n/a |         1.00x |          n/a |            4.35x |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |    67ms |    n/a |      288ms |    n/a |         1.00x |          n/a |            4.28x |         n/a |
| `ffill`                   |                (1000,) |      1000 |     0ms |    0ms |        0ms |    n/a |         1.00x |        0.67x |            0.01x |         n/a |
|                           |            (10000000,) |  10000000 |    18ms |   21ms |       21ms |    n/a |         1.00x |        1.19x |            1.19x |         n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   59ms |       21ms |    n/a |         1.00x |       10.93x |            3.89x |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |       19ms |    n/a |         1.00x |          n/a |            4.26x |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |    66ms |    n/a |      248ms |    n/a |         1.00x |          n/a |            3.74x |         n/a |
| `group_nanall`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.84x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    51ms |   72ms |        n/a |    n/a |         1.00x |        1.41x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     2ms |   19ms |        n/a |    n/a |         1.00x |        9.65x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     1ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanany`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.82x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    61ms |   73ms |        n/a |    n/a |         1.00x |        1.21x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   19ms |        n/a |    n/a |         1.00x |        4.26x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanargmax`         |                (1000,) |      1000 |     0ms |    1ms |        n/a |    n/a |         1.00x |        7.59x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    64ms |  199ms |        n/a |    n/a |         1.00x |        3.12x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     6ms |   50ms |        n/a |    n/a |         1.00x |        8.92x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanargmin`         |                (1000,) |      1000 |     0ms |    1ms |        n/a |    n/a |         1.00x |        8.12x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    64ms |  188ms |        n/a |    n/a |         1.00x |        2.92x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   45ms |        n/a |    n/a |         1.00x |        8.38x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nancount`          |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.84x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    61ms |   65ms |        n/a |    n/a |         1.00x |        1.06x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   18ms |        n/a |    n/a |         1.00x |        4.58x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanfirst`          |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.86x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    54ms |   75ms |        n/a |    n/a |         1.00x |        1.39x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     2ms |   17ms |        n/a |    n/a |         1.00x |        8.98x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     2ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanlast`           |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        1.01x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    61ms |   69ms |        n/a |    n/a |         1.00x |        1.13x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   17ms |        n/a |    n/a |         1.00x |        3.64x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanmax`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        1.07x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    65ms |   75ms |        n/a |    n/a |         1.00x |        1.16x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     4ms |   19ms |        n/a |    n/a |         1.00x |        4.69x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanmean`           |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.84x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    61ms |   76ms |        n/a |    n/a |         1.00x |        1.24x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   21ms |        n/a |    n/a |         1.00x |        4.46x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanmin`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.84x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    65ms |   75ms |        n/a |    n/a |         1.00x |        1.16x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   17ms |        n/a |    n/a |         1.00x |        3.72x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     6ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanprod`           |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        1.02x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    62ms |   71ms |        n/a |    n/a |         1.00x |        1.14x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   18ms |        n/a |    n/a |         1.00x |        3.77x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanstd`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        1.01x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    67ms |   98ms |        n/a |    n/a |         1.00x |        1.46x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     7ms |   23ms |        n/a |    n/a |         1.00x |        3.10x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     7ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nansum`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.86x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    62ms |   74ms |        n/a |    n/a |         1.00x |        1.19x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   21ms |        n/a |    n/a |         1.00x |        4.43x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nanvar`            |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        1.03x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    66ms |   92ms |        n/a |    n/a |         1.00x |        1.40x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   21ms |        n/a |    n/a |         1.00x |        4.40x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `group_nansum_of_squares` |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        1.12x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    63ms |   83ms |        n/a |    n/a |         1.00x |        1.31x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     5ms |   30ms |        n/a |    n/a |         1.00x |        5.55x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     4ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `move_corr`               |                (1000,) |      1000 |     0ms |    1ms |        n/a |    n/a |         1.00x |        4.93x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    52ms | 1004ms |        n/a |    n/a |         1.00x |       19.33x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |    13ms |  976ms |        n/a |    n/a |         1.00x |       74.36x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    11ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |   143ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `move_cov`                |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        4.49x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    48ms |  698ms |        n/a |    n/a |         1.00x |       14.62x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |    11ms |  638ms |        n/a |    n/a |         1.00x |       56.92x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    12ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |   156ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `move_mean`               |                (1000,) |      1000 |     0ms |    0ms |        0ms |    n/a |         1.00x |        0.85x |            0.01x |         n/a |
|                           |            (10000000,) |  10000000 |    32ms |  131ms |       28ms |    n/a |         1.00x |        4.10x |            0.87x |         n/a |
|                           |          (100, 100000) |  10000000 |     7ms |  112ms |       29ms |    n/a |         1.00x |       15.80x |            4.04x |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    11ms |    n/a |       27ms |    n/a |         1.00x |          n/a |            2.54x |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |    70ms |    n/a |      312ms |    n/a |         1.00x |          n/a |            4.44x |         n/a |
| `move_std`                |                (1000,) |      1000 |     0ms |    0ms |        0ms |    n/a |         1.00x |        1.01x |            0.03x |         n/a |
|                           |            (10000000,) |  10000000 |    32ms |  195ms |       34ms |    n/a |         1.00x |        6.08x |            1.06x |         n/a |
|                           |          (100, 100000) |  10000000 |     8ms |  183ms |       37ms |    n/a |         1.00x |       24.27x |            4.93x |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    11ms |    n/a |       36ms |    n/a |         1.00x |          n/a |            3.35x |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |    97ms |    n/a |      400ms |    n/a |         1.00x |          n/a |            4.13x |         n/a |
| `move_sum`                |                (1000,) |      1000 |     0ms |    0ms |        0ms |    n/a |         1.00x |        0.92x |            0.01x |         n/a |
|                           |            (10000000,) |  10000000 |    34ms |  122ms |       28ms |    n/a |         1.00x |        3.62x |            0.84x |         n/a |
|                           |          (100, 100000) |  10000000 |     7ms |  110ms |       28ms |    n/a |         1.00x |       14.81x |            3.79x |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     8ms |    n/a |       27ms |    n/a |         1.00x |          n/a |            3.29x |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |    68ms |    n/a |      319ms |    n/a |         1.00x |          n/a |            4.73x |         n/a |
| `move_var`                |                (1000,) |      1000 |     0ms |    0ms |        0ms |    n/a |         1.00x |        1.39x |            0.04x |         n/a |
|                           |            (10000000,) |  10000000 |    31ms |  187ms |       36ms |    n/a |         1.00x |        6.02x |            1.17x |         n/a |
|                           |          (100, 100000) |  10000000 |     7ms |  177ms |       37ms |    n/a |         1.00x |       24.40x |            5.14x |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     8ms |    n/a |       34ms |    n/a |         1.00x |          n/a |            4.45x |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |    92ms |    n/a |      393ms |    n/a |         1.00x |          n/a |            4.28x |         n/a |
| `move_exp_nancorr`        |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        3.77x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    86ms |  492ms |        n/a |    n/a |         1.00x |        5.74x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |    16ms |  499ms |        n/a |    n/a |         1.00x |       30.59x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    16ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |   224ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `move_exp_nancount`       |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.94x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    34ms |   93ms |        n/a |    n/a |         1.00x |        2.70x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     8ms |   76ms |        n/a |    n/a |         1.00x |        9.05x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     9ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |   125ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `move_exp_nancov`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        3.75x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    54ms |  317ms |        n/a |    n/a |         1.00x |        5.92x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |    12ms |  349ms |        n/a |    n/a |         1.00x |       28.43x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    12ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |   210ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `move_exp_nanmean`        |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.65x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    35ms |   74ms |        n/a |    n/a |         1.00x |        2.12x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     9ms |   80ms |        n/a |    n/a |         1.00x |        9.27x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     7ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |    78ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `move_exp_nanstd`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        1.09x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    50ms |   97ms |        n/a |    n/a |         1.00x |        1.91x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |    12ms |  101ms |        n/a |    n/a |         1.00x |        8.32x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    19ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |   142ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `move_exp_nansum`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.92x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    34ms |   69ms |        n/a |    n/a |         1.00x |        2.01x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |     9ms |   75ms |        n/a |    n/a |         1.00x |        8.46x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     9ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |   111ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `move_exp_nanvar`         |                (1000,) |      1000 |     0ms |    0ms |        n/a |    n/a |         1.00x |        0.98x |              n/a |         n/a |
|                           |            (10000000,) |  10000000 |    45ms |   89ms |        n/a |    n/a |         1.00x |        1.99x |              n/a |         n/a |
|                           |          (100, 100000) |  10000000 |    10ms |   92ms |        n/a |    n/a |         1.00x |        9.01x |              n/a |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    12ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |   114ms |    n/a |        n/a |    n/a |         1.00x |          n/a |              n/a |         n/a |
| `nanargmax`               |                (1000,) |      1000 |     0ms |    0ms |        0ms |    n/a |         1.00x |       13.36x |            0.21x |         n/a |
|                           |            (10000000,) |  10000000 |    13ms |   31ms |       13ms |    n/a |         1.00x |        2.32x |            0.97x |         n/a |
|                           |          (100, 100000) |  10000000 |    13ms |   28ms |       13ms |    n/a |         1.00x |        2.11x |            0.99x |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    14ms |    n/a |       15ms |    n/a |         1.00x |          n/a |            1.07x |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |   139ms |    n/a |      153ms |    n/a |         1.00x |          n/a |            1.10x |         n/a |
| `nanargmin`               |                (1000,) |      1000 |     0ms |    0ms |        0ms |    n/a |         1.00x |       14.64x |            0.21x |         n/a |
|                           |            (10000000,) |  10000000 |    14ms |   33ms |       13ms |    n/a |         1.00x |        2.46x |            0.99x |         n/a |
|                           |          (100, 100000) |  10000000 |    13ms |   32ms |       13ms |    n/a |         1.00x |        2.40x |            0.98x |         n/a |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    13ms |    n/a |       14ms |    n/a |         1.00x |          n/a |            1.07x |         n/a |
|                           |      (100, 1000, 1000) | 100000000 |   140ms |    n/a |      148ms |    n/a |         1.00x |          n/a |            1.06x |         n/a |
| `nancount`                |                (1000,) |      1000 |     0ms |    0ms |        n/a |    0ms |         1.00x |        0.97x |              n/a |       0.02x |
|                           |            (10000000,) |  10000000 |     4ms |    5ms |        n/a |    4ms |         1.00x |        1.44x |              n/a |       0.99x |
|                           |          (100, 100000) |  10000000 |     1ms |   11ms |        n/a |    4ms |         1.00x |        9.87x |              n/a |       3.37x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     1ms |    n/a |        n/a |    4ms |         1.00x |          n/a |              n/a |       2.99x |
|                           |      (100, 1000, 1000) | 100000000 |    11ms |    n/a |        n/a |   48ms |         1.00x |          n/a |              n/a |       4.44x |
| `nanmax`                  |                (1000,) |      1000 |     0ms |    0ms |        0ms |    0ms |         1.00x |        7.67x |            0.22x |       0.36x |
|                           |            (10000000,) |  10000000 |    13ms |   13ms |       13ms |    1ms |         1.00x |        1.04x |            1.03x |       0.11x |
|                           |          (100, 100000) |  10000000 |    13ms |   18ms |       13ms |    2ms |         1.00x |        1.37x |            1.01x |       0.12x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    13ms |    n/a |       12ms |    2ms |         1.00x |          n/a |            0.97x |       0.14x |
|                           |      (100, 1000, 1000) | 100000000 |   140ms |    n/a |      134ms |   18ms |         1.00x |          n/a |            0.96x |       0.13x |
| `nanmean`                 |                (1000,) |      1000 |     0ms |    0ms |        0ms |    0ms |         1.00x |        0.56x |            0.01x |       0.08x |
|                           |            (10000000,) |  10000000 |    11ms |   26ms |       10ms |   28ms |         1.00x |        2.49x |            0.95x |       2.67x |
|                           |          (100, 100000) |  10000000 |     3ms |   32ms |       11ms |   29ms |         1.00x |       11.44x |            3.83x |      10.39x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     2ms |    n/a |       10ms |   30ms |         1.00x |          n/a |            4.99x |      14.27x |
|                           |      (100, 1000, 1000) | 100000000 |    21ms |    n/a |      101ms |  328ms |         1.00x |          n/a |            4.75x |      15.39x |
| `nanmin`                  |                (1000,) |      1000 |     0ms |    0ms |        0ms |    0ms |         1.00x |        8.43x |            0.21x |       0.36x |
|                           |            (10000000,) |  10000000 |    14ms |   13ms |       13ms |    2ms |         1.00x |        0.93x |            0.92x |       0.12x |
|                           |          (100, 100000) |  10000000 |    13ms |   19ms |       13ms |    2ms |         1.00x |        1.39x |            0.99x |       0.13x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    13ms |    n/a |       14ms |    2ms |         1.00x |          n/a |            1.13x |       0.13x |
|                           |      (100, 1000, 1000) | 100000000 |   135ms |    n/a |      133ms |   16ms |         1.00x |          n/a |            0.98x |       0.12x |
| `nanquantile`             |                (1000,) |      1000 |     0ms |    0ms |        n/a |    0ms |         1.00x |        1.06x |              n/a |       0.25x |
|                           |            (10000000,) |  10000000 |   228ms |  200ms |        n/a |  166ms |         1.00x |        0.88x |              n/a |       0.73x |
|                           |          (100, 100000) |  10000000 |   227ms |  209ms |        n/a |  175ms |         1.00x |        0.92x |              n/a |       0.77x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |   237ms |    n/a |        n/a |  170ms |         1.00x |          n/a |              n/a |       0.72x |
|                           |      (100, 1000, 1000) | 100000000 |  2324ms |    n/a |        n/a | 1928ms |         1.00x |          n/a |              n/a |       0.83x |
| `nanstd`                  |                (1000,) |      1000 |     0ms |    0ms |        0ms |    0ms |         1.00x |        0.64x |            0.03x |       0.27x |
|                           |            (10000000,) |  10000000 |    21ms |   31ms |       33ms |   56ms |         1.00x |        1.52x |            1.58x |       2.71x |
|                           |          (100, 100000) |  10000000 |     4ms |   38ms |       31ms |   57ms |         1.00x |        8.80x |            7.27x |      13.31x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       30ms |   58ms |         1.00x |          n/a |            6.32x |      12.33x |
|                           |      (100, 1000, 1000) | 100000000 |    42ms |    n/a |      310ms |  640ms |         1.00x |          n/a |            7.35x |      15.15x |
| `nansum`                  |                (1000,) |      1000 |     0ms |    0ms |        0ms |    0ms |         1.00x |        0.90x |            0.01x |       0.05x |
|                           |            (10000000,) |  10000000 |    10ms |   23ms |       10ms |   31ms |         1.00x |        2.33x |            1.00x |       3.11x |
|                           |          (100, 100000) |  10000000 |     3ms |   31ms |       10ms |   28ms |         1.00x |       10.94x |            3.48x |       9.79x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     2ms |    n/a |        9ms |   27ms |         1.00x |          n/a |            3.83x |      11.19x |
|                           |      (100, 1000, 1000) | 100000000 |    26ms |    n/a |      107ms |  298ms |         1.00x |          n/a |            4.05x |      11.33x |
| `nanvar`                  |                (1000,) |      1000 |     0ms |    0ms |        0ms |    0ms |         1.00x |        0.73x |            0.04x |       0.28x |
|                           |            (10000000,) |  10000000 |    21ms |   30ms |       32ms |   57ms |         1.00x |        1.43x |            1.52x |       2.68x |
|                           |          (100, 100000) |  10000000 |     5ms |   35ms |       31ms |   59ms |         1.00x |        7.36x |            6.58x |      12.33x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |     5ms |    n/a |       31ms |   63ms |         1.00x |          n/a |            5.80x |      11.70x |
|                           |      (100, 1000, 1000) | 100000000 |    43ms |    n/a |      303ms |  623ms |         1.00x |          n/a |            7.00x |      14.39x |

</details>

[^1][^2][^3][^4][^5]

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
