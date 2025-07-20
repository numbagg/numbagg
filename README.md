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

Two benchmarks summarize numbagg's performance — the first with a 1D array of 10M elements without
parallelization, and a second with a 2D array of 100x10K elements with parallelization. Numbagg's relative
performance is much higher where parallelization is possible. A wider range of arrays is
listed in the full set of benchmarks below.

The values in the table are numbagg's performance as a multiple of other libraries for a
given shaped array calculated over the final axis. (so 1.00x means numbagg is equal,
higher means numbagg is faster.)

| func                      | 1D<br>pandas | 1D<br>bottleneck | 1D<br>numpy | 2D<br>pandas | 2D<br>bottleneck | 2D<br>numpy |
| :------------------------ | -----------: | ---------------: | ----------: | -----------: | ---------------: | ----------: |
| `bfill`                   |        1.06x |            1.13x |         n/a |       11.11x |            5.04x |         n/a |
| `ffill`                   |        1.12x |            0.99x |         n/a |       11.50x |            4.25x |         n/a |
| `group_nanall`            |        1.38x |              n/a |         n/a |        7.77x |              n/a |         n/a |
| `group_nanany`            |        1.12x |              n/a |         n/a |        6.21x |              n/a |         n/a |
| `group_nanargmax`         |        1.16x |              n/a |         n/a |        6.81x |              n/a |         n/a |
| `group_nanargmin`         |        1.17x |              n/a |         n/a |        6.48x |              n/a |         n/a |
| `group_nancount`          |        1.05x |              n/a |         n/a |        4.94x |              n/a |         n/a |
| `group_nanfirst`          |        1.52x |              n/a |         n/a |       11.13x |              n/a |         n/a |
| `group_nanlast`           |        1.12x |              n/a |         n/a |        5.56x |              n/a |         n/a |
| `group_nanmax`            |        1.13x |              n/a |         n/a |        5.13x |              n/a |         n/a |
| `group_nanmean`           |        1.14x |              n/a |         n/a |        5.61x |              n/a |         n/a |
| `group_nanmin`            |        1.12x |              n/a |         n/a |        5.75x |              n/a |         n/a |
| `group_nanprod`           |        1.15x |              n/a |         n/a |        5.25x |              n/a |         n/a |
| `group_nanstd`            |        1.14x |              n/a |         n/a |        5.41x |              n/a |         n/a |
| `group_nansum_of_squares` |        1.33x |              n/a |         n/a |        8.00x |              n/a |         n/a |
| `group_nansum`            |        1.18x |              n/a |         n/a |        5.63x |              n/a |         n/a |
| `group_nanvar`            |        1.13x |              n/a |         n/a |        4.88x |              n/a |         n/a |
| `move_corr`               |       16.42x |              n/a |         n/a |      115.76x |              n/a |         n/a |
| `move_cov`                |       12.30x |              n/a |         n/a |       86.56x |              n/a |         n/a |
| `move_exp_nancorr`        |        6.65x |              n/a |         n/a |       46.98x |              n/a |         n/a |
| `move_exp_nancount`       |        1.88x |              n/a |         n/a |        9.95x |              n/a |         n/a |
| `move_exp_nancov`         |        6.53x |              n/a |         n/a |       43.63x |              n/a |         n/a |
| `move_exp_nanmean`        |        1.61x |              n/a |         n/a |       10.65x |              n/a |         n/a |
| `move_exp_nanstd`         |        1.76x |              n/a |         n/a |       12.40x |              n/a |         n/a |
| `move_exp_nansum`         |        1.09x |              n/a |         n/a |        9.01x |              n/a |         n/a |
| `move_exp_nanvar`         |        1.77x |              n/a |         n/a |       11.41x |              n/a |         n/a |
| `move_mean`               |        6.03x |            1.34x |         n/a |       26.60x |            6.25x |         n/a |
| `move_std`                |        4.76x |            0.89x |         n/a |       29.09x |            6.24x |         n/a |
| `move_sum`                |        5.16x |            1.13x |         n/a |       24.02x |            6.10x |         n/a |
| `move_var`                |        5.45x |            1.05x |         n/a |       29.54x |            6.05x |         n/a |
| `nanargmax`[^5]           |        2.40x |            0.53x |         n/a |        2.32x |            0.93x |         n/a |
| `nanargmin`[^5]           |        2.35x |            0.50x |         n/a |        2.53x |            1.00x |         n/a |
| `nancount`                |        2.01x |              n/a |       1.59x |       12.26x |              n/a |       3.96x |
| `nanmax`[^5]              |        3.15x |            0.50x |       0.09x |        3.59x |            3.24x |       0.09x |
| `nanmean`                 |        3.00x |            1.01x |       3.82x |       18.98x |            5.04x |      19.33x |
| `nanmin`[^5]              |        3.07x |            0.50x |       0.09x |        3.39x |            3.03x |       0.09x |
| `nanquantile`             |        0.69x |              n/a |       0.53x |        4.94x |              n/a |       4.33x |
| `nanstd`                  |        1.63x |            1.61x |       3.39x |       12.39x |           10.18x |      22.03x |
| `nansum`                  |        2.48x |            0.94x |       3.31x |       20.47x |            4.65x |      17.90x |
| `nanvar`                  |        1.61x |            1.65x |       3.40x |       12.62x |           10.49x |      22.13x |

### Full benchmarks

<details>

| func                      |                  shape |      size | ndim | pandas | bottleneck |  numpy | numbagg | pandas_ratio | bottleneck_ratio | numpy_ratio | numbagg_ratio |
| :------------------------ | ---------------------: | --------: | ---: | -----: | ---------: | -----: | ------: | -----------: | ---------------: | ----------: | ------------: |
| `bfill`                   |                (1000,) |      1000 |    1 |    0ms |        0ms |    n/a |     0ms |        0.38x |            0.01x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   15ms |       16ms |    n/a |    14ms |        1.06x |            1.13x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   37ms |       17ms |    n/a |     3ms |       11.11x |            5.04x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       18ms |    n/a |     3ms |          n/a |            6.13x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      199ms |    n/a |    31ms |          n/a |            6.44x |         n/a |         1.00x |
| `ffill`                   |                (1000,) |      1000 |    1 |    0ms |        0ms |    n/a |     0ms |        0.37x |            0.01x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   15ms |       14ms |    n/a |    14ms |        1.12x |            0.99x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   37ms |       14ms |    n/a |     3ms |       11.50x |            4.25x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       14ms |    n/a |     3ms |          n/a |            4.64x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      176ms |    n/a |    31ms |          n/a |            5.72x |         n/a |         1.00x |
| `group_nanall`            |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.72x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   48ms |        n/a |    n/a |    35ms |        1.38x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   18ms |        n/a |    n/a |     2ms |        7.77x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     1ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanany`            |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.70x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   49ms |        n/a |    n/a |    44ms |        1.12x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   18ms |        n/a |    n/a |     3ms |        6.21x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanargmax`         |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        1.07x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   49ms |        n/a |    n/a |    42ms |        1.16x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   17ms |        n/a |    n/a |     3ms |        6.81x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanargmin`         |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        1.06x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   49ms |        n/a |    n/a |    42ms |        1.17x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   17ms |        n/a |    n/a |     3ms |        6.48x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nancount`          |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.66x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   44ms |        n/a |    n/a |    42ms |        1.05x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   13ms |        n/a |    n/a |     3ms |        4.94x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     1ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanfirst`          |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.73x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   52ms |        n/a |    n/a |    34ms |        1.52x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   16ms |        n/a |    n/a |     1ms |       11.13x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     1ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanlast`           |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.72x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   47ms |        n/a |    n/a |    42ms |        1.12x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   14ms |        n/a |    n/a |     2ms |        5.56x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     1ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanmax`            |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.71x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   48ms |        n/a |    n/a |    43ms |        1.13x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   14ms |        n/a |    n/a |     3ms |        5.13x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanmean`           |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.72x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   50ms |        n/a |    n/a |    44ms |        1.14x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   16ms |        n/a |    n/a |     3ms |        5.61x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanmin`            |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.73x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   48ms |        n/a |    n/a |    43ms |        1.12x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   14ms |        n/a |    n/a |     2ms |        5.75x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanprod`           |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.70x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   48ms |        n/a |    n/a |    42ms |        1.15x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   14ms |        n/a |    n/a |     3ms |        5.25x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     1ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanstd`            |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.71x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   51ms |        n/a |    n/a |    45ms |        1.14x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   17ms |        n/a |    n/a |     3ms |        5.41x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nansum`            |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.74x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   51ms |        n/a |    n/a |    43ms |        1.18x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   16ms |        n/a |    n/a |     3ms |        5.63x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nanvar`            |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.70x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   51ms |        n/a |    n/a |    45ms |        1.13x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   16ms |        n/a |    n/a |     3ms |        4.88x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     2ms |          n/a |              n/a |         n/a |         1.00x |
| `group_nansum_of_squares` |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.88x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   57ms |        n/a |    n/a |    43ms |        1.33x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   22ms |        n/a |    n/a |     3ms |        8.00x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     1ms |          n/a |              n/a |         n/a |         1.00x |
| `move_corr`               |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        2.68x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |  710ms |        n/a |    n/a |    43ms |       16.42x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |  683ms |        n/a |    n/a |     6ms |      115.76x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     5ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a |    n/a |    49ms |          n/a |              n/a |         n/a |         1.00x |
| `move_cov`                |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        2.43x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |  490ms |        n/a |    n/a |    40ms |       12.30x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |  460ms |        n/a |    n/a |     5ms |       86.56x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a |    n/a |    44ms |          n/a |              n/a |         n/a |         1.00x |
| `move_mean`               |                (1000,) |      1000 |    1 |    0ms |        0ms |    n/a |     0ms |        0.46x |            0.01x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   92ms |       21ms |    n/a |    15ms |        6.03x |            1.34x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   88ms |       21ms |    n/a |     3ms |       26.60x |            6.25x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       20ms |    n/a |     3ms |          n/a |            6.66x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      228ms |    n/a |    32ms |          n/a |            7.12x |         n/a |         1.00x |
| `move_std`                |                (1000,) |      1000 |    1 |    0ms |        0ms |    n/a |     0ms |        0.53x |            0.02x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |  141ms |       26ms |    n/a |    30ms |        4.76x |            0.89x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |  123ms |       26ms |    n/a |     4ms |       29.09x |            6.24x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       26ms |    n/a |     4ms |          n/a |            7.37x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      291ms |    n/a |    37ms |          n/a |            7.82x |         n/a |         1.00x |
| `move_sum`                |                (1000,) |      1000 |    1 |    0ms |        0ms |    n/a |     0ms |        0.46x |            0.01x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   95ms |       21ms |    n/a |    18ms |        5.16x |            1.13x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   83ms |       21ms |    n/a |     3ms |       24.02x |            6.10x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       21ms |    n/a |     3ms |          n/a |            6.79x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      227ms |    n/a |    31ms |          n/a |            7.29x |         n/a |         1.00x |
| `move_var`                |                (1000,) |      1000 |    1 |    0ms |        0ms |    n/a |     0ms |        0.50x |            0.02x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |  131ms |       25ms |    n/a |    24ms |        5.45x |            1.05x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |  122ms |       25ms |    n/a |     4ms |       29.54x |            6.05x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       25ms |    n/a |     4ms |          n/a |            7.12x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      275ms |    n/a |    36ms |          n/a |            7.69x |         n/a |         1.00x |
| `move_exp_nancorr`        |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        2.33x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |  344ms |        n/a |    n/a |    52ms |        6.65x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |  338ms |        n/a |    n/a |     7ms |       46.98x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     6ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a |    n/a |    55ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nancount`       |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.57x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   51ms |        n/a |    n/a |    27ms |        1.88x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   47ms |        n/a |    n/a |     5ms |        9.95x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a |    n/a |    40ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nancov`         |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        2.19x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |  215ms |        n/a |    n/a |    33ms |        6.53x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |  234ms |        n/a |    n/a |     5ms |       43.63x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     5ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a |    n/a |    43ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nanmean`        |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.39x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   47ms |        n/a |    n/a |    30ms |        1.61x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   52ms |        n/a |    n/a |     5ms |       10.65x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a |    n/a |    43ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nanstd`         |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.68x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   64ms |        n/a |    n/a |    36ms |        1.76x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   74ms |        n/a |    n/a |     6ms |       12.40x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     5ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a |    n/a |    44ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nansum`         |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.38x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   36ms |        n/a |    n/a |    33ms |        1.09x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   43ms |        n/a |    n/a |     5ms |        9.01x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a |    n/a |    42ms |          n/a |              n/a |         n/a |         1.00x |
| `move_exp_nanvar`         |                (1000,) |      1000 |    1 |    0ms |        n/a |    n/a |     0ms |        0.40x |              n/a |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   56ms |        n/a |    n/a |    32ms |        1.77x |              n/a |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   64ms |        n/a |    n/a |     6ms |       11.41x |              n/a |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    n/a |     4ms |          n/a |              n/a |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a |    n/a |    46ms |          n/a |              n/a |         n/a |         1.00x |
| `nanargmax`[^5]           |                (1000,) |      1000 |    1 |    0ms |        0ms |    n/a |     0ms |       17.65x |            0.17x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   24ms |        5ms |    n/a |    10ms |        2.40x |            0.53x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   25ms |       10ms |    n/a |    11ms |        2.32x |            0.93x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       11ms |    n/a |    11ms |          n/a |            1.00x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      107ms |    n/a |   108ms |          n/a |            0.99x |         n/a |         1.00x |
| `nanargmin`[^5]           |                (1000,) |      1000 |    1 |    0ms |        0ms |    n/a |     0ms |       17.72x |            0.17x |         n/a |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   25ms |        5ms |    n/a |    11ms |        2.35x |            0.50x |         n/a |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   25ms |       10ms |    n/a |    10ms |        2.53x |            1.00x |         n/a |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       11ms |    n/a |    11ms |          n/a |            1.00x |         n/a |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      108ms |    n/a |   108ms |          n/a |            1.00x |         n/a |         1.00x |
| `nancount`                |                (1000,) |      1000 |    1 |    0ms |        n/a |    0ms |     0ms |        0.77x |              n/a |       0.02x |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |    3ms |        n/a |    3ms |     2ms |        2.01x |              n/a |       1.59x |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |    8ms |        n/a |    3ms |     1ms |       12.26x |              n/a |       3.96x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |    3ms |     1ms |          n/a |              n/a |       3.97x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a |   33ms |     7ms |          n/a |              n/a |       5.07x |         1.00x |
| `nanmax`[^5]              |                (1000,) |      1000 |    1 |    0ms |        0ms |    0ms |     0ms |       11.07x |            0.17x |       0.55x |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   32ms |        5ms |    1ms |    10ms |        3.15x |            0.50x |       0.09x |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   36ms |       33ms |    1ms |    10ms |        3.59x |            3.24x |       0.09x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       32ms |    1ms |    10ms |          n/a |            3.24x |       0.10x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      320ms |   11ms |    98ms |          n/a |            3.26x |       0.11x |         1.00x |
| `nanmean`                 |                (1000,) |      1000 |    1 |    0ms |        0ms |    0ms |     0ms |        0.39x |            0.00x |       0.05x |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   17ms |        6ms |   21ms |     6ms |        3.00x |            1.01x |       3.82x |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   21ms |        5ms |   21ms |     1ms |       18.98x |            5.04x |      19.33x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        5ms |   21ms |     1ms |          n/a |            6.10x |      23.77x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |       54ms |  258ms |     8ms |          n/a |            7.00x |      33.59x |         1.00x |
| `nanmin`[^5]              |                (1000,) |      1000 |    1 |    0ms |        0ms |    0ms |     0ms |       10.86x |            0.17x |       0.55x |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   33ms |        5ms |    1ms |    11ms |        3.07x |            0.50x |       0.09x |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   36ms |       32ms |    1ms |    11ms |        3.39x |            3.03x |       0.09x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       32ms |    1ms |    10ms |          n/a |            3.12x |       0.10x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      320ms |   11ms |   102ms |          n/a |            3.12x |       0.11x |         1.00x |
| `nanquantile`             |                (1000,) |      1000 |    1 |    0ms |        n/a |    0ms |     0ms |        0.56x |              n/a |       0.21x |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |  114ms |        n/a |   87ms |   164ms |        0.69x |              n/a |       0.53x |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |  131ms |        n/a |  115ms |    27ms |        4.94x |              n/a |       4.33x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        n/a |  315ms |    19ms |          n/a |              n/a |      16.51x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |        n/a | 3118ms |   165ms |          n/a |              n/a |      18.88x |         1.00x |
| `nanstd`                  |                (1000,) |      1000 |    1 |    0ms |        0ms |    0ms |     0ms |        0.31x |            0.02x |       0.14x |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   21ms |       20ms |   43ms |    13ms |        1.63x |            1.61x |       3.39x |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   24ms |       20ms |   43ms |     2ms |       12.39x |           10.18x |      22.03x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       20ms |   46ms |     1ms |          n/a |           14.17x |      32.66x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      202ms |  513ms |    13ms |          n/a |           16.08x |      40.78x |         1.00x |
| `nansum`                  |                (1000,) |      1000 |    1 |    0ms |        0ms |    0ms |     0ms |        0.46x |            0.01x |       0.03x |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   14ms |        5ms |   19ms |     6ms |        2.48x |            0.94x |       3.31x |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   22ms |        5ms |   19ms |     1ms |       20.47x |            4.65x |      17.90x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |        5ms |   20ms |     1ms |          n/a |            6.21x |      22.95x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |       53ms |  226ms |     8ms |          n/a |            6.98x |      29.90x |         1.00x |
| `nanvar`                  |                (1000,) |      1000 |    1 |    0ms |        0ms |    0ms |     0ms |        0.32x |            0.02x |       0.13x |         1.00x |
|                           |            (10000000,) |  10000000 |    1 |   21ms |       21ms |   44ms |    13ms |        1.61x |            1.65x |       3.40x |         1.00x |
|                           |          (100, 100000) |  10000000 |    2 |   25ms |       21ms |   43ms |     2ms |       12.62x |           10.49x |      22.13x |         1.00x |
|                           | (10, 10, 10, 10, 1000) |  10000000 |    5 |    n/a |       20ms |   46ms |     1ms |          n/a |           14.02x |      32.28x |         1.00x |
|                           |      (100, 1000, 1000) | 100000000 |    3 |    n/a |      202ms |  503ms |    13ms |          n/a |           15.68x |      38.98x |         1.00x |

</details>

[^1][^2][^3][^4]

[^1]:
    Benchmarks were run on a Mac M3 Max laptop in September 2024 on numbagg's HEAD,
    pandas 2.2.2, bottleneck 1.4.0 numpy 2.0.1, with `python
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
# Testing CI trigger
