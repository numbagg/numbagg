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

| func                |  numbagg |    pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------ | -------: | --------: | ---------: | -----------: | ---------------: |
| `bfill`             |  55.20ms |  594.29ms |    60.87ms |       10.77x |            1.10x |
| `ffill`             |  52.21ms |  590.25ms |    55.24ms |       11.31x |            1.06x |
| `move_corr`         | 150.35ms | 3530.51ms |        n/a |       23.48x |              n/a |
| `move_cov`          | 126.73ms | 2438.95ms |        n/a |       19.25x |              n/a |
| `move_mean`         | 101.72ms |  394.02ms |    95.03ms |        3.87x |            0.93x |
| `move_std`          |  80.35ms |  566.41ms |   105.56ms |        7.05x |            1.31x |
| `move_sum`          | 102.02ms |  370.33ms |    75.96ms |        3.63x |            0.74x |
| `move_var`          |  83.61ms |  542.07ms |   113.81ms |        6.48x |            1.36x |
| `move_exp_nancorr`  | 195.03ms | 1571.76ms |        n/a |        8.06x |              n/a |
| `move_exp_nancount` | 105.58ms |  239.07ms |        n/a |        2.26x |              n/a |
| `move_exp_nancov`   | 137.16ms |  879.90ms |        n/a |        6.42x |              n/a |
| `move_exp_nanmean`  | 109.33ms |  211.15ms |        n/a |        1.93x |              n/a |
| `move_exp_nanstd`   | 141.98ms |  274.00ms |        n/a |        1.93x |              n/a |
| `move_exp_nansum`   | 102.38ms |  190.09ms |        n/a |        1.86x |              n/a |
| `move_exp_nanvar`   | 134.98ms |  247.61ms |        n/a |        1.83x |              n/a |

<details>
<summary>Full benchmarks</summary>

| func                |     size |  numbagg |    pandas | bottleneck | pandas_ratio | bottleneck_ratio |
| :------------------ | -------: | -------: | --------: | ---------: | -----------: | ---------------: |
| `bfill`             |     1000 |   0.01ms |    0.15ms |     0.00ms |       30.43x |            0.58x |
|                     |   100000 |   0.49ms |    5.57ms |     0.61ms |       11.43x |            1.26x |
|                     | 10000000 |  55.20ms |  594.29ms |    60.87ms |       10.77x |            1.10x |
| `ffill`             |     1000 |   0.01ms |    0.15ms |     0.00ms |       28.97x |            0.38x |
|                     |   100000 |   0.49ms |    5.60ms |     0.60ms |       11.34x |            1.22x |
|                     | 10000000 |  52.21ms |  590.25ms |    55.24ms |       11.31x |            1.06x |
| `move_corr`         |     1000 |   0.01ms |    0.82ms |        n/a |       94.59x |              n/a |
|                     |   100000 |   1.43ms |   26.70ms |        n/a |       18.65x |              n/a |
|                     | 10000000 | 150.35ms | 3530.51ms |        n/a |       23.48x |              n/a |
| `move_cov`          |     1000 |   0.01ms |    0.72ms |        n/a |      101.91x |              n/a |
|                     |   100000 |   1.22ms |   18.54ms |        n/a |       15.17x |              n/a |
|                     | 10000000 | 126.73ms | 2438.95ms |        n/a |       19.25x |              n/a |
| `move_mean`         |     1000 |   0.00ms |    0.13ms |     0.00ms |       28.48x |            0.70x |
|                     |   100000 |   0.91ms |    3.30ms |     0.88ms |        3.65x |            0.97x |
|                     | 10000000 | 101.72ms |  394.02ms |    95.03ms |        3.87x |            0.93x |
| `move_std`          |     1000 |   0.01ms |    0.17ms |     0.01ms |       22.87x |            1.12x |
|                     |   100000 |   0.71ms |    5.24ms |     1.00ms |        7.36x |            1.40x |
|                     | 10000000 |  80.35ms |  566.41ms |   105.56ms |        7.05x |            1.31x |
| `move_sum`          |     1000 |   0.00ms |    0.13ms |     0.00ms |       28.04x |            0.66x |
|                     |   100000 |   0.90ms |    3.37ms |     0.74ms |        3.74x |            0.82x |
|                     | 10000000 | 102.02ms |  370.33ms |    75.96ms |        3.63x |            0.74x |
| `move_var`          |     1000 |   0.01ms |    0.15ms |     0.01ms |       20.86x |            1.12x |
|                     |   100000 |   0.68ms |    5.32ms |     0.97ms |        7.81x |            1.42x |
|                     | 10000000 |  83.61ms |  542.07ms |   113.81ms |        6.48x |            1.36x |
| `move_exp_nancorr`  |     1000 |   0.02ms |    0.65ms |        n/a |       33.53x |              n/a |
|                     |   100000 |   1.88ms |   13.82ms |        n/a |        7.34x |              n/a |
|                     | 10000000 | 195.03ms | 1571.76ms |        n/a |        8.06x |              n/a |
| `move_exp_nancount` |     1000 |   0.01ms |    0.14ms |        n/a |       11.34x |              n/a |
|                     |   100000 |   0.99ms |    2.33ms |        n/a |        2.36x |              n/a |
|                     | 10000000 | 105.58ms |  239.07ms |        n/a |        2.26x |              n/a |
| `move_exp_nancov`   |     1000 |   0.01ms |    0.58ms |        n/a |       43.37x |              n/a |
|                     |   100000 |   1.32ms |    9.53ms |        n/a |        7.24x |              n/a |
|                     | 10000000 | 137.16ms |  879.90ms |        n/a |        6.42x |              n/a |
| `move_exp_nanmean`  |     1000 |   0.01ms |    0.11ms |        n/a |        8.47x |              n/a |
|                     |   100000 |   0.98ms |    2.22ms |        n/a |        2.25x |              n/a |
|                     | 10000000 | 109.33ms |  211.15ms |        n/a |        1.93x |              n/a |
| `move_exp_nanstd`   |     1000 |   0.02ms |    0.17ms |        n/a |       10.26x |              n/a |
|                     |   100000 |   1.33ms |    2.90ms |        n/a |        2.18x |              n/a |
|                     | 10000000 | 141.98ms |  274.00ms |        n/a |        1.93x |              n/a |
| `move_exp_nansum`   |     1000 |   0.01ms |    0.10ms |        n/a |        8.21x |              n/a |
|                     |   100000 |   0.95ms |    1.97ms |        n/a |        2.08x |              n/a |
|                     | 10000000 | 102.38ms |  190.09ms |        n/a |        1.86x |              n/a |
| `move_exp_nanvar`   |     1000 |   0.02ms |    0.11ms |        n/a |        6.97x |              n/a |
|                     |   100000 |   1.25ms |    2.54ms |        n/a |        2.03x |              n/a |
|                     | 10000000 | 134.98ms |  247.61ms |        n/a |        1.83x |              n/a |

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
