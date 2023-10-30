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

| func                | numbagg | pandas |  ratio |
| :------------------ | ------: | -----: | -----: |
| `move_corr`         |   145ms | 2965ms | 20.39x |
| `move_cov`          |   123ms | 2014ms | 16.37x |
| `move_mean`         |    94ms |  390ms |  4.17x |
| `move_std`          |    73ms |  560ms |  7.65x |
| `move_sum`          |    92ms |  371ms |  4.05x |
| `move_var`          |    69ms |  526ms |  7.62x |
| `move_exp_nancorr`  |   191ms | 1482ms |  7.77x |
| `move_exp_nancount` |    98ms |  224ms |  2.29x |
| `move_exp_nancov`   |   131ms |  878ms |  6.68x |
| `move_exp_nanmean`  |    97ms |  215ms |  2.21x |
| `move_exp_nanstd`   |   139ms |  292ms |  2.10x |
| `move_exp_nansum`   |    94ms |  196ms |  2.10x |
| `move_exp_nanvar`   |   123ms |  247ms |  2.01x |

<details>
<summary>Full benchmarks</summary>

| func                |     size |  numbagg |    pandas |   ratio |
| :------------------ | -------: | -------: | --------: | ------: |
| `move_corr`         |     1000 |   0.01ms |    0.86ms |  99.07x |
|                     |   100000 |   1.40ms |   27.13ms |  19.35x |
|                     | 10000000 | 145.39ms | 2965.21ms |  20.39x |
| `move_cov`          |     1000 |   0.01ms |    0.74ms | 106.56x |
|                     |   100000 |   1.19ms |   18.18ms |  15.23x |
|                     | 10000000 | 123.02ms | 2014.06ms |  16.37x |
| `move_mean`         |     1000 |   0.00ms |    0.14ms |  29.81x |
|                     |   100000 |   0.88ms |    3.24ms |   3.69x |
|                     | 10000000 |  93.55ms |  390.48ms |   4.17x |
| `move_std`          |     1000 |   0.01ms |    0.17ms |  23.45x |
|                     |   100000 |   0.76ms |    5.21ms |   6.86x |
|                     | 10000000 |  73.16ms |  559.98ms |   7.65x |
| `move_sum`          |     1000 |   0.00ms |    0.14ms |  29.25x |
|                     |   100000 |   0.91ms |    3.05ms |   3.36x |
|                     | 10000000 |  91.63ms |  371.12ms |   4.05x |
| `move_var`          |     1000 |   0.01ms |    0.16ms |  21.28x |
|                     |   100000 |   0.71ms |    4.82ms |   6.77x |
|                     | 10000000 |  69.03ms |  525.93ms |   7.62x |
| `move_exp_nancorr`  |     1000 |   0.02ms |    0.69ms |  36.65x |
|                     |   100000 |   1.97ms |   13.85ms |   7.02x |
|                     | 10000000 | 190.80ms | 1482.14ms |   7.77x |
| `move_exp_nancount` |     1000 |   0.01ms |    0.11ms |   8.55x |
|                     |   100000 |   0.98ms |    2.05ms |   2.09x |
|                     | 10000000 |  97.66ms |  223.60ms |   2.29x |
| `move_exp_nancov`   |     1000 |   0.01ms |    0.59ms |  43.28x |
|                     |   100000 |   1.31ms |    9.98ms |   7.62x |
|                     | 10000000 | 131.45ms |  877.99ms |   6.68x |
| `move_exp_nanmean`  |     1000 |   0.01ms |    0.11ms |   8.75x |
|                     |   100000 |   1.03ms |    2.22ms |   2.15x |
|                     | 10000000 |  97.40ms |  215.02ms |   2.21x |
| `move_exp_nanstd`   |     1000 |   0.02ms |    0.19ms |  11.81x |
|                     |   100000 |   1.41ms |    3.44ms |   2.44x |
|                     | 10000000 | 138.97ms |  291.62ms |   2.10x |
| `move_exp_nansum`   |     1000 |   0.01ms |    0.11ms |   7.90x |
|                     |   100000 |   0.96ms |    2.04ms |   2.13x |
|                     | 10000000 |  93.59ms |  196.27ms |   2.10x |
| `move_exp_nanvar`   |     1000 |   0.02ms |    0.12ms |   7.45x |
|                     |   100000 |   1.22ms |    2.56ms |   2.10x |
|                     | 10000000 | 122.70ms |  247.01ms |   2.01x |

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
