# Numbagg: Fast N-dimensional aggregation functions with Numba

[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/numbagg/numbagg/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/numbagg/numbagg/actions/workflows/test.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/numbagg?style=for-the-badge)](https://pypi.python.org/pypi/numbagg/)

Fast, flexible N-dimensional array functions written with
[Numba](https://github.com/numba/numba) and NumPy's [generalized
ufuncs](http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html).

Currently accelerated functions:

- Array functions: `allnan`, `anynan`, `count`, `nanargmax`,
  `nanargmin`, `nanmax`, `nanmean`, `nanstd`, `nanvar`, `nanmin`,
  `nansum`, `nanquantile`.
- Grouped functions: `group_nanall`, `group_nanany`, `group_nanargmax`,
  `group_nanargmin`, `group_nancount`, `group_nanfirst`, `group_nanlast`,
  `group_nanmax`, `group_nanmean`, `group_nanmin`, `group_nanprod`,
  `group_nanstd`, `group_nansum`, `group_nansum_of_squares`, `group_nanvar`.
- Moving window functions listed below
- Exponentially weighted moving functions listed below

## Benchmarks[^1]

| func                |  numbagg |    pandas |  ratio |
| :------------------ | -------: | --------: | -----: |
| `move_corr`         | 143.80ms | 2814.70ms | 19.57x |
| `move_cov`          | 121.79ms | 1904.31ms | 15.64x |
| `move_exp_nancorr`  | 189.31ms | 1354.62ms |  7.16x |
| `move_exp_nancount` |  96.72ms |  212.32ms |  2.20x |
| `move_exp_nancov`   | 127.37ms |  855.13ms |  6.71x |
| `move_exp_nanmean`  |  95.83ms |  210.79ms |  2.20x |
| `move_exp_nanstd`   | 132.17ms |  268.04ms |  2.03x |
| `move_exp_nansum`   |  92.92ms |  185.70ms |  2.00x |
| `move_exp_nanvar`   | 122.77ms |  237.48ms |  1.93x |
| `move_mean`         |  89.09ms |  360.08ms |  4.04x |
| `move_std`          |  72.78ms |  542.02ms |  7.45x |
| `move_sum`          |  92.01ms |  351.78ms |  3.82x |
| `move_var`          |  79.82ms |  534.13ms |  6.69x |

<details>
<summary>Full benchmarks</summary>

| func                |     size |  numbagg |    pandas |  ratio |
| :------------------ | -------: | -------: | --------: | -----: |
| `move_corr`         |     1000 |   0.01ms |    0.78ms | 92.47x |
|                     |   100000 |   1.39ms |   26.33ms | 18.94x |
|                     | 10000000 | 143.80ms | 2814.70ms | 19.57x |
| `move_cov`          |     1000 |   0.01ms |    0.67ms | 95.70x |
|                     |   100000 |   1.22ms |   17.99ms | 14.76x |
|                     | 10000000 | 121.79ms | 1904.31ms | 15.64x |
| `move_exp_nancorr`  |     1000 |   0.02ms |    0.60ms | 31.78x |
|                     |   100000 |   1.84ms |   13.44ms |  7.31x |
|                     | 10000000 | 189.31ms | 1354.62ms |  7.16x |
| `move_exp_nancount` |     1000 |   0.01ms |    0.10ms |  8.27x |
|                     |   100000 |   0.95ms |    2.04ms |  2.15x |
|                     | 10000000 |  96.72ms |  212.32ms |  2.20x |
| `move_exp_nancov`   |     1000 |   0.01ms |    0.51ms | 39.48x |
|                     |   100000 |   1.27ms |    9.32ms |  7.34x |
|                     | 10000000 | 127.37ms |  855.13ms |  6.71x |
| `move_exp_nanmean`  |     1000 |   0.01ms |    0.10ms |  8.17x |
|                     |   100000 |   0.95ms |    2.16ms |  2.26x |
|                     | 10000000 |  95.83ms |  210.79ms |  2.20x |
| `move_exp_nanstd`   |     1000 |   0.02ms |    0.16ms |  9.89x |
|                     |   100000 |   1.34ms |    2.71ms |  2.03x |
|                     | 10000000 | 132.17ms |  268.04ms |  2.03x |
| `move_exp_nansum`   |     1000 |   0.01ms |    0.10ms |  7.90x |
|                     |   100000 |   0.96ms |    1.85ms |  1.93x |
|                     | 10000000 |  92.92ms |  185.70ms |  2.00x |
| `move_exp_nanvar`   |     1000 |   0.02ms |    0.11ms |  7.01x |
|                     |   100000 |   1.23ms |    2.51ms |  2.04x |
|                     | 10000000 | 122.77ms |  237.48ms |  1.93x |
| `move_mean`         |     1000 |   0.00ms |    0.13ms | 27.48x |
|                     |   100000 |   0.90ms |    3.15ms |  3.51x |
|                     | 10000000 |  89.09ms |  360.08ms |  4.04x |
| `move_std`          |     1000 |   0.01ms |    0.16ms | 21.73x |
|                     |   100000 |   0.72ms |    4.99ms |  6.92x |
|                     | 10000000 |  72.78ms |  542.02ms |  7.45x |
| `move_sum`          |     1000 |   0.00ms |    0.13ms | 27.49x |
|                     |   100000 |   0.90ms |    3.04ms |  3.37x |
|                     | 10000000 |  92.01ms |  351.78ms |  3.82x |
| `move_var`          |     1000 |   0.01ms |    0.15ms | 18.91x |
|                     |   100000 |   0.71ms |    5.40ms |  7.66x |
|                     | 10000000 |  79.82ms |  534.13ms |  6.69x |

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

[^1]:
    Benchmarks were run on a Mac M1 in October 2023 on numbagg's HEAD and
    pandas 2.1.1. Any contributions to improve the benchmarks for other
    libraries are more than welcome.
