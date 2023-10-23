# Numbagg: Fast N-dimensional aggregation functions with Numba

[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/numbagg/numbagg/test.yaml?branch=main&logo=github&style=for-the-badge)](https://github.com/numbagg/numbagg/actions/workflows/test.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/numbagg?style=for-the-badge)](https://pypi.python.org/pypi/numbagg/)

Fast, flexible N-dimensional array functions written with
[Numba](https://github.com/numba/numba) and NumPy's [generalized
ufuncs](http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html).

Currently accelerated functions:

- Array functions: `allnan`, `anynan`, `count`, `nanargmax`,
  `nanargmin`, `nanmax`, `nanmean`, `nanstd`, `nanvar`, `nanmin`,
  `nansum`
- Moving window functions: `move_exp_nanmean`, `move_exp_nansum`,
  `move_exp_nanvar`, `move_mean`, `move_sum`

Note: Only functions listed here (exposed in Numbagg's top level namespace) are
supported as part of Numbagg's public API.

## Benchmarks[^1]

| func                |     size |  numbagg |    pandas |  ratio |
| :------------------ | -------: | -------: | --------: | -----: |
| `move_corr`         |     1000 |   0.01ms |    0.79ms | 93.02x |
|                     |   100000 |   1.40ms |   26.57ms | 18.99x |
|                     | 10000000 | 144.50ms | 2786.15ms | 19.28x |
| `move_cov`          |     1000 |   0.01ms |    0.68ms | 96.10x |
|                     |   100000 |   1.18ms |   18.10ms | 15.31x |
|                     | 10000000 | 122.73ms | 1926.51ms | 15.70x |
| `move_exp_nancorr`  |     1000 |   0.02ms |    0.60ms | 32.16x |
|                     |   100000 |   1.88ms |   13.49ms |  7.19x |
|                     | 10000000 | 192.36ms | 1379.48ms |  7.17x |
| `move_exp_nancount` |     1000 |   0.01ms |    0.10ms |  8.37x |
|                     |   100000 |   0.95ms |    2.01ms |  2.12x |
|                     | 10000000 |  96.27ms |  218.17ms |  2.27x |
| `move_exp_nancov`   |     1000 |   0.01ms |    0.52ms | 39.64x |
|                     |   100000 |   1.28ms |    9.57ms |  7.50x |
|                     | 10000000 | 130.04ms |  907.02ms |  6.97x |
| `move_exp_nanmean`  |     1000 |   0.01ms |    0.10ms |  8.19x |
|                     |   100000 |   1.01ms |    2.12ms |  2.10x |
|                     | 10000000 |  96.71ms |  209.90ms |  2.17x |
| `move_exp_nanstd`   |     1000 |   0.02ms |    0.16ms |  9.99x |
|                     |   100000 |   1.32ms |    2.78ms |  2.12x |
|                     | 10000000 | 131.97ms |  276.32ms |  2.09x |
| `move_exp_nansum`   |     1000 |   0.01ms |    0.10ms |  8.35x |
|                     |   100000 |   0.96ms |    1.92ms |  2.00x |
|                     | 10000000 |  93.43ms |  186.85ms |  2.00x |
| `move_exp_nanvar`   |     1000 |   0.02ms |    0.11ms |  6.98x |
|                     |   100000 |   1.22ms |    2.47ms |  2.02x |
|                     | 10000000 | 123.18ms |  241.24ms |  1.96x |
| `move_mean`         |     1000 |   0.00ms |    0.13ms | 28.27x |
|                     |   100000 |   0.91ms |    3.20ms |  3.52x |
|                     | 10000000 |  90.63ms |  356.37ms |  3.93x |
| `move_std`          |     1000 |   0.01ms |    0.17ms | 22.79x |
|                     |   100000 |   0.71ms |    5.26ms |  7.36x |
|                     | 10000000 |  72.28ms |  546.70ms |  7.56x |
| `move_sum`          |     1000 |   0.00ms |    0.13ms | 27.45x |
|                     |   100000 |   0.89ms |    3.01ms |  3.37x |
|                     | 10000000 |  89.51ms |  351.26ms |  3.92x |
| `move_var`          |     1000 |   0.01ms |    0.15ms | 20.61x |
|                     |   100000 |   0.68ms |    4.84ms |  7.08x |
|                     | 10000000 |  69.95ms |  518.37ms |  7.41x |

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
Bottleneck's battle-hardened implementations. Still, Numbagg is experimental,
and probably not yet ready for production.

## Benchmarks

Initial benchmarks are quite encouraging. Numbagg/Numba has comparable (slightly
better) performance than Bottleneck's hand-written C:

```python
import numbagg
import numpy as np
import bottleneck

x = np.random.RandomState(42).randn(1000, 1000)
x[x < -1] = np.NaN

# timings with numba=0.41.0 and bottleneck=1.2.1

In [2]: %timeit numbagg.nanmean(x)
1.8 ms ± 92.3 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [3]: %timeit numbagg.nanmean(x, axis=0)
3.63 ms ± 136 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [4]: %timeit numbagg.nanmean(x, axis=1)
1.81 ms ± 41 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

In [5]: %timeit bottleneck.nanmean(x)
2.22 ms ± 119 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [6]: %timeit bottleneck.nanmean(x, axis=0)
4.45 ms ± 107 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [7]: %timeit bottleneck.nanmean(x, axis=1)
2.19 ms ± 13.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

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
