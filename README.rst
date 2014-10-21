numbagg: Fast N-dimensional aggregation functions with Numba
============================================================

.. image:: https://travis-ci.org/shoyer/numbagg.svg?branch=master
    :target: https://travis-ci.org/shoyer/numbagg

Re-implementations of (some) functions found in bottleneck_ with Numba_ and
NumPy's `generalized ufuncs`_.

.. _bottleneck: https://github.com/kwgoodman/bottleneck
.. _Numba: https://github.com/numba/numba
.. _generalized ufuncs: http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html

Initial benchmarks are quite encouraging::

    import numbagg
    import numpy as np
    import bottleneck

    x = np.random.RandomState(42).randn(1000, 1000)
    x[x < -1] = np.NaN

    # timings with numba=0.15.1-20-gd877602 and bottleneck=0.8.0

    In [4]: %timeit numbagg.nanmean(x)
    100 loops, best of 3: 2.39 ms per loop

    In [5]: %timeit numbagg.nanmean(x, axis=0)
    100 loops, best of 3: 9.54 ms per loop

    In [6]: %timeit numbagg.nanmean(x, axis=1)
    100 loops, best of 3: 2.77 ms per loop

    In [7]: %timeit bottleneck.nanmean(x)
    100 loops, best of 3: 2.27 ms per loop

    In [8]: %timeit bottleneck.nanmean(x, axis=0)
    100 loops, best of 3: 9.03 ms per loop

    In [9]: %timeit bottleneck.nanmean(x, axis=1)
    100 loops, best of 3: 2.3 ms per loop

To see these performance numbers, you'll need to install the dev version of
Numba, as Numba's handling of the ``.flat`` iterator was sped up considerably
in `a recent PR`__.

__ https://github.com/numba/numba/pull/817

License: MIT
