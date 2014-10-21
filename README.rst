Numbagg: Fast N-dimensional aggregation functions with Numba
============================================================

.. image:: https://travis-ci.org/shoyer/numbagg.svg?branch=master
    :target: https://travis-ci.org/shoyer/numbagg

Re-implementations of (a few) functions found in Bottleneck_ with Numba_ and
NumPy's `generalized ufuncs`_.

.. _Bottleneck: https://github.com/kwgoodman/bottleneck
.. _Numba: https://github.com/numba/numba
.. _generalized ufuncs: http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html

Currently accelerated functions: ``nansum``, ``nanmean``, ``nanmin``,
``count``, ``move_nanmean``.

Easy to extend
--------------

Numbagg makes it easy to write, in pure Python/NumPy, flexible aggregation
functions accelerated by Numba. These aggregation functions work on arrays with
any number of dimensions and support an ``axis`` argument that handles
``None``, integers and tuples of integers. All the hard work is done by Numba's
JIT compiler and NumPy's gufunc machinery (as wrapped by Numba).

For example, here is how we wrote ``nansum``::

    import numpy as np
    from numbagg.decorators import ndreduce

    @ndreduce
    def nansum(a):
        asum = 0.0
        for ai in a.flat:
            if np.nansum(ai):
                asum += ai
        return asum

Not bad, huh?

Benchmarks
----------

Initial benchmarks are quite encouraging. In many cases, Numbagg/Numba has
competitive performance with Bottleneck/Cython::

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

License
-------

MIT. Includes portions of Bottleneck, which is distributed under a
Simplified BSD license.
