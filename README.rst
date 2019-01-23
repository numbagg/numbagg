Numbagg: Fast N-dimensional aggregation functions with Numba
============================================================

.. image:: https://travis-ci.org/shoyer/numbagg.svg?branch=master
    :target: https://travis-ci.org/shoyer/numbagg

Re-implementations of (some) functions found in Bottleneck_ with Numba_ and
NumPy's `generalized ufuncs`_.

.. _Bottleneck: https://github.com/kwgoodman/bottleneck
.. _Numba: https://github.com/numba/numba
.. _generalized ufuncs: http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html

Currently accelerated functions: ``allnan``, ``anynan``, ``count``,
``nanargmax``, ``nanargmin``, ``nansum``, ``nanmean``, ``nanstd``, ``nanvar``,
``nanmax``, ``nanmin``, ``move_nanmean``.

Easy to extend
--------------

Numbagg makes it easy to write, in pure Python/NumPy, flexible aggregation
functions accelerated by Numba. All the hard work is done by Numba's
JIT compiler and NumPy's gufunc machinery (as wrapped by Numba).

For example, here is how we wrote ``nansum``::

    import numpy as np
    from numbagg.decorators import ndreduce

    @ndreduce
    def nansum(a):
        asum = 0.0
        for ai in a.flat:
            if not np.isnan(ai):
                asum += ai
        return asum

Advantages over Bottleneck
--------------------------

* Way less code. Easier to add new functions. No ad-hoc templating system.
  No Cython!
* Fast functions still work for >3 dimensions.
* ``axis`` argument handles tuples of integers.
* ufunc broadcasting lets us supply an array for the ``window`` in moving
  window functions.

The functions in Numbagg are adapted from (and soon to be tested against)
Bottleneck's battle-hardened Cython. Still, Numbagg is experimental, and
probably not yet ready for production.

.. Differences
.. -----------

.. * ``nanargmax`` and ``nanargmin`` currently return ``-1`` when they encounter
..   an array of all ``NaN`` instead of raising an exception like numpy. This due
..   to a limitation of numba, but it's also arguably more useful.

Benchmarks
----------

Initial benchmarks are quite encouraging. Numbagg/Numba has comparable
(slightly better) performance than Bottleneck's hand-written C::

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

Philosophy
----------

Numbagg includes somewhat awkward workarounds for features missing from
NumPy/Numba:

* It implements its own cache for functions wrapped by Numba's ``guvectorize``,
  because that decorator is rather slow.
* It does its `own handling of array transposes <https://github.com/shoyer/numbagg/blob/master/numbagg/decorators.py#L69>`_ to handle the ``axis`` argument,
  which we hope will `eventually be directly supported <https://github.com/numpy/numpy/issues/5197>`_
  by all NumPy gufuncs.
* It uses some `terrible hacks <https://github.com/shoyer/numbagg/blob/master/numbagg/transform.py>`_
  to hide the out-of-bound memory access necessary to write
  `gufuncs that handle scalar values <https://github.com/numba/numba/blob/master/numba/tests/test_guvectorize_scalar.py>`_ with Numba.

I hope that the need for most of these will eventually go away. In the
meantime, expect Numbagg to be tightly coupled to Numba and NumPy release
cycles.

License
-------

3-clause BSD. Includes portions of Bottleneck, which is distributed under a
Simplified BSD license.
