import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(20,)], indirect=True)
def test_multithreading(func_callable):
    # Test whether the functions work in a multithreaded context
    with warnings.catch_warnings():
        # TODO: `nanquantile` raises a warning here (and not in other testes...); I
        # can't figure out where it's coming from, and can't reproduce it locally. So
        # I'm ignoring so that we can still raise errors on other warnings.
        warnings.simplefilter("ignore")

        # First compile outside the threadpool to ensure the caching doesn't affect it
        func_callable()
        with ThreadPoolExecutor(max_workers=2) as executor:
            foo = [executor.submit(func_callable) for _ in range(5)]
            results = [f.result() for f in foo]
            assert (len(results)) == 5
            assert all([isinstance(r, np.ndarray) for r in results])
