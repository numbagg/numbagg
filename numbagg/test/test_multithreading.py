from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


@pytest.mark.parametrize("library", ["numbagg"], indirect=True)
def test_multithreading(func_callable):
    # Test whether the functions work in a multithreaded context

    with ThreadPoolExecutor(max_workers=2) as executor:
        foo = [executor.submit(func_callable) for _ in range(10)]
        results = [f.result() for f in foo]
        assert (len(results)) == 10
        assert all([isinstance(r, np.ndarray) for r in results])
