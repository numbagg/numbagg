from concurrent.futures import ThreadPoolExecutor

import numpy as np


def test_multithreading(func_callable):
    # Test whether the functions work in a multithreaded context

    # First compile outside the threadpool to ensure the caching doesn't affect it
    func_callable()
    with ThreadPoolExecutor(max_workers=2) as executor:
        foo = [executor.submit(func_callable) for _ in range(10)]
        results = [f.result() for f in foo]
        assert (len(results)) == 10
        assert all([isinstance(r, np.ndarray) for r in results])
