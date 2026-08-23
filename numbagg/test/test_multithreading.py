from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


# The `move_*` comparisons default to `window=20`, so a length-20 array leaves every
# window position short of `min_count` and every result is NaN. Keep the shape well
# clear of the window so the comparison below actually has values to compare.
@pytest.mark.parametrize("shape", [(200,)], indirect=True)
def test_multithreading(func_callable):
    # Test whether the functions work in a multithreaded context
    # First compile outside the threadpool to ensure the caching doesn't affect it
    expected = func_callable()
    with ThreadPoolExecutor(max_workers=2) as executor:
        foo = [executor.submit(func_callable) for _ in range(5)]
        results = [f.result() for f in foo]
        assert len(results) == 5
        # `func_callable` is deterministic, so a threaded call must match the
        # single-threaded baseline — otherwise the kernels race on shared state.
        for result in results:
            np.testing.assert_allclose(np.asarray(result), np.asarray(expected))
