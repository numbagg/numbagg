import logging

import pytest


@pytest.fixture(autouse=True)
def numba_logger():
    # This is exteremly noisy, so we turn it off. We can make this a setting if it would
    # be occasionally useful.
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
