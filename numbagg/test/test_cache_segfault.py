import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from numbagg.grouped import group_nanmean

pytestmark = pytest.mark.nightly


@pytest.fixture()
def clean_pycache():
    """Clean all __pycache__ directories, seems to be necessary to get a segfault repro"""
    # Clean before test
    for pycache in Path().rglob("__pycache__"):
        shutil.rmtree(pycache)

    yield

    # Clean after test
    for pycache in Path().rglob("__pycache__"):
        shutil.rmtree(pycache)


def f_numba(i, x):
    """Import and run a numba function from generated module"""
    mod = importlib.import_module(f"numba_segfault._{i:04d}")
    return mod.f(x)


@pytest.mark.xfail(
    reason="Known segfault issue with numba caching in multiprocessing - https://github.com/numba/numba/issues/4807"
)
def test_numba_cache_segfault():
    """Test that reproduces numba cache segfault with multiple processes

    From https://github.com/numba/numba/issues/4807#issuecomment-551167331
    """

    jobs = 32
    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = Path(tmpdir) / "numba_segfault"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").touch()

        # Create all the test modules up front
        for i in range(1600):
            with open(pkg_dir / f"_{i:04d}.py", "w") as fh:
                fh.write(
                    dedent(
                        """
                        from numba import guvectorize, f8

                        @guvectorize([(f8, f8[:])], "()->()", nopython=True, cache=True)
                        def f(x, out):
                            out[0] = x * 2
                        """
                    ).lstrip()
                )

        # Add package dir to Python path so we can import our generated modules
        sys.path.insert(0, str(tmpdir))
        with ProcessPoolExecutor(jobs) as pool:
            futures = []
            for i in range(1600):
                futures.extend([pool.submit(f_numba, i, x) for x in range(jobs)])

            for i, future in enumerate(futures):
                future.result()
                if i % 100 == 0:
                    print(f"Completed {i} numba tasks")


def numbagg_worker(i, x):
    """Worker function that creates similar work to the numba test"""
    arr = np.full(1, x)
    labels = np.zeros(1, dtype=np.int64)
    return group_nanmean(arr, labels)


@pytest.mark.xfail(
    reason="Known segfault issue with numba caching in multiprocessing - https://github.com/numba/numba/issues/4807"
)
def test_numbagg_cache_segfault(clean_pycache):
    """Test that reproduces numba cache segfault with numbagg's group_nanmean — doesn't
    seem to currently fail though, even when `_NUMBAGG_CACHE` is set to `True`"""
    jobs = 32

    with ProcessPoolExecutor(jobs) as pool:
        futures = []
        for i in range(1600):
            futures.extend([pool.submit(numbagg_worker, i, x) for x in range(jobs)])

        for i, future in enumerate(futures):
            future.result()
            if i % 100 == 0:
                print(f"Completed {i} numbagg tasks")


def test_numbagg_module_cache_segfault(clean_pycache):
    """
    Test that sets the cache to be true, and more closely reproduces the numba cache
    segfault repro with numbagg functions in separate modules, since I was having issues
    reproducing the segfault with the simpler `test_numbagg_cache_segfault` test"""

    # Create a temporary script that runs our test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            dedent(
                """
            import numpy as np
            from concurrent.futures import ProcessPoolExecutor
            from numbagg.grouped import group_nanmean

            def worker(i, x):
                arr = np.full(1, x)
                labels = np.zeros(1, dtype=np.int64)
                return group_nanmean(arr, labels)

            def main():
                jobs = 32
                with ProcessPoolExecutor(jobs) as pool:
                    futures = []
                    for i in range(100):
                        futures.extend([pool.submit(worker, i, x) for x in range(jobs)])

                    for future in futures:
                        future.result()

            if __name__ == "__main__":
                main()
        """
            )
        )
        f.flush()

        env = os.environ.copy()
        env["NUMBAGG_CACHE"] = "True"
        env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)  # project root

        # Run the script and expect it to fail
        process = subprocess.run(
            [sys.executable, f.name], env=env, capture_output=True, text=True
        )

        # Process should have failed with a segfault and show warning about caching
        assert "will likely cause segfaults" in process.stderr
        assert process.returncode != 0, "Process should have failed with segfault"
