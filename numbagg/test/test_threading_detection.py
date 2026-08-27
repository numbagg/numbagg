"""Tests for threading layer detection functions."""

import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numba
import pytest

from numbagg.decorators import (
    _is_in_unsafe_thread_pool,
    _is_threading_layer_threadsafe,
    _thread_backend,
)


@pytest.fixture
def reset_numba_config() -> Iterator[None]:
    """Fixture to save/restore numba configuration and clear cache."""
    # Save original config (numba.config attributes are not in type stubs)
    orig_layer = numba.config.THREADING_LAYER  # ty:ignore[unresolved-attribute]
    orig_priority = list(numba.config.THREADING_LAYER_PRIORITY)  # ty:ignore[unresolved-attribute]

    # Clear cache before test
    _thread_backend.cache_clear()

    yield

    # Restore original config after test
    numba.config.THREADING_LAYER = orig_layer  # ty:ignore[unresolved-attribute]
    numba.config.THREADING_LAYER_PRIORITY = orig_priority  # ty:ignore[unresolved-attribute]
    _thread_backend.cache_clear()


class TestThreadingDetection:
    """Test the threading layer detection functions."""

    def test_thread_backend_default(self):
        """Test that _thread_backend returns a valid backend."""
        backend = _thread_backend()
        assert backend in {"tbb", "omp", "workqueue"}

    @pytest.mark.parametrize(
        "priority",
        [
            ["workqueue", "omp", "tbb"],
            ["tbb", "workqueue", "omp"],  # tbb likely unavailable, falls back
            ["omp", "tbb", "workqueue"],  # omp likely unavailable, falls back
        ],
    )
    def test_thread_backend_respects_priority(self, reset_numba_config, priority):
        """Test that backend respects THREADING_LAYER_PRIORITY."""
        numba.config.THREADING_LAYER = "default"  # ty:ignore[unresolved-attribute]
        numba.config.THREADING_LAYER_PRIORITY = priority  # ty:ignore[unresolved-attribute]

        # workqueue is always available, so it should be selected when first
        backend = _thread_backend()
        if priority[0] == "workqueue":
            assert backend == "workqueue"
        else:
            # If other backends are first but unavailable, falls back to workqueue
            assert backend in {"tbb", "omp", "workqueue"}

    @pytest.mark.parametrize(
        "layer,expected",
        [
            ("workqueue", {"workqueue"}),
            ("tbb", {"tbb", "workqueue"}),  # Falls back to workqueue if tbb unavailable
            ("omp", {"omp", "workqueue"}),  # Falls back to workqueue if omp unavailable
        ],
    )
    def test_thread_backend_explicit_layer(self, reset_numba_config, layer, expected):
        """Test that explicit layer selection works."""
        numba.config.THREADING_LAYER = layer  # ty:ignore[unresolved-attribute]
        backend = _thread_backend()
        assert backend in expected

    @pytest.mark.parametrize(
        "category,allowed_backends",
        [
            ("default", {"tbb", "omp", "workqueue"}),
            ("safe", {"tbb", "workqueue"}),  # Only tbb allowed, falls back to workqueue
            ("threadsafe", {"tbb", "omp", "workqueue"}),
            (
                "forksafe",
                {"tbb", "omp", "workqueue"},
            ),  # All could be valid depending on platform
        ],
    )
    def test_thread_backend_layer_categories(
        self, reset_numba_config, category, allowed_backends
    ):
        """Test that layer categories work correctly."""
        numba.config.THREADING_LAYER = category  # ty:ignore[unresolved-attribute]
        backend = _thread_backend()
        assert backend in allowed_backends

    def test_is_threading_layer_threadsafe(self):
        """Test thread safety detection."""
        backend = _thread_backend()
        is_safe = _is_threading_layer_threadsafe()

        # Check consistency
        if backend in {"tbb", "omp"}:
            assert is_safe is True
        elif backend == "workqueue":
            assert is_safe is False

    def test_is_in_unsafe_thread_pool_main_thread(self):
        """Test that main thread is not considered unsafe."""
        # In the main thread, should always return False
        assert _is_in_unsafe_thread_pool() is False

    def test_is_in_unsafe_thread_pool_main_thread_unsafe_backend(self):
        """The main thread is safe even when the backend isn't."""
        # Without pinning the backend, this passes trivially on a runner where
        # tbb or omp is available — the thread-name check is never reached.
        with patch(
            "numbagg.decorators._is_threading_layer_threadsafe", return_value=False
        ):
            assert _is_in_unsafe_thread_pool() is False

    @pytest.mark.parametrize("threadsafe,expected", [(True, False), (False, True)])
    def test_is_in_unsafe_thread_pool_backend_branches(self, threadsafe, expected):
        """Both in-pool branches, independent of the runner's actual backend."""
        result: dict[str, bool] = {}

        def check_in_pool():
            result["in_pool"] = _is_in_unsafe_thread_pool()

        with patch(
            "numbagg.decorators._is_threading_layer_threadsafe",
            return_value=threadsafe,
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(check_in_pool).result()

        assert result["in_pool"] is expected

    def test_is_in_unsafe_thread_pool_executor(self):
        """Test detection inside ThreadPoolExecutor."""
        result: dict[str, bool | str | None] = {"in_pool": None, "thread_name": None}

        def check_in_pool():
            result["in_pool"] = _is_in_unsafe_thread_pool()
            result["thread_name"] = threading.current_thread().name

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(check_in_pool)
            future.result()

        # Thread name should indicate it's in a pool
        assert isinstance(result["thread_name"], str)
        assert result["thread_name"].startswith("ThreadPoolExecutor")

        # Whether it's unsafe depends on the backend
        backend = _thread_backend()
        if backend == "workqueue":
            assert result["in_pool"] is True
        else:
            # tbb or omp are safe
            assert result["in_pool"] is False

    def test_thread_backend_caching(self, reset_numba_config):
        """Test that _thread_backend memoizes rather than re-evaluating."""
        # The fixture clears the cache, so the first call is a miss and the
        # second a hit. Comparing the two return values alone wouldn't test
        # anything — the function is deterministic for a fixed config.
        assert _thread_backend() == _thread_backend()
        info = _thread_backend.cache_info()
        assert (info.misses, info.hits) == (1, 1)

    @pytest.mark.skipif(
        sys.platform == "linux", reason="Test is for non-Linux forksafe behavior"
    )
    def test_forksafe_includes_omp_non_linux(self):
        """Test that forksafe includes omp on non-Linux platforms."""
        from numbagg.decorators import _LAYER_CATEGORIES

        forksafe = _LAYER_CATEGORIES["forksafe"]
        assert "omp" in forksafe  # On non-Linux, omp should be in forksafe

    @pytest.mark.skipif(
        sys.platform != "linux", reason="Test is for Linux forksafe behavior"
    )
    def test_forksafe_excludes_omp_linux(self):
        """Test that forksafe excludes omp on Linux."""
        from numbagg.decorators import _LAYER_CATEGORIES

        forksafe = _LAYER_CATEGORIES["forksafe"]
        assert "omp" not in forksafe


class TestThreadingWithMocks:
    """Test with mocked imports to simulate different backend availability."""

    def test_all_backends_available(self, reset_numba_config):
        """Test when all backends are available."""
        with patch("numbagg.decorators.importlib.import_module") as mock_import:
            # Mock successful imports
            mock_import.return_value = True

            numba.config.THREADING_LAYER = "default"  # ty:ignore[unresolved-attribute]
            numba.config.THREADING_LAYER_PRIORITY = ["tbb", "omp", "workqueue"]  # ty:ignore[unresolved-attribute]

            # Should return tbb since it's first and "available"
            assert _thread_backend() == "tbb"

    def test_only_workqueue_available(self, reset_numba_config):
        """Test when only workqueue is available."""

        def mock_import_side_effect(name):
            if "tbb" in name or "omp" in name:
                raise ImportError(f"{name} not available")
            return True

        with patch("numbagg.decorators.importlib.import_module") as mock_import:
            mock_import.side_effect = mock_import_side_effect

            numba.config.THREADING_LAYER = "default"  # ty:ignore[unresolved-attribute]

            # Should return workqueue as fallback
            assert _thread_backend() == "workqueue"

    @pytest.mark.parametrize(
        "priority,category,expected",
        [
            # workqueue first: only the categories that admit it pick it up, so
            # `threadsafe` has to skip past it to omp.
            (["workqueue", "omp", "tbb"], "default", "workqueue"),
            (["workqueue", "omp", "tbb"], "safe", "tbb"),
            (["workqueue", "omp", "tbb"], "threadsafe", "omp"),
            (["workqueue", "omp", "tbb"], "forksafe", "workqueue"),
            # omp first: on Linux `forksafe` excludes it and falls through to tbb.
            (["omp", "tbb", "workqueue"], "default", "omp"),
            (["omp", "tbb", "workqueue"], "safe", "tbb"),
            (["omp", "tbb", "workqueue"], "threadsafe", "omp"),
            (
                ["omp", "tbb", "workqueue"],
                "forksafe",
                "tbb" if sys.platform == "linux" else "omp",
            ),
        ],
    )
    def test_layer_categories_pick_first_allowed_in_priority(
        self, reset_numba_config, priority, category, expected
    ):
        """A category takes the first *allowed* entry of the priority list.

        The unmocked category tests can only assert membership in the full set of
        backends, since a runner without tbb/omp falls back to workqueue for every
        category. Pretending all three are importable pins the actual semantics.
        """
        with patch("numbagg.decorators.importlib.import_module", return_value=True):
            numba.config.THREADING_LAYER = category  # ty:ignore[unresolved-attribute]
            numba.config.THREADING_LAYER_PRIORITY = priority  # ty:ignore[unresolved-attribute]

            assert _thread_backend() == expected

    @pytest.mark.parametrize("layer", ["tbb", "omp"])
    def test_explicit_layer_available(self, reset_numba_config, layer):
        """An explicitly requested backend is used when it imports."""
        with patch("numbagg.decorators.importlib.import_module", return_value=True):
            numba.config.THREADING_LAYER = layer  # ty:ignore[unresolved-attribute]

            assert _thread_backend() == layer

    @pytest.mark.parametrize("layer", ["tbb", "omp"])
    def test_explicit_layer_falls_back_when_unavailable(
        self, reset_numba_config, layer
    ):
        """An explicitly requested backend falls back when it doesn't import."""
        with patch(
            "numbagg.decorators.importlib.import_module", side_effect=ImportError
        ):
            numba.config.THREADING_LAYER = layer  # ty:ignore[unresolved-attribute]

            assert _thread_backend() == "workqueue"
