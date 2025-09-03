"""Tests for threading layer detection functions."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numba
import pytest

from numbagg.decorators import (
    _is_in_unsafe_thread_pool,
    _is_threading_layer_threadsafe,
    _thread_backend,
)


class TestThreadingDetection:
    """Test the threading layer detection functions."""

    def test_thread_backend_default(self):
        """Test that _thread_backend returns a valid backend."""
        backend = _thread_backend()
        assert backend in {"tbb", "omp", "workqueue"}

    def test_thread_backend_respects_priority(self):
        """Test that backend respects THREADING_LAYER_PRIORITY."""
        # Save original config
        orig_priority = list(numba.config.THREADING_LAYER_PRIORITY)
        orig_layer = numba.config.THREADING_LAYER

        try:
            # Clear the cache since we're changing config
            _thread_backend.cache_clear()

            # Test with workqueue first in priority
            numba.config.THREADING_LAYER = "default"
            numba.config.THREADING_LAYER_PRIORITY = ["workqueue", "omp", "tbb"]

            # Should return workqueue since it's first and always available
            assert _thread_backend() == "workqueue"

        finally:
            # Restore original config
            numba.config.THREADING_LAYER_PRIORITY = orig_priority
            numba.config.THREADING_LAYER = orig_layer
            _thread_backend.cache_clear()

    def test_thread_backend_explicit_layer(self):
        """Test that explicit layer selection works."""
        # Save original config
        orig_layer = numba.config.THREADING_LAYER

        try:
            # Clear the cache
            _thread_backend.cache_clear()

            # Explicitly select workqueue
            numba.config.THREADING_LAYER = "workqueue"
            assert _thread_backend() == "workqueue"

            # Clear cache for next test
            _thread_backend.cache_clear()

            # Try to select a backend that might not be available
            numba.config.THREADING_LAYER = "tbb"
            backend = _thread_backend()
            # Should either return tbb if available, or fallback to workqueue
            assert backend in {"tbb", "workqueue"}

        finally:
            # Restore original config
            numba.config.THREADING_LAYER = orig_layer
            _thread_backend.cache_clear()

    def test_thread_backend_layer_categories(self):
        """Test that layer categories work correctly."""
        # Save original config
        orig_layer = numba.config.THREADING_LAYER

        try:
            # Test "safe" category (only allows tbb)
            _thread_backend.cache_clear()
            numba.config.THREADING_LAYER = "safe"
            backend = _thread_backend()
            # Since tbb might not be available, it could fallback to workqueue
            assert backend in {"tbb", "workqueue"}

            # Test "threadsafe" category (allows tbb and omp)
            _thread_backend.cache_clear()
            numba.config.THREADING_LAYER = "threadsafe"
            backend = _thread_backend()
            assert backend in {"tbb", "omp", "workqueue"}

        finally:
            numba.config.THREADING_LAYER = orig_layer
            _thread_backend.cache_clear()

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

    def test_thread_backend_caching(self):
        """Test that _thread_backend result is cached."""
        # First call
        backend1 = _thread_backend()

        # Second call should return same result without re-evaluation
        # We can't easily test this without mocking, but we can verify
        # the result is consistent
        backend2 = _thread_backend()
        assert backend1 == backend2

    @pytest.mark.skipif(
        sys.platform == "linux", reason="Test is for non-Linux forksafe behavior"
    )
    def test_forksafe_includes_omp_non_linux(self):
        """Test that forksafe includes omp on non-Linux platforms."""
        from numbagg.decorators import _LAYER_CATEGORIES

        forksafe = _LAYER_CATEGORIES["forksafe"]
        assert "omp" in forksafe or sys.platform == "linux"

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

    def test_all_backends_available(self):
        """Test when all backends are available."""
        with patch("numbagg.decorators.importlib.import_module") as mock_import:
            # Mock successful imports
            mock_import.return_value = True

            # Clear cache and reset config
            _thread_backend.cache_clear()
            orig_layer = numba.config.THREADING_LAYER
            orig_priority = list(numba.config.THREADING_LAYER_PRIORITY)

            try:
                numba.config.THREADING_LAYER = "default"
                numba.config.THREADING_LAYER_PRIORITY = ["tbb", "omp", "workqueue"]

                # Should return tbb since it's first and "available"
                assert _thread_backend() == "tbb"

            finally:
                numba.config.THREADING_LAYER = orig_layer
                numba.config.THREADING_LAYER_PRIORITY = orig_priority
                _thread_backend.cache_clear()

    def test_only_workqueue_available(self):
        """Test when only workqueue is available."""

        def mock_import_side_effect(name):
            if "tbb" in name or "omp" in name:
                raise ImportError(f"{name} not available")
            return True

        with patch("numbagg.decorators.importlib.import_module") as mock_import:
            mock_import.side_effect = mock_import_side_effect

            # Clear cache
            _thread_backend.cache_clear()
            orig_layer = numba.config.THREADING_LAYER

            try:
                numba.config.THREADING_LAYER = "default"

                # Should return workqueue as fallback
                assert _thread_backend() == "workqueue"

            finally:
                numba.config.THREADING_LAYER = orig_layer
                _thread_backend.cache_clear()
