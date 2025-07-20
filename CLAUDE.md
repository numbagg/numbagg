# numbagg Development Guidelines

## Project Context

Numbagg provides fast N-dimensional aggregation functions using Numba and NumPy's generalized ufuncs. It focuses on performance through JIT compilation and parallelization, offering 2-30x speedups over pandas, bottleneck, and numpy for various operations.

### Key Documentation
- **README.md**: Overview, benchmarks, and usage examples
- **pyproject.toml**: Project configuration and dependencies

### Core Functions
The library provides aggregation functions (like `nansum`, `nanmean`), moving window functions (like `move_mean`, `move_std`), exponential moving functions, and grouping operations. All functions work on N-dimensional arrays with arbitrary axes.

## Running Commands

Prefix commands with `uv run` to ensure that the correct virtual environment is used. For example:
- `uv run pytest` instead of `python -m pytest`
- `uv run mypy` instead of `mypy`
- `uv run pre-commit run --all-files` instead of `pre-commit run --all-files`

## Testing

### Test Organization
- **Comparative testing**: Tests compare numbagg functions against pandas, bottleneck, and numpy equivalents
- **Property-based tests**: Uses hypothesis for mathematical property verification (in `test_property.py`)
- **Benchmark tests**: Performance testing with pytest-benchmark (in `test_benchmark.py`)
- **Edge case coverage**: Extensive testing with NaN, inf, empty arrays, and various dtypes

### Running Tests
```bash
# Run all tests
uv run pytest

# Skip slow tests for faster feedback
uv run pytest --skip-slow

# Run benchmarks (disabled by default)
uv run pytest --benchmark-enable test/test_benchmark.py

# Run specific test file
uv run pytest numbagg/test/test_funcs.py
```

### Test Configuration
- Tests are configured in `conftest.py` files (root and test directory)
- Custom markers: `slow` (longer tests), `nightly` (tests for nightly builds)
- The test suite uses fixtures extensively for parameterized testing across different array shapes and configurations

## Before Returning

Always run lints and tests before returning to the user:
- `uv run pre-commit run --all-files`
- `uv run pytest`
