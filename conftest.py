import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="Skip slow tests"
    )
    parser.addoption(
        "--run-nightly", action="store_true", default=False, help="Run nightly tests"
    )


def pytest_runtest_setup(item):
    # based on https://stackoverflow.com/questions/47559524
    if "slow" in item.keywords and item.config.getoption("--skip-slow"):
        pytest.skip(f"--skip-slow passed — skipping {item}")
    if "nightly" in item.keywords and not item.config.getoption("--run-nightly"):
        pytest.skip(f"--run-nightly not passed — skipping {item}")
