"""Check that the README's Python snippets actually run.

The README is the project's main documentation, and its snippets go stale
silently — nothing else imports them. Each ``python`` block is executed with
``np`` and ``nb`` predefined (the blocks under "Axis parameter behavior" are
fragments that assume those names), and any ``# result.shape is (...)`` comment
is checked against the value the block produced.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

import numbagg as nb

README = Path(__file__).parents[2] / "README.md"
PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"

PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.M | re.S)
SHAPE_COMMENT = re.compile(r"^# result\.shape is (\([^)]*\))", re.M)


def _blocks() -> list[str]:
    # `numbagg.test` ships in the wheel, so the README isn't always alongside it.
    return PYTHON_BLOCK.findall(README.read_text()) if README.exists() else []


pytestmark = pytest.mark.skipif(
    not README.exists(), reason="README.md isn't installed with the package"
)


@pytest.mark.parametrize("i", range(len(_blocks())))
def test_readme_python_blocks(i):
    block = _blocks()[i]

    namespace: dict = {"np": np, "nb": nb}
    exec(compile(block, f"README.md:block{i}", "exec"), namespace)

    for shape in SHAPE_COMMENT.findall(block):
        assert "result" in namespace, (
            f"block {i} documents `result.shape` but never assigns `result`"
        )
        assert namespace["result"].shape == ast.literal_eval(shape)


def test_readme_has_python_blocks():
    # Guards the parametrization above: a regex that stops matching would
    # otherwise turn every block into a silently-skipped test.
    assert len(_blocks()) >= 4


def test_readme_python_floor_matches_pyproject():
    # The Installation section states the Python floor in prose, duplicating
    # `requires-python`. Nothing else ties the two together, so a floor bump
    # would leave the README — and, via `readme = "README.md"`, the PyPI page —
    # advertising support for a version the package rejects. Read the floor from
    # `pyproject.toml` rather than the installed distribution's metadata: that
    # file is what a bump edits, and `numbagg/__init__.py` treats absent
    # distribution metadata as a supported state ("Local copy or not installed").
    stated = re.search(r"supports Python (\d+\.\d+) and later", README.read_text())
    assert stated, "README no longer states a Python floor"
    declared = re.search(r'^requires-python = "([^"]+)"', PYPROJECT.read_text(), re.M)
    assert declared, "pyproject.toml no longer declares requires-python"
    assert f">={stated.group(1)}" in declared.group(1), (
        f"README says Python {stated.group(1)} and later, "
        f"but requires-python is {declared.group(1)}"
    )
