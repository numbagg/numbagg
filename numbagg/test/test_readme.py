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
