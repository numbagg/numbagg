"""Check the `.pyi` stubs declare every re-exported name, with a return type.

`numbagg` ships `py.typed`, so a type checker resolving `numbagg.<name>` follows
the import in `numbagg/__init__.py` to the defining module. If that module has no
stub declaring the name, the checker falls back to the runtime source — where the
public functions are instances of the decorator classes in `decorators.py`, whose
`__call__` returns an untyped gufunc result. Callers then get `Any` back with no
error, which is worse than no stubs at all.
"""

import ast
from pathlib import Path

import numbagg

_PACKAGE = Path(numbagg.__file__).parent

# `grouped.py` has no stub; see CLAUDE.md.
_UNSTUBBED_MODULES = {"grouped"}


def _reexport_origins() -> dict[str, str]:
    """Map each name `numbagg/__init__.py` re-exports to the module it comes from."""
    tree = ast.parse((_PACKAGE / "__init__.py").read_text())
    return {
        alias.asname or alias.name: node.module
        for node in tree.body
        # `level == 1` picks out the `from .<module> import ...` lines, skipping
        # absolute imports such as `importlib.metadata`.
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module
        for alias in node.names
    }


def _declared_in_stub(module: str) -> set[str]:
    """The names a module's stub declares, whether defined there or re-exported."""
    stub = _PACKAGE / f"{module}.pyi"
    if not stub.exists():
        return set()
    tree = ast.parse(stub.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            # Aliases such as `count = nancount`.
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_reexported_names_are_declared_in_stubs():
    missing = sorted(
        f"{module}.{name}"
        for name, module in _reexport_origins().items()
        if module not in _UNSTUBBED_MODULES and name not in _declared_in_stub(module)
    )
    assert not missing, f"public names with no stub declaration: {missing}"


# `nansum(a)` reduces to a scalar while `nansum(a, axis=0)` returns an array, and the
# result dtype is not always the input's (`nanmax` on int32 returns int64). Expressing
# that needs overloads keyed on both `axis` and the input dtype — an open design
# question tracked in #811. Until it is settled these declarations stay unannotated, and
# the test below asserts this set matches them exactly so it cannot go stale.
_PENDING_RETURN_TYPES = {
    "funcs.allnan",
    "funcs.anynan",
    "funcs.nanargmax",
    "funcs.nanargmin",
    "funcs.nancount",
    "funcs.nanmax",
    "funcs.nanmean",
    "funcs.nanmin",
    "funcs.nanstd",
    "funcs.nansum",
    "funcs.nanvar",
}


def test_stub_declarations_have_return_types():
    # A stub declaration with no return annotation is worse than no stub at all, for the
    # reason in the module docstring: the call itself type-checks clean while everything
    # downstream of the result silently stops being checked.
    unannotated = {
        f"{stub.stem}.{node.name}"
        for stub in sorted(_PACKAGE.glob("*.pyi"))
        for node in ast.parse(stub.read_text()).body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.returns is None
    }
    assert unannotated == _PENDING_RETURN_TYPES, (
        "stub return-type coverage moved: "
        f"newly missing {sorted(unannotated - _PENDING_RETURN_TYPES)}, "
        f"no longer pending {sorted(_PENDING_RETURN_TYPES - unannotated)}"
    )
