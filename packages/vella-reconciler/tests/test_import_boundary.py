"""Import-boundary gate.

The reconciler depends DOWNWARD only and through the published front door. This
test walks every ``*.py`` under ``src/vella/reconciler``, parses it with ``ast``,
and asserts:

* no import targets a private ``vella.runtime._*`` symbol or submodule; and
* every name imported ``from vella.runtime`` is one of the 7 published symbols in
  ``vella.runtime.__all__``.

The runtime's private internals (``vella.runtime._inmemory`` and friends) are off
limits by construction; only the 7-symbol contract is allowed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "vella" / "reconciler"

# The 7 published symbols in vella.runtime.__all__ (the only allowed imports).
_ALLOWED_RUNTIME = frozenset(
    {
        "ConcurrencyConflict",
        "Cursor",
        "LogEntry",
        "Runtime",
        "Store",
        "StoreTxn",
        "TransitionKind",
    }
)

_PRIVATE_RUNTIME = re.compile(r"^vella\.runtime\._")


def _modules() -> list[Path]:
    return sorted(_SRC.rglob("*.py"))


def test_no_private_runtime_imports() -> None:
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _PRIVATE_RUNTIME.match(alias.name):
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _PRIVATE_RUNTIME.match(module):
                    offenders.append(f"{path.name}: from {module} import ...")
    assert not offenders, f"private vella.runtime imports: {offenders}"


def test_runtime_imports_are_in_the_published_surface() -> None:
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "vella.runtime":
                for alias in node.names:
                    if alias.name not in _ALLOWED_RUNTIME:
                        offenders.append(f"{path.name}: from vella.runtime import {alias.name}")
    assert not offenders, f"unpublished vella.runtime imports: {offenders}"
