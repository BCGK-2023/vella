"""Import-boundary gate.

The agent cognition core depends DOWNWARD only and through the published front
doors. This test walks every ``*.py`` under ``src/vella/agent``, parses it with
``ast``, and asserts:

* no import targets a private ``vella.runtime._*`` / ``vella.graph._*`` /
  ``vella.core._*`` symbol or submodule;
* every name imported ``from vella.runtime`` is one of the 7 published symbols in
  ``vella.runtime.__all__``, and every name imported ``from vella.graph`` is one of
  the published symbols in ``vella.graph.__all__`` (core is allowed through its full
  public surface — not enumerated here, but private ``vella.core._*`` is forbidden);
  and
* NO import targets ``vella.reconciler`` at all — the reconciler is a sibling, NOT a
  dependency (depend downward only). The agent depends on {core, runtime, graph};
  reconciler is off the dependency graph entirely.

This is **AST-based** (``ast.parse`` + ``ast.walk``) and NEVER executes an import.
That is load-bearing for the reconciler forbid: ``vella-reconciler`` is not installed
on this branch/worktree (its source is not in git here), so an execution-based check
could not even see a stray ``import vella.reconciler`` — but a parse-based walk does,
so the forbid holds regardless of what is installed. The runtime's private internals
(``vella.runtime._inmemory`` and friends), the graph's privates, and core's privates
are off limits by construction.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "vella" / "agent"

# The 7 published symbols in vella.runtime.__all__ (the only allowed runtime imports).
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

# The published symbols in vella.graph.__all__ (the only allowed graph imports).
_ALLOWED_GRAPH = frozenset(
    {
        "Clock",
        "GraphFollower",
        "GraphProjection",
        "GraphView",
        "ManualClock",
        "Match",
        "MaterializationMode",
        "MotifHop",
        "MotifPattern",
        "Neighbor",
        "Path",
        "WeightOverrideRequiresFullMode",
    }
)

_PRIVATE_RUNTIME = re.compile(r"^vella\.runtime\._")
_PRIVATE_GRAPH = re.compile(r"^vella\.graph\._")
_PRIVATE_CORE = re.compile(r"^vella\.core\._")
# vella.reconciler is a SIBLING, not a dependency: any import of it (bare or
# submodule) is forbidden — the agent depends on {core, runtime, graph} only.
_RECONCILER = re.compile(r"^vella\.reconciler(\.|$)")


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


def test_no_private_graph_imports() -> None:
    """No src module imports a private ``vella.graph._*`` symbol or submodule.

    The agent depends on ``vella.graph`` only through its public surface; a reach
    into a private ``vella.graph._*`` internal trips this gate.
    """
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _PRIVATE_GRAPH.match(alias.name):
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _PRIVATE_GRAPH.match(module):
                    offenders.append(f"{path.name}: from {module} import ...")
    assert not offenders, f"private vella.graph imports: {offenders}"


def test_graph_imports_are_in_the_published_surface() -> None:
    """Every ``from vella.graph`` name is one of the published ``__all__`` symbols."""
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "vella.graph":
                for alias in node.names:
                    if alias.name not in _ALLOWED_GRAPH:
                        offenders.append(f"{path.name}: from vella.graph import {alias.name}")
    assert not offenders, f"unpublished vella.graph imports: {offenders}"


def test_no_private_core_imports() -> None:
    """No src module imports a private ``vella.core._*`` symbol or submodule.

    Mirrors :func:`test_no_private_runtime_imports` for the core layer: the agent
    depends on ``vella.core`` only through its public surface. A reach into a
    private ``vella.core._*`` internal trips this gate.
    """
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _PRIVATE_CORE.match(alias.name):
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _PRIVATE_CORE.match(module):
                    offenders.append(f"{path.name}: from {module} import ...")
    assert not offenders, f"private vella.core imports: {offenders}"


def test_no_reconciler_imports() -> None:
    """No src module imports ``vella.reconciler`` — it is a sibling, not a dependency.

    The agent depends DOWNWARD on {core, runtime, graph} only; ``vella.reconciler``
    is off the dependency graph entirely. This is AST-based and never executes the
    import, so the forbid holds even though ``vella-reconciler`` is not installed on
    this branch/worktree — a parse-based walk still sees a stray
    ``import vella.reconciler`` / ``from vella.reconciler import ...``.
    """
    offenders: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _RECONCILER.match(alias.name):
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if _RECONCILER.match(module):
                    offenders.append(f"{path.name}: from {module} import ...")
    assert not offenders, f"forbidden vella.reconciler imports: {offenders}"
