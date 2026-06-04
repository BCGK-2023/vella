"""Dependency-hygiene gate.

The agent cognition core ships exactly five runtime dependencies — pydantic,
typing_extensions, vella-core, vella-runtime, and vella-graph — and nothing more.
A stray dependency added to ``[project].dependencies`` (rather than the optional
extras) fails here, keeping the shipped footprint deliberate and depending strictly
downward (graph, runtime, core), never up. The heavy I/O transports live in the
``[openrouter]`` / ``[mcp]`` extras (excluded here on purpose), so importing the
core never pulls ``httpx`` / ``mcp``.

The names are compared with their EXACT spellings from ``[project].dependencies``;
there is no PEP 503 normalization, so a hyphen/underscore mismatch
(``vella_graph`` vs ``vella-graph``) fails the set equality on purpose.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
# Split on the first version/marker/extras delimiter to recover the bare name.
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")

_EXPECTED = frozenset(
    {"pydantic", "typing_extensions", "vella-core", "vella-runtime", "vella-graph"}
)


def _package_names(requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for req in requirements:
        match = _NAME_RE.match(req.strip())
        assert match is not None, f"unparseable requirement: {req!r}"
        names.add(match.group(0))
    return names


def test_agent_dependencies_are_exactly_five() -> None:
    data = tomllib.loads(_PYPROJECT.read_text())
    deps = data["project"]["dependencies"]
    assert _package_names(deps) == _EXPECTED


# --- the heavy I/O extras stay OUT of the published frozenset (and OUT of the core
# import). Two complementary guards: (1) a fresh-import subprocess proves importing
# `vella.agent` pulls neither `httpx` nor `mcp` into `sys.modules`; (2) an AST walk
# proves no GATED-CORE module (everything under src/vella/agent EXCEPT the optional
# `adapters/` package) imports `httpx`/`mcp` at module top — so the adapters' lazy
# imports are the ONLY place those names appear. ---

_SRC = Path(__file__).resolve().parent.parent / "src" / "vella" / "agent"
_HEAVY = frozenset({"httpx", "mcp"})


def test_importing_core_pulls_no_heavy_dep() -> None:
    """A fresh ``import vella.agent`` leaves ``httpx``/``mcp`` out of ``sys.modules``.

    Run in a SUBPROCESS so the check sees a pristine interpreter (an in-process
    import would not reset already-loaded modules). Importing the cognition core —
    including its ``adapters`` package marker — must never import the heavy
    transports; they are pulled only when an adapter is actually constructed.
    """
    code = (
        "import sys, importlib\n"
        "import vella.agent\n"
        "import vella.agent.adapters\n"  # the package marker must stay dep-free too
        "leaked = sorted(m for m in ('httpx', 'mcp') if m in sys.modules)\n"
        "assert not leaked, f'core import leaked heavy deps: {leaked}'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_no_gated_core_module_imports_a_heavy_dep_at_module_top() -> None:
    """No gated-core module imports ``httpx``/``mcp`` at module top (AST-based).

    The optional ``adapters/`` package is the ONE place those names may appear, and
    even there only LAZILY (inside functions). This walks every gated-core ``*.py``
    (everything under ``src/vella/agent`` except ``adapters/``) and asserts no
    module-level ``import httpx`` / ``from mcp import ...`` — the mutation guard for
    "``import httpx`` at the top of a gated-core file ⇒ RED".
    """
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        if "adapters" in path.relative_to(_SRC).parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        # Only MODULE-LEVEL statements (tree.body) — a lazy import nested in a
        # function body is fine and is exactly how the adapters guard their deps.
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _HEAVY:
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in _HEAVY:
                    offenders.append(f"{path.name}: from {node.module} import ...")
    assert not offenders, f"gated-core modules importing heavy deps: {offenders}"
