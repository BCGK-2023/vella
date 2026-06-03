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

import re
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
