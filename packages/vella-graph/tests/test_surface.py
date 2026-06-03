"""Public-surface tripwire test.

Asserts the live ``vella.graph.__all__`` equals the frozen baseline (empty at M1,
grows milestone by milestone), and that the committed ``schema/graph_surface.json``
baseline is in sync via the export script's ``--check`` path. A new/removed/renamed
export trips one of these.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import vella.graph as graph

# Grows as each milestone adds to vella.graph.__all__ (M2 added the fold builder,
# the frozen view, and the materialization mode; M3 adds the frozen result models).
_FROZEN_ALL: tuple[str, ...] = (
    "GraphProjection",
    "GraphView",
    "MaterializationMode",
    "Neighbor",
    "Path",
)

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "export_graph_surface.py"


def _load_export_module() -> object:
    spec = importlib.util.spec_from_file_location("_export_graph_surface", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_matches_frozen_baseline() -> None:
    assert tuple(sorted(graph.__all__)) == _FROZEN_ALL


def test_surface_baseline_in_sync() -> None:
    export = _load_export_module()
    surface = export.generate()  # type: ignore[attr-defined]
    committed = export.json.loads(  # type: ignore[attr-defined]
        export.BASELINE.read_text()  # type: ignore[attr-defined]
    )
    assert export._serialize(surface) == export._serialize(committed)  # type: ignore[attr-defined]
