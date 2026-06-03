"""Public-surface tripwire test.

Asserts the live ``vella.reconciler.__all__`` equals the frozen baseline
(including the public ``ManualClock`` testing seam), and that the committed
``schema/reconciler_surface.json`` baseline is in sync via the export script's
``--check`` path. A new/removed/renamed export trips one of these.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import vella.reconciler as reconciler

_FROZEN_ALL = (
    "Clock",
    "Context",
    "CursorStore",
    "DeadLetterRecord",
    "DeadLetterStore",
    "InMemoryCursorStore",
    "InMemoryDeadLetterStore",
    "ManualClock",
    "ReconcileResult",
    "Reconciler",
    "Registry",
)

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "export_reconciler_surface.py"
)


def _load_export_module() -> object:
    spec = importlib.util.spec_from_file_location("_export_reconciler_surface", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_matches_frozen_baseline() -> None:
    assert tuple(sorted(reconciler.__all__)) == _FROZEN_ALL
    assert "ManualClock" in reconciler.__all__


def test_surface_baseline_in_sync() -> None:
    export = _load_export_module()
    surface = export.generate()  # type: ignore[attr-defined]
    committed = export.json.loads(  # type: ignore[attr-defined]
        export.BASELINE.read_text()  # type: ignore[attr-defined]
    )
    assert export._serialize(surface) == export._serialize(committed)  # type: ignore[attr-defined]
