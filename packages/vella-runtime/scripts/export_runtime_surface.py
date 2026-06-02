#!/usr/bin/env python
"""
Runtime public-surface breaking-change tripwire.

Snapshots the runtime's public surface — the ``__all__`` export list plus, for
each exported error type, its name and qualified base classes — into
``schema/runtime_surface.json`` and fails ``--check`` on undeclared drift
(diff-and-ack). This is the M1 stub: it locks the *shape* of the surface
(what is exported and the error hierarchy) so an accidental removal or
re-parenting trips the gate. M5 fleshes it out to snapshot full model/JSON
schemas as the surface grows.

Usage:
    python scripts/export_runtime_surface.py            # (re)write the baseline
    python scripts/export_runtime_surface.py --check     # fail on undeclared drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vella.runtime as runtime  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "schema" / "runtime_surface.json"


def generate() -> dict[str, Any]:
    """Build the deterministic public-surface snapshot.

    ``__all__`` is sorted (set-derived ordering must never leak into a
    serialized artifact), and each exported symbol that is an exception class
    contributes its name and the sorted qualified names of its base classes —
    enough to catch a removed export or a re-parented error type.
    """
    exported = sorted(runtime.__all__)
    errors: dict[str, list[str]] = {}
    for name in exported:
        obj = getattr(runtime, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            errors[name] = sorted(
                f"{base.__module__}.{base.__qualname__}"
                for base in obj.__mro__
                if base is not obj
            )
    return {"__all__": exported, "errors": errors}


def _serialize(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write or check the runtime-surface baseline; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail on undeclared surface drift"
    )
    args = parser.parse_args()

    surface = generate()

    if not args.check:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(_serialize(surface))
        print(f"Wrote baseline: {len(surface['__all__'])} public symbols")
        return 0

    if not BASELINE.exists():
        print(
            "No surface baseline. Run without --check to create "
            "schema/runtime_surface.json.",
            file=sys.stderr,
        )
        return 1

    committed = json.loads(BASELINE.read_text())
    if _serialize(surface) != _serialize(committed):
        print("Runtime public-surface drift:", file=sys.stderr)
        print(f"  committed: {json.dumps(committed, sort_keys=True)}", file=sys.stderr)
        print(f"  current:   {json.dumps(surface, sort_keys=True)}", file=sys.stderr)
        print(
            "If intentional, re-run scripts/export_runtime_surface.py and commit "
            "schema/runtime_surface.json.",
            file=sys.stderr,
        )
        return 1

    print(f"Surface check passed ({len(surface['__all__'])} public symbols).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
