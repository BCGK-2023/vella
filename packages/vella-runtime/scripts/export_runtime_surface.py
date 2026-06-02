#!/usr/bin/env python
"""
Runtime public-surface breaking-change tripwire.

Snapshots the runtime's public surface into ``schema/runtime_surface.json`` and
fails ``--check`` on undeclared drift (diff-and-ack). It captures:

* ``__all__`` — the sorted export list (a removed export trips the gate).
* ``errors`` — each exported exception type's sorted qualified base classes (a
  re-parented error trips the gate).
* ``models`` — each exported ``BaseModel``'s field name -> JSON-schema ``type``
  (so e.g. ``Cursor`` freezes as ``{"token": {"type": "string"}}``; a field
  rename, removal, or retype trips the gate without freezing the Python type).
* ``literals`` — each exported ``Literal`` alias's sorted allowed values (so the
  ``TransitionKind`` set is frozen against silent additions/removals).

Everything is emitted with ``sort_keys=True`` — set-derived ordering must never
leak into the serialized artifact.

Usage:
    python scripts/export_runtime_surface.py            # (re)write the baseline
    python scripts/export_runtime_surface.py --check     # fail on undeclared drift
"""

from __future__ import annotations

import argparse
import json
import sys
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vella.runtime as runtime  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "schema" / "runtime_surface.json"


def _field_types(model: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """Field name -> its JSON-schema ``type`` entry (the stable, portable shape).

    Reads the model's JSON schema and keeps only each property's ``type`` (or, for
    composite shapes like optionals/unions, the ``anyOf``) so the snapshot freezes
    the wire contract without freezing the underlying Python type.
    """
    schema = model.model_json_schema()
    props = schema.get("properties", {})
    out: dict[str, dict[str, Any]] = {}
    for field_name, prop in props.items():
        if "type" in prop:
            out[field_name] = {"type": prop["type"]}
        elif "anyOf" in prop:
            out[field_name] = {"anyOf": prop["anyOf"]}
        elif "$ref" in prop:
            out[field_name] = {"$ref": prop["$ref"]}
        else:
            out[field_name] = {}
    return out


def generate() -> dict[str, Any]:
    """Build the deterministic public-surface snapshot.

    ``__all__`` is sorted (set-derived ordering must never leak into a serialized
    artifact); each exported exception contributes its sorted base classes; each
    exported ``BaseModel`` contributes its field-type map; and each exported
    ``Literal`` alias contributes its sorted allowed values.
    """
    exported = sorted(runtime.__all__)
    errors: dict[str, list[str]] = {}
    models: dict[str, dict[str, dict[str, Any]]] = {}
    literals: dict[str, list[Any]] = {}
    for name in exported:
        obj = getattr(runtime, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            errors[name] = sorted(
                f"{base.__module__}.{base.__qualname__}"
                for base in obj.__mro__
                if base is not obj
            )
        elif isinstance(obj, type) and issubclass(obj, BaseModel):
            models[name] = _field_types(obj)
        elif typing.get_origin(obj) is typing.Literal:
            literals[name] = sorted(typing.get_args(obj))
    return {
        "__all__": exported,
        "errors": errors,
        "models": models,
        "literals": literals,
    }


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
