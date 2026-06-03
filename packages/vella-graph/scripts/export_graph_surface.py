#!/usr/bin/env python
"""
Graph public-surface breaking-change tripwire.

Snapshots the graph projection's public surface into
``schema/graph_surface.json`` and fails ``--check`` on undeclared drift
(diff-and-ack). It captures:

* ``__all__`` — the sorted export list (a removed export trips the gate).
* ``errors`` — each exported exception type's sorted qualified base classes (a
  re-parented error trips the gate).
* ``models`` — each exported ``BaseModel``'s field name -> JSON-schema ``type``
  (so e.g. a future ``GraphView`` result freezes its field shape; a field rename,
  removal, or retype trips the gate without freezing the Python type).
* ``literals`` — each exported ``Literal`` alias's sorted allowed values, AND each
  exported ``BaseModel`` field whose annotation is a ``Literal`` (so e.g. a future
  ``MaterializationMode``'s ``{"full","lean"}`` value set is frozen rather than
  erased to a bare ``"string"`` by the JSON-schema ``type`` — mirrors the runtime
  surface script's handling of ``Literal`` values).
* ``verbs`` — public method signatures of exported coordination classes (the query
  contract — ``neighbors`` / ``bfs`` / ``shortest_path`` and friends as they land,
  plus constructors), rendered via ``inspect.signature`` (deterministic, free of
  the import-order nondeterminism ``typing.get_type_hints`` can introduce) —
  mirrors how the runtime surface script freezes its ``Runtime`` verb signatures.
  At M1 ``__all__`` is empty, so this is ``{}``; it populates as the surface grows.

Field/method-name lists are sorted — the graph sorts its OWN derived structures,
never core/runtime model fields. Everything is emitted with ``sort_keys=True`` —
set-derived ordering must never leak into the serialized artifact.

Usage:
    python scripts/export_graph_surface.py            # (re)write the baseline
    python scripts/export_graph_surface.py --check     # fail on undeclared drift
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vella.graph as graph  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "schema" / "graph_surface.json"


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


def _literal_fields(model: type[BaseModel]) -> dict[str, list[Any]]:
    """Field name -> sorted ``Literal`` value set, for fields annotated ``Literal``.

    The JSON schema erases a ``Literal[...]`` field to a bare ``"string"`` ``type``,
    losing its allowed value set. This reads the model's resolved annotations and,
    for any field whose annotation is a ``Literal``, captures its sorted args — so
    a value set is frozen against silent additions/removals (mirrors the runtime
    script's ``Literal`` handling).
    """
    out: dict[str, list[Any]] = {}
    for field_name, field in model.model_fields.items():
        annotation = field.annotation
        if typing.get_origin(annotation) is typing.Literal:
            out[field_name] = sorted(typing.get_args(annotation))
    return out


def _method_signatures(cls: type) -> dict[str, str]:
    """Public method name -> stable ``inspect.signature`` string for ``cls``.

    Captures the constructor as ``__init__`` (its params are part of the frozen
    surface) plus every public (non-dunder, non-``_``-prefixed) method DECLARED ON
    ``cls`` ITSELF (``vars(cls)`` — never an inherited one). Restricting to
    own-declared methods is what lets this freeze the query/verb contract of a
    ``BaseModel`` subclass (``GraphView.neighbors`` / ``refresh`` / ...) without
    dragging in pydantic's inherited ``model_*`` machinery, whose set is
    pydantic-version-sensitive. Annotations render via ``inspect.signature``'s own
    string form — deterministic and free of the import-order nondeterminism
    ``typing.get_type_hints`` can introduce. Mirrors the runtime surface script.
    """
    out: dict[str, str] = {"__init__": str(inspect.signature(cls.__init__))}
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(member) or isinstance(member, (classmethod, staticmethod))):
            continue
        out[name] = str(inspect.signature(getattr(cls, name)))
    return out


def generate() -> dict[str, Any]:
    """Build the deterministic public-surface snapshot.

    ``__all__`` is sorted (set-derived ordering must never leak into a serialized
    artifact); each exported exception contributes its sorted base classes; each
    exported ``BaseModel`` contributes its field-type map (and any ``Literal``
    field's value set); each exported ``Literal`` alias contributes its sorted
    allowed values; and each exported class contributes its public method
    signatures. At M1 ``__all__`` is empty, so every section is empty.
    """
    exported = sorted(graph.__all__)
    errors: dict[str, list[str]] = {}
    models: dict[str, dict[str, dict[str, Any]]] = {}
    literals: dict[str, list[Any]] = {}
    verbs: dict[str, dict[str, str]] = {}
    for name in exported:
        obj = getattr(graph, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            errors[name] = sorted(
                f"{base.__module__}.{base.__qualname__}"
                for base in obj.__mro__
                if base is not obj
            )
        elif isinstance(obj, type) and issubclass(obj, BaseModel):
            models[name] = _field_types(obj)
            # Freeze any Literal-typed field's value set under "<Model>.<field>"
            # (the JSON-schema "type" erases it to a bare "string").
            for field_name, values in _literal_fields(obj).items():
                literals[f"{name}.{field_name}"] = values
            # A BaseModel that also declares public methods (e.g. GraphView's query
            # surface incl. refresh) freezes those own-declared signatures too — the
            # JSON-schema/field view captures only data, not the verb contract.
            own_verbs = {
                k: v
                for k, v in _method_signatures(obj).items()
                if k != "__init__" and not k.startswith("model_")
            }
            if own_verbs:
                verbs[name] = own_verbs
        elif typing.get_origin(obj) is typing.Literal:
            literals[name] = sorted(typing.get_args(obj))
        elif isinstance(obj, type):
            # A coordination/result class without a model/error base: freeze its
            # public method signatures (the query contract as the surface grows).
            verbs[name] = _method_signatures(obj)
    return {
        "__all__": exported,
        "errors": errors,
        "models": models,
        "literals": literals,
        "verbs": verbs,
    }


def _serialize(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write or check the graph-surface baseline; return a process exit code."""
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
            "schema/graph_surface.json.",
            file=sys.stderr,
        )
        return 1

    committed = json.loads(BASELINE.read_text())
    if _serialize(surface) != _serialize(committed):
        print("Graph public-surface drift:", file=sys.stderr)
        print(f"  committed: {json.dumps(committed, sort_keys=True)}", file=sys.stderr)
        print(f"  current:   {json.dumps(surface, sort_keys=True)}", file=sys.stderr)
        print(
            "If intentional, re-run scripts/export_graph_surface.py and commit "
            "schema/graph_surface.json.",
            file=sys.stderr,
        )
        return 1

    print(f"Surface check passed ({len(surface['__all__'])} public symbols).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
