#!/usr/bin/env python
"""
Schema-export breaking-change tripwire.

Two jobs, both wired into ``--check`` (run in CI):
  1. Snapshot the JSON Schema of the core envelope + value objects into
     ``schema/core.json`` and fail on undeclared drift (diff-and-ack — catches
     *accidental* envelope breaks; the envelope has no per-type policy).
  2. Snapshot every registered node/edge type's schema + declared ``compat``
     policy into ``schema/types.json`` and fail any change that violates that
     policy (Confluent semantics). Integration packages import their types, then
     run this script, to gate their own evolution.

The compatibility checker resolves ``$ref``/``$defs`` and recurses into nested
objects and array items, and classifies: added/removed/optional<->required
fields, type changes, enum narrowing/widening, and format changes.

Usage:
    python scripts/export_schema.py            # (re)write baselines
    python scripts/export_schema.py --check     # fail on undeclared drift / incompat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vella.core import (  # noqa: E402
    Actuator,
    Embedding,
    FlexibleEdge,
    FlexibleNode,
    HistoryEntry,
    IntegrationBinding,
    NodeFlags,
    Overlay,
    Provenance,
    ResolvedRef,
    ToolDeclaration,
    ToolOverride,
    UnresolvedRef,
    default_registry,
)

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "schema" / "core.json"
TYPES_BASELINE = ROOT / "schema" / "types.json"

CORE_MODELS: dict[str, Any] = {
    "Node": FlexibleNode,
    "Edge": FlexibleEdge,
    "Overlay": Overlay,
    "Actuator": Actuator,
    "Provenance": Provenance,
    "UnresolvedRef": UnresolvedRef,
    "ResolvedRef": ResolvedRef,
    "Embedding": Embedding,
    "IntegrationBinding": IntegrationBinding,
    "NodeFlags": NodeFlags,
    "HistoryEntry": HistoryEntry,
    "ToolDeclaration": ToolDeclaration,
    "ToolOverride": ToolOverride,
}


def generate() -> dict[str, Any]:
    return {name: model.model_json_schema() for name, model in CORE_MODELS.items()}


def registry_type_schemas(registry: Any = None) -> dict[str, Any]:
    """
    Schema + declared compat policy for every registered type with a data class.

    Core itself registers no concrete node types, so in *this* repo's CI this is
    empty and the active protection is the envelope diff-and-ack above. The
    per-type compat gate activates in integration packages: they import their
    types (populating ``default_registry``) and run this script.
    """
    reg = registry if registry is not None else default_registry
    out: dict[str, Any] = {}
    for name in reg.names():
        spec = reg.get(name)
        if spec is None or spec.data_cls is object:
            continue
        out[name] = {
            "compat": spec.compat,
            "version": spec.version,
            "schema": spec.data_cls.model_json_schema(),
        }
    return out


# --- Compatibility checker (Confluent semantics, recursive, $ref-aware) -------
# A change "kind" breaks BACKWARD (new reader can't read old data), FORWARD (old
# reader can't read new data), or both. ``check_compat`` filters by the policy.
_BACKWARD_BREAKING = {
    "added_required", "became_required", "type_changed", "format_changed",
    "enum_value_removed", "union_narrowed", "null_removed", "extra_closed",
    "closed_field_removed",
}
_FORWARD_BREAKING = {
    "removed_required", "became_optional", "type_changed", "format_changed",
    "enum_value_added", "union_widened", "null_added", "extra_opened",
    "closed_field_added",
}


def _deref(node: Any, defs: dict[str, Any]) -> dict[str, Any]:
    seen: set[str] = set()
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if ref in seen:
            break
        seen.add(ref)
        node = defs.get(ref.split("/")[-1], {})
    return node if isinstance(node, dict) else {}


def _split_nullable(schema: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Split an anyOf/oneOf into (non-null arms, nullable?). pydantic emits
    ``anyOf: [{...}, {"type": "null"}]`` for every ``Optional[...]`` field."""
    for key in ("anyOf", "oneOf"):
        arms = schema.get(key)
        if isinstance(arms, list):
            non_null = [a for a in arms if isinstance(a, dict) and a.get("type") != "null"]
            nullable = any(isinstance(a, dict) and a.get("type") == "null" for a in arms)
            return non_null, nullable
    return [schema], False


def _arm_key(arm: dict[str, Any]) -> Any:
    """Fingerprint a union arm: $ref if present, else (type, format) so that e.g.
    ``str`` and ``bytes`` (both type=string, differing format) are distinguished."""
    return arm.get("$ref") or (arm.get("type"), arm.get("format"))


def _value_set(schema: dict[str, Any]) -> Optional[set[str]]:
    """The allowed scalar values of an enum/const arm, normalized so that a
    ``const`` and a single-member ``enum`` compare equal. pydantic emits ``enum``
    for ``Literal[a, b]`` and ``const`` for ``Literal[a]`` — narrowing a Literal
    to one member is therefore an enum→const transition that must still be seen."""
    if "enum" in schema:
        return {json.dumps(x, sort_keys=True) for x in schema["enum"]}
    if "const" in schema:
        return {json.dumps(schema["const"], sort_keys=True)}
    return None


def _collect(
    old: dict[str, Any], new: dict[str, Any],
    old_defs: dict[str, Any], new_defs: dict[str, Any],
    path: str, acc: list[tuple[str, str]], seen: set[tuple[Any, Any]],
) -> None:
    # Recursion guard: a self-referential type (TreeNode.children -> TreeNode)
    # would recurse forever. Comparing the same ($ref, $ref) pair twice yields
    # the same verdict, so record-and-skip is both safe and termination-ensuring.
    o_ref = old.get("$ref") if isinstance(old, dict) else None
    n_ref = new.get("$ref") if isinstance(new, dict) else None
    if o_ref is not None or n_ref is not None:
        pair = (o_ref, n_ref)
        if pair in seen:
            return
        seen.add(pair)
    o_pre, n_pre = old, new  # pre-deref, so a lone $ref keeps its ref identity
    old, new = _deref(old, old_defs), _deref(new, new_defs)
    here = path or "."

    # Normalize Optional / unions (anyOf/oneOf) and detect nullability changes.
    o_arms, o_null = _split_nullable(old)
    n_arms, n_null = _split_nullable(new)
    if o_null and not n_null:
        acc.append((here, "null_removed"))   # old data may be null; new rejects it
    if n_null and not o_null:
        acc.append((here, "null_added"))      # new data may be null; old reader rejects it
    if (len(o_arms) == 1 and len(n_arms) == 1) and (o_arms[0] is not old or n_arms[0] is not new):
        _collect(o_arms[0], n_arms[0], old_defs, new_defs, path, acc, seen)
        return
    if len(o_arms) > 1 or len(n_arms) > 1:  # true union: compare arm fingerprints
        # Fingerprint from the PRE-deref arms: a lone `$ref A` widening to
        # `Union[A, B]` must compare ref-vs-ref ({A} ⊂ {A, B} → widened), not the
        # deref'd ('object', None) which would never subset and mis-flag type_changed.
        ok = {_arm_key(a) for a in _split_nullable(o_pre)[0]}
        nk = {_arm_key(a) for a in _split_nullable(n_pre)[0]}
        if ok != nk:
            # Directional, like enum: widening (added arms) only breaks FORWARD (old
            # reader rejects the new arm); narrowing (removed arms) only breaks
            # BACKWARD (new reader rejects old data's arm); a mixed change breaks both.
            if ok < nk:
                acc.append((here, "union_widened"))
            elif nk < ok:
                acc.append((here, "union_narrowed"))
            else:
                acc.append((here, "type_changed"))
        return

    ot, nt = old.get("type"), new.get("type")
    if ot is not None and nt is not None and ot != nt:
        acc.append((here, "type_changed"))
        return

    of_, nf = old.get("format"), new.get("format")
    if of_ != nf and (of_ or nf):
        acc.append((here, "format_changed"))

    # enum/const allowed-value set (const == single-member enum). Removing a value
    # breaks backward (new reader rejects old data); adding breaks forward.
    ov, nv = _value_set(old), _value_set(new)
    if ov is not None and nv is not None:
        if ov - nv:
            acc.append((here, "enum_value_removed"))
        if nv - ov:
            acc.append((here, "enum_value_added"))

    op, np_ = old.get("properties"), new.get("properties")
    if isinstance(op, dict) and isinstance(np_, dict):
        oreq, nreq = set(old.get("required", [])), set(new.get("required", []))
        # Under a closed content model (extra="forbid" → additionalProperties:false)
        # the peer reader rejects ANY unknown field, so add/remove is breaking even
        # for optional fields: adding one breaks FORWARD (old closed reader rejects
        # it), removing one breaks BACKWARD (new closed reader rejects old data's field).
        o_closed = old.get("additionalProperties") is False
        n_closed = new.get("additionalProperties") is False
        for f in op.keys() & np_.keys():
            sub = f"{path}.{f}" if path else f
            _collect(op[f], np_[f], old_defs, new_defs, sub, acc, seen)
            if f in nreq and f not in oreq:
                acc.append((sub, "became_required"))
            if f in oreq and f not in nreq:
                acc.append((sub, "became_optional"))
        for f in np_.keys() - op.keys():
            loc = f"{path}.{f}" if path else f
            if f in nreq:
                acc.append((loc, "added_required"))
            if o_closed:
                acc.append((loc, "closed_field_added"))
        for f in op.keys() - np_.keys():
            loc = f"{path}.{f}" if path else f
            if f in oreq:
                acc.append((loc, "removed_required"))
            if n_closed:
                acc.append((loc, "closed_field_removed"))

    # additionalProperties (open dicts / Dict[str, X]); also the extra=allow/forbid
    # gate: pydantic emits `false` for extra="forbid", and absent defaults to open.
    oa, na = old.get("additionalProperties"), new.get("additionalProperties")
    if isinstance(oa, dict) and isinstance(na, dict):
        _collect(oa, na, old_defs, new_defs, f"{path}{{}}", acc, seen)
    o_closed, n_closed = (oa is False), (na is False)
    if n_closed and not o_closed:
        acc.append((here, "extra_closed"))   # new reader forbids extras old data may carry
    if o_closed and not n_closed:
        acc.append((here, "extra_opened"))    # new data may carry extras old reader forbids

    # array items, and tuple-form prefixItems
    oi, ni = old.get("items"), new.get("items")
    if isinstance(oi, dict) and isinstance(ni, dict):
        _collect(oi, ni, old_defs, new_defs, f"{path}[]", acc, seen)
    op_, np2 = old.get("prefixItems"), new.get("prefixItems")
    if isinstance(op_, list) and isinstance(np2, list):
        for idx in range(min(len(op_), len(np2))):
            _collect(op_[idx], np2[idx], old_defs, new_defs, f"{path}[{idx}]", acc, seen)
        if len(op_) != len(np2):
            acc.append((here, "type_changed"))  # tuple arity change
    # tuple<->list switch: both are arrays (so the type==type check above passed),
    # but the item shape moved between prefixItems (fixed tuple) and items (list /
    # variadic tuple). Neither branch above fires, yet it is a breaking change.
    if old.get("type") == "array" and new.get("type") == "array":
        if ("prefixItems" in old) != ("prefixItems" in new):
            acc.append((here, "type_changed"))


def check_compat(old: dict[str, Any], new: dict[str, Any], policy: str) -> list[str]:
    """Return a list of compatibility violations for ``old -> new`` under ``policy``."""
    if policy == "NONE":
        return []
    base = policy.replace("_TRANSITIVE", "")
    breaking: set[str] = set()
    if base in ("BACKWARD", "FULL"):
        breaking |= _BACKWARD_BREAKING
    if base in ("FORWARD", "FULL"):
        breaking |= _FORWARD_BREAKING

    changes: list[tuple[str, str]] = []
    _collect(old, new, old.get("$defs", {}), new.get("$defs", {}), "", changes, set())
    return [f"{loc}: {kind}" for loc, kind in changes if kind in breaking]


def _serialize(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on undeclared drift / incompat")
    args = parser.parse_args()

    core = generate()
    types = registry_type_schemas()

    if not args.check:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(_serialize(core))
        TYPES_BASELINE.write_text(_serialize(types))
        print(f"Wrote baselines: core ({len(core)} models), types ({len(types)} registered)")
        return 0

    failed = False

    # (1) Envelope: diff-and-ack.
    if not BASELINE.exists():
        print("No core baseline. Run without --check to create schema/core.json.", file=sys.stderr)
        return 1
    committed_core = json.loads(BASELINE.read_text())
    drift = [n for n in set(core) | set(committed_core)
             if json.dumps(core.get(n), sort_keys=True) != json.dumps(committed_core.get(n), sort_keys=True)]
    if drift:
        print("Core envelope schema drift:", *(f"  - {n}" for n in sorted(drift)), sep="\n", file=sys.stderr)
        print("If intentional, re-run scripts/export_schema.py and commit schema/core.json.", file=sys.stderr)
        failed = True

    # (2) Registered types: enforce each declared compat policy.
    committed_types = json.loads(TYPES_BASELINE.read_text()) if TYPES_BASELINE.exists() else {}
    for name, entry in types.items():
        prior = committed_types.get(name)
        if prior is None:
            continue  # new type: additive
        violations = check_compat(prior["schema"], entry["schema"], entry["compat"])
        if violations:
            failed = True
            print(f"Incompatible change to {name!r} (policy {entry['compat']}):", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)
            print("  Bump schema_version + register a migration, then re-run the exporter.", file=sys.stderr)

    if failed:
        return 1
    print(f"Schema check passed (core: {len(core)} models, types: {len(types)} registered).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
