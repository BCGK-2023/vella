"""
Polymorphic round-trip — turning a raw stored/wire node back into a typed Node.

``Node`` is generic, and generics are erased at runtime: ``Node.model_validate``
has no way to know ``data`` should be ``OutlookEmailData`` and would silently
parse it into an empty ``BaseModel``, dropping every field. ``parse_node`` closes
that gap using the registry (the open-tagged-union / Expression-Problem solution):
read ``type`` → look up the data and state classes → validate into the right
``Node[Data, State]``.

Tolerant reader (Postel):
  * Unknown *envelope*-level fields (a newer writer added one) are **ignored**.
  * Unknown *types* resolve to ``FlexibleData`` (``strict=True`` to demand a
    registered type).
  * Schema drift is repaired by version-chained migrations; anything that still
    fails validation is **quarantined** into a ``FlexibleData`` node carrying a
    ``vella_repair`` marker rather than throwing.

Quarantine is guaranteed not to throw: every strict sub-surface (state, tools,
integrations, embedding, flags) is stripped into the marker, and a last-resort
synthetic node is produced if even the minimal envelope is malformed. The repair
``reason`` is built from validation error *types/locations only* — never the raw
input values — so quarantine markers do not persist (possibly sensitive) payloads.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, TypeVar, cast

from .base import FlexibleData, VellaModel
from .edge import Edge
from ._uuid7 import uuid7
from .errors import SchemaMigrationError, UnregisteredTypeError
from .node import Node
from .references import UnresolvedRef
from .tooling import Registry, TypeSpec, default_registry

FlexibleNode = Node[FlexibleData]
FlexibleEdge = Edge[FlexibleData]

_REPAIR_KEY = "vella_repair"
_NODE_FIELDS = frozenset(Node.model_fields)
_EDGE_FIELDS = frozenset(Edge.model_fields)
# Envelope sub-surfaces that are strictly validated and can independently fail;
# stripped during quarantine so it can never throw.
_STRICT_SUBSURFACES = ("state", "tool_overrides", "extra_tools", "integrations", "embedding", "flags")


def _envelope_only(raw: Mapping[str, Any], fields: frozenset[str]) -> dict[str, Any]:
    """Keep only known envelope fields — tolerant reader ignores unknown ones."""
    return {k: v for k, v in raw.items() if k in fields}


_M = TypeVar("_M", bound=VellaModel)


def _set_registry(obj: _M, reg: Registry) -> _M:
    """Carry the resolving registry on the instance so evolve/model_copy reuse it."""
    obj._vella_registry = reg  # pyright: ignore[reportPrivateUsage]  # intra-package plumbing
    return obj


def _safe_reason(exc: Exception) -> str:
    """A reason string built from error types/locations only — never input values."""
    errors_fn = getattr(exc, "errors", None)
    if errors_fn is not None:
        try:
            details = cast("list[dict[str, Any]]", errors_fn())
            parts = [
                f"{'.'.join(str(p) for p in e.get('loc', ()))}:{e.get('type', '?')}"
                for e in details
            ]
            return f"{type(exc).__name__}: " + "; ".join(parts)
        except Exception:
            pass
    return type(exc).__name__


def _migrate_data(data: dict[str, Any], spec: TypeSpec, writer_version: int) -> dict[str, Any]:
    """Apply the registered migration chain from writer_version up to spec.version."""
    version = writer_version
    while version < spec.version:
        step = spec.migrations.get(version)
        if step is None:
            raise SchemaMigrationError(
                f"No migration registered for type {spec.name!r} from schema "
                f"version {version} to {version + 1}.",
                type_name=spec.name,
                from_version=writer_version,
                to_version=spec.version,
            )
        data = step(data)
        version += 1
    return data


def _as_version(value: Any) -> int:
    """Coerce a writer's schema_version to int, tolerating garbage (→ 1)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _str_keys(mapping: Any) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """A copy with top-level keys coerced to ``str`` (pydantic requires field
    names to be strings; a ``data`` mapping from a binary codec can carry int/
    None/tuple keys, which would otherwise make tolerant parsing throw).

    Returns ``(result, collisions)``. When two distinct keys stringify to the
    same name (e.g. ``{1: "a", "1": "b"}``) only the last can survive as a field,
    so every value that landed on a contested name is recorded in ``collisions``
    — nothing is silently dropped (the data-loss invariant)."""
    out: dict[str, Any] = {}
    grouped: dict[str, list[Any]] = {}
    for k, v in cast("Mapping[Any, Any]", mapping).items():
        sk = str(k)
        grouped.setdefault(sk, []).append(v)
        out[sk] = v
    collisions = {sk: vals for sk, vals in grouped.items() if len(vals) > 1}
    return out, collisions


# A repair marker is exactly one of our own dicts: it has reason + schema_version
# and NO other keys beyond the ones we write. Real user data that merely happens
# to contain a "reason" key therefore is NOT mistaken for a marker.
_MARKER_KEYS = {"reason", "schema_version", "shadowed", "stripped", "shadowed_data", "key_collisions"}


def _is_prior_marker(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    v = cast("Mapping[str, Any]", value)
    return "reason" in v and "schema_version" in v and set(v.keys()) <= _MARKER_KEYS


def _repair_marker(existing: Any, reason: str, writer_version: int) -> dict[str, Any]:
    marker: dict[str, Any] = {"reason": reason[:500], "schema_version": writer_version}
    if _is_prior_marker(existing):
        # Do not nest; carry forward previously-preserved values.
        for carried in ("shadowed", "shadowed_data", "key_collisions"):
            if carried in existing:
                marker[carried] = existing[carried]
    elif existing is not None:
        marker["shadowed"] = existing  # preserve a real clobbered user value
    return marker


def _quarantine_payload(
    raw: Mapping[str, Any], fields: frozenset[str], reason: str
) -> dict[str, Any]:
    clean = _envelope_only(raw, fields)
    stripped = [k for k in _STRICT_SUBSURFACES if clean.pop(k, None) is not None]
    raw_data = raw.get("data")
    # Top-level keys must be strings to land as FlexibleData fields — a dict from a
    # binary codec (msgpack/BSON/YAML) may carry int/None/tuple keys. Stringify so
    # quarantine can never throw on a non-str key; the value is preserved verbatim.
    data: dict[str, Any]
    collisions: dict[str, list[Any]]
    data, collisions = _str_keys(raw_data) if isinstance(raw_data, Mapping) else ({}, {})
    marker = _repair_marker(data.get(_REPAIR_KEY), reason, _as_version(raw.get("schema_version")))
    if stripped:
        marker["stripped"] = stripped
    # A non-mapping `data` (e.g. a stray string/list) can't hold the marker — preserve it.
    if raw_data is not None and not isinstance(raw_data, Mapping):
        marker["shadowed_data"] = raw_data
    # Keys that collided on stringification keep only their last value as a field;
    # record every contested value so none is silently lost.
    if collisions:
        marker["key_collisions"] = collisions
    data[_REPAIR_KEY] = marker
    clean["data"] = data
    # Coerce malformed identity fields so the FlexibleNode/Edge envelope can still
    # validate (and the full marker — shadowed/shadowed_data — is preserved) rather
    # than falling to the last-resort path. schema_version likewise.
    clean["schema_version"] = _as_version(raw.get("schema_version"))
    clean["type"] = str(clean.get("type") or "unparseable")
    is_node = "from_node_id" not in fields
    if is_node and not isinstance(clean.get("name"), str):
        clean["name"] = str(clean.get("name") or "unparseable")
    elif "name" in clean and not isinstance(clean["name"], str):
        clean["name"] = str(clean["name"])
    return clean


def _flexible_data(payload: Mapping[str, Any]) -> FlexibleData:
    """The quarantine payload's full `data` (sibling fields + marker) as FlexibleData.
    The last-resort path was reached because an *envelope scalar* failed, not the
    data — so the data dict is FlexibleData-valid and must be preserved whole.
    This is the final backstop for the "never throws" invariant: keys are
    stringified and any residual validation failure falls back to a minimal
    marker node, so it cannot raise regardless of what the body contains."""
    data = payload.get("data")
    if isinstance(data, Mapping):
        # ``payload`` is the already-cleaned quarantine body, so its keys are
        # strings and any collisions were recorded upstream; stringify defensively.
        body, _collisions = _str_keys(data)
        try:
            return FlexibleData.model_validate(body)
        except Exception:
            pass
    return FlexibleData.model_validate({_REPAIR_KEY: {"reason": "unparseable", "schema_version": 1}})


def _last_resort_node(raw: Mapping[str, Any], payload: Mapping[str, Any]) -> "Node[Any, Any]":
    return FlexibleNode(
        type=str(raw.get("type") or "unparseable"),
        name=str(raw.get("name") or "unparseable"),
        created_by=UnresolvedRef(identifier="vella:unparseable"),
        data=_flexible_data(payload),  # preserve the full data body, not just the marker
    )


def _last_resort_edge(raw: Mapping[str, Any], payload: Mapping[str, Any]) -> "Edge[Any, Any]":
    return FlexibleEdge(
        type=str(raw.get("type") or "unparseable"),
        from_node_id=uuid7(),
        to_node_id=uuid7(),
        created_by=UnresolvedRef(identifier="vella:unparseable"),
        data=_flexible_data(payload),
    )


def parse_node(
    raw: Mapping[str, Any], *, registry: Optional[Registry] = None, strict: bool = False
) -> "Node[Any, Any]":
    """
    Reconstruct a typed ``Node`` from a raw mapping.

    * Unknown envelope fields → ignored. Unknown type → ``FlexibleData`` node
      (or ``UnregisteredTypeError`` if strict).
    * Older ``schema_version`` → migrated up the registered chain.
    * Validation failure → quarantined repairable node (or re-raised if strict).
      Quarantine never throws.
    """
    reg = registry or default_registry
    ctx = {"registry": reg}
    type_name = raw.get("type")
    spec = reg.get(type_name) if isinstance(type_name, str) else None

    if spec is None or spec.data_cls is object:
        if strict:
            raise UnregisteredTypeError(type_name if isinstance(type_name, str) else None, reg.names())
        try:
            return _set_registry(FlexibleNode.model_validate(_envelope_only(raw, _NODE_FIELDS), context=ctx), reg)
        except Exception as exc:
            payload = _quarantine_payload(raw, _NODE_FIELDS, _safe_reason(exc))
            return _quarantine_node(payload, raw, reg)

    writer_version = _as_version(raw.get("schema_version"))
    model: type[Node[Any, Any]]
    if spec.state_cls:
        model = Node[spec.data_cls, spec.state_cls]  # type: ignore[name-defined]
    else:
        model = Node[spec.data_cls]  # type: ignore[name-defined]
    try:
        # Migration is inside the try: a missing step (SchemaMigrationError) or a
        # raising user migration must quarantine, not throw — the non-strict
        # "never throws" invariant covers schema drift too (the original, un-migrated
        # raw["data"] is what _quarantine_payload preserves).
        prepared = _envelope_only(raw, _NODE_FIELDS)
        if writer_version < spec.version and isinstance(raw.get("data"), Mapping):
            prepared["data"] = _migrate_data(dict(raw["data"]), spec, writer_version)
            prepared["schema_version"] = spec.version
        return _set_registry(model.model_validate(prepared, context=ctx), reg)
    except Exception as exc:
        if strict:
            raise
        return _quarantine_node(_quarantine_payload(raw, _NODE_FIELDS, _safe_reason(exc)), raw, reg)


def _quarantine_node(payload: dict[str, Any], raw: Mapping[str, Any], reg: Registry) -> "Node[Any, Any]":
    try:
        return _set_registry(FlexibleNode.model_validate(payload, context={"registry": reg}), reg)
    except Exception:
        return _set_registry(_last_resort_node(raw, payload), reg)


def parse_edge(
    raw: Mapping[str, Any], *, registry: Optional[Registry] = None, strict: bool = False
) -> "Edge[Any, Any]":
    """Edge counterpart of ``parse_node`` (see its docstring). Quarantine never throws."""
    reg = registry or default_registry
    ctx = {"registry": reg}
    type_name = raw.get("type")
    spec = reg.get(type_name) if isinstance(type_name, str) else None

    if spec is None or spec.data_cls is object:
        if strict:
            raise UnregisteredTypeError(type_name if isinstance(type_name, str) else None, reg.names())
        try:
            return _set_registry(FlexibleEdge.model_validate(_envelope_only(raw, _EDGE_FIELDS), context=ctx), reg)
        except Exception as exc:
            return _quarantine_edge(_quarantine_payload(raw, _EDGE_FIELDS, _safe_reason(exc)), raw, reg)

    writer_version = _as_version(raw.get("schema_version"))
    model: type[Edge[Any, Any]]
    if spec.state_cls:
        model = Edge[spec.data_cls, spec.state_cls]  # type: ignore[name-defined]
    else:
        model = Edge[spec.data_cls]  # type: ignore[name-defined]
    try:
        # Migration inside the try (see parse_node): drift quarantines, never throws.
        prepared = _envelope_only(raw, _EDGE_FIELDS)
        if writer_version < spec.version and isinstance(raw.get("data"), Mapping):
            prepared["data"] = _migrate_data(dict(raw["data"]), spec, writer_version)
            prepared["schema_version"] = spec.version
        return _set_registry(model.model_validate(prepared, context=ctx), reg)
    except Exception as exc:
        if strict:
            raise
        return _quarantine_edge(_quarantine_payload(raw, _EDGE_FIELDS, _safe_reason(exc)), raw, reg)


def _quarantine_edge(payload: dict[str, Any], raw: Mapping[str, Any], reg: Registry) -> "Edge[Any, Any]":
    try:
        return _set_registry(FlexibleEdge.model_validate(payload, context={"registry": reg}), reg)
    except Exception:
        return _set_registry(_last_resort_edge(raw, payload), reg)


__all__ = ["FlexibleNode", "FlexibleEdge", "parse_node", "parse_edge"]
