"""Subprocess fixture for the graph determinism artifact (M3, must-fix / pre-mortem (d)).

Run as a script under a fixed ``PYTHONHASHSEED`` (set by the parent test). It folds
a FIXED multi-tenant graph scenario into a :class:`~vella.graph.GraphView` per
tenant, then serializes a SET-derived topology projection — the sorted neighbour-id
set per ``(node, direction, edge_type)`` across every tenant — to canonical,
byte-stable JSON and prints it to stdout.

The NAMED determinism artifact is that **sorted topology projection** serialized
byte-identically across hash seeds. This mirrors core's discipline: any set-derived
serialized value is ``sorted()``. The endpoint ids are collected into a genuine
``set`` (``set`` iteration over string-bearing / UUID elements is
``PYTHONHASHSEED``-sensitive), so the ONLY thing that makes the serialization
reproducible is the explicit ``sorted()`` here.

Reproducibility design — every source of nondeterminism EXCEPT iteration order is
pinned:

* **Fixed ids / tenants.** The scenario uses explicit ``UUID`` and tenant-id
  constants (no ``uuid4``), so the topology never depends on a random id. The
  (tenant, id) pairs are chosen so their SORTED order ``sorted(key=(tenant,
  str(id)))`` differs from their tuple-insertion / hash-iteration order — that is
  what makes removing ``sorted()`` genuinely diverge (the non-vacuity mutation the
  verifier runs).
* **Volatile keys scrubbed.** The topology projection carries no wall-clock fields,
  but the helper scrubs the same ``_VOLATILE_KEYS`` the runtime/reconciler helpers
  use, defensively and consistently, so a future volatile field cannot silently
  break determinism.
* **Sorted set-derived output is the thing under test.** Each ``(node, direction,
  edge_type)`` bucket's endpoint ids are gathered into a ``set`` then emitted via
  ``sorted(set, ...)`` (UUIDs sorted via ``str``), and the buckets themselves are
  emitted in sorted ``(tenant, node, direction, edge_type)`` order, then
  ``json.dumps(..., sort_keys=True, separators=(",", ":"))``. If any serialized
  value derived its order from set hash iteration, two seeds would diverge.

This is a script, not a test module — invoked via ``subprocess.run`` so each run
gets a fresh interpreter with the parent-supplied hash seed. ``PYTHONHASHSEED`` is
read once at interpreter start, so an in-process re-import would NOT reset it; a
subprocess is the only sound way to vary it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID

from vella.core import EdgeTypes, Node, Registry as CoreRegistry, VellaModel, node_type
from vella.runtime import Runtime

from vella.graph import GraphProjection

# --- pinned scenario constants (no uuid4, no random tenant) ------------------
# >= 3 tenants, >= 6 (tenant, node) entries. The pairs are deliberately NOT in
# (tenant, str(id)) order: their tuple-insertion order below differs from their
# SORTED order, and set iteration over the string-bearing endpoint ids is genuinely
# hash-seed sensitive — so the explicit sorted() is load-bearing. (Mirrors the
# reconciler's _SCENARIO discipline.)
#
# Insertion order here is t-gamma, t-alpha, t-beta, t-alpha, t-gamma, t-beta with
# high-then-low ids; sorted(key=(tenant, str(id))) reorders to t-alpha(4..,9..),
# t-beta(1..,f..), t-gamma(2..,d..) — provably != insertion order.
_NODES: tuple[tuple[str, UUID], ...] = (
    ("t-gamma", UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")),
    ("t-alpha", UUID("99999999-9999-9999-9999-999999999999")),
    ("t-beta", UUID("11111111-1111-1111-1111-111111111111")),
    ("t-alpha", UUID("44444444-4444-4444-4444-444444444444")),
    ("t-gamma", UUID("22222222-2222-2222-2222-222222222222")),
    ("t-beta", UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")),
)

# Extra fan-out targets (also pinned ids) so at least one (node, direction,
# edge_type) bucket holds a set of >= 2 DISTINCT endpoints — that multi-element set
# is what makes the artifact's per-bucket sorted() non-vacuous: removing it lets the
# endpoint order follow hash-seed-driven set iteration and the seeds diverge.
_T_GAMMA_FANOUT = (
    UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"),
    UUID("33333333-3333-3333-3333-333333333333"),
)

# Edges within each tenant, exercising multiple edge types (so multiple buckets per
# node) AND multi-endpoint fan-out (so a bucket's endpoint set has >= 2 members).
_EDGES: tuple[tuple[str, UUID, UUID, str], ...] = (
    # t-gamma: d -KNOWS-> {2, e, 3} — ONE bucket, THREE distinct endpoints. Their
    # sorted order ("2.." < "3.." < "e..") differs from this insertion order
    # (2, e, 3), so the per-bucket sorted() is load-bearing and provably non-vacuous.
    ("t-gamma",
     UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
     UUID("22222222-2222-2222-2222-222222222222"),
     EdgeTypes.KNOWS),
    ("t-gamma",
     UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
     _T_GAMMA_FANOUT[0],
     EdgeTypes.KNOWS),
    ("t-gamma",
     UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
     _T_GAMMA_FANOUT[1],
     EdgeTypes.KNOWS),
    ("t-gamma",
     UUID("22222222-2222-2222-2222-222222222222"),
     UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
     EdgeTypes.OWNED_BY),
    # t-alpha: 9 -> 4 (KNOWS) and 9 -> 4 (PART_OF) — two types, same endpoint.
    ("t-alpha",
     UUID("99999999-9999-9999-9999-999999999999"),
     UUID("44444444-4444-4444-4444-444444444444"),
     EdgeTypes.KNOWS),
    ("t-alpha",
     UUID("99999999-9999-9999-9999-999999999999"),
     UUID("44444444-4444-4444-4444-444444444444"),
     EdgeTypes.PART_OF),
    # t-beta: 1 -> f and f -> 1 (KNOWS both ways).
    ("t-beta",
     UUID("11111111-1111-1111-1111-111111111111"),
     UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
     EdgeTypes.KNOWS),
    ("t-beta",
     UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
     UUID("11111111-1111-1111-1111-111111111111"),
     EdgeTypes.KNOWS),
)

_TENANTS: tuple[str, ...] = ("t-gamma", "t-alpha", "t-beta")

# Mirror runtime/reconciler helpers: scrub per-run wall-clock keys before serializing
# so the only surviving variable is iteration order. The topology projection has none
# today; the scrub is defensive + consistent with the sibling artifacts.
_VOLATILE_KEYS = frozenset({"recorded_at", "created_at", "updated_at"})


def _scrub(obj: Any) -> Any:
    """Recursively drop per-run wall-clock keys from a JSON-mode structure."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _core_registry() -> type:
    """Isolated core registry with one ``thing`` node kind (never the global)."""
    reg = CoreRegistry()

    @node_type("thing", registry=reg)
    class ThingData(VellaModel):
        label: str = "x"

    return ThingData


def _node(ThingData: type, *, tenant_id: str, node_id: UUID) -> "Node[Any, Any]":
    """A minimal ``thing`` node for the scenario."""
    return Node[ThingData, Any](  # type: ignore[valid-type]
        id=node_id,
        type="thing",
        name="n",
        created_by=UUID("00000000-0000-0000-0000-000000000001"),
        data=ThingData(label="x"),
        tenant_id=tenant_id,
    )


async def _topology_rows() -> "list[dict[str, Any]]":
    """Fold every tenant and build the SET-derived topology projection rows.

    Each row is ``{tenant, node, direction, edge_type, endpoints}`` where
    ``endpoints`` is the SORTED list of the genuine ``set`` of endpoint ids for that
    bucket. The set is the load-bearing nondeterminism source: without ``sorted()``
    the endpoint order would follow hash-seed-driven set iteration.
    """
    rt = Runtime()
    ThingData = _core_registry()
    for tenant_id, node_id in _NODES:
        await rt.create(_node(ThingData, tenant_id=tenant_id, node_id=node_id))
    for tenant_id, from_id, to_id, edge_type in _EDGES:
        await rt.link(tenant_id, from_id, to_id, edge_type)

    proj = GraphProjection()
    rows: list[dict[str, Any]] = []
    for tenant_id in _TENANTS:
        view = await proj.fold(rt, tenant_id, mode="full")
        index = view._internal_index()  # noqa: SLF001 - artifact owns the topology read
        for direction in ("out", "in"):
            for node_id in index.adj[direction]:
                by_type = index.adj[direction][node_id]
                for edge_type, bucket in by_type.items():
                    # Genuine set of endpoint ids -> the set-derived value the
                    # artifact's sorted() tames (set iteration is hash-seed sensitive).
                    endpoints: set[str] = {
                        str(rec.to_id if direction == "out" else rec.from_id)
                        for rec in bucket
                    }
                    rows.append(
                        {
                            "tenant": tenant_id,
                            "node": str(node_id),
                            "direction": direction,
                            "edge_type": edge_type,
                            # THE set-derived sort: removing this sorted() makes the
                            # bytes follow hash-driven set iteration and diverge.
                            "endpoints": sorted(endpoints),
                        }
                    )
    return rows


def build_artifact() -> str:
    """Serialize the topology projection to canonical, byte-stable JSON.

    Rows are ordered by ``sorted(rows, key=(tenant, node, direction, edge_type))`` —
    a stable total order over the buckets — each scrubbed of volatile keys, then
    dumped with ``sort_keys=True, separators=(",", ":")``. The per-bucket endpoint
    lists are already the set-derived ``sorted()`` from :func:`_topology_rows`.

    Returns:
        The canonical JSON string for the sorted topology projection.
    """
    rows = asyncio.run(_topology_rows())
    ordered = sorted(
        rows,
        key=lambda r: (r["tenant"], r["node"], r["direction"], r["edge_type"]),
    )
    scrubbed = [_scrub(r) for r in ordered]
    return json.dumps(scrubbed, sort_keys=True, separators=(",", ":"))


def main() -> None:
    """Print the topology artifact as canonical, byte-stable JSON to stdout."""
    print(build_artifact(), end="")


if __name__ == "__main__":
    main()
