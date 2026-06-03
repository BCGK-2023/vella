"""Topology mode-equivalence (M2): full and lean yield identical topology.

ADVERSARIAL fixture (pre-mortem scenario (a)) — it MUST include all three traps:

* (a) a DANGLING edge whose ``to`` endpoint node was never created;
* (b) a node DELETED in the stream (a ``delete`` transition the fold must apply),
  whose incident edges survive as dangling;
* (c) MULTI-BUCKET topology: >= 3 edge types across >= 4 nodes, both directions.

Non-vacuity argument: the topology projection is **id-derived, not body-derived**
(spec decision #4) — ``neighbors`` returns endpoint ids straight from the index,
independent of whether a body is resident. So the sorted-id topology is identical
under ``full`` (bodies resident) and ``lean`` (no bodies) EVEN for the dangling
edge (whose endpoint has no body in either mode) and the deleted node (whose body
is ``None`` in both). A mode that silently dropped dangling/deleted-incident edges
in ``lean`` (mut-m2-lean-drops-dangling) would make full != lean here. The
"always-built" assertion (lean topology == full topology, only resident counts
differ) catches mut-m2-lean-skips-topology.
"""

from __future__ import annotations

from uuid import uuid4

from vella.core import EdgeTypes
from vella.runtime import Runtime

from vella.graph import GraphProjection
from vella.graph._index import GraphIndex

from _fixtures import drive, make_node, thing_registry


def _topology_projection(idx: GraphIndex) -> list[tuple[str, str, str, list[str]]]:
    """Canonical sorted-id topology: ``(node, direction, edge_type, sorted to/from)``.

    Walks every anchor in both directions, every edge-type bucket, and emits the
    sorted endpoint ids. Fully id-derived — no body access — so it is the exact
    quantity mode-equivalence claims is byte-identical across modes.
    """
    rows: list[tuple[str, str, str, list[str]]] = []
    for direction in ("in", "out"):
        for node_id in sorted(idx.adj[direction], key=str):
            by_type = idx.adj[direction][node_id]
            for edge_type in sorted(by_type):
                endpoints = sorted(
                    str(r.to_id if direction == "out" else r.from_id)
                    for r in by_type[edge_type]
                )
                rows.append((str(node_id), direction, edge_type, endpoints))
    return sorted(rows)


def test_full_and_lean_topology_byte_identical() -> None:
    """Sorted-id topology is identical under full and lean; resident counts differ."""
    drive(_case())


async def _case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"

    # (c) >= 4 nodes
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    for nid in (a, b, c, d):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))

    # (c) >= 3 edge types, both directions exercised
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)
    await rt.link(tenant, b, c, EdgeTypes.OWNED_BY)
    await rt.link(tenant, c, a, EdgeTypes.PART_OF)
    await rt.link(tenant, d, a, EdgeTypes.KNOWS)

    # (a) DANGLING edge: endpoint node 'ghost' was never created.
    ghost = uuid4()
    await rt.link(tenant, a, ghost, EdgeTypes.REFERENCES)

    # (b) DELETED node: delete b (its incident KNOWS / OWNED_BY edges stay dangling).
    await rt.delete(tenant, b)

    proj = GraphProjection()
    full = await proj.fold(rt, tenant, mode="full")
    lean = await proj.fold(rt, tenant, mode="lean")

    full_topo = _topology_projection(full._internal_index())
    lean_topo = _topology_projection(lean._internal_index())

    # Mode-equivalence: byte-identical topology projection.
    assert full_topo == lean_topo

    # "Always-built" index: lean topology equals full topology (only residency
    # differs — full holds bodies, lean holds none).
    assert lean._internal_index().live_edges == full._internal_index().live_edges
    assert full._resident_count() > 0
    assert lean._resident_count() == 0

    # Sanity: the adversarial traps are actually present in the fixture.
    full_idx = full._internal_index()
    # (a) dangling edge a->ghost present though ghost was never created.
    assert ghost in {r.to_id for r in full_idx.neighbors(a, "out")}
    assert ghost not in full_idx.node_types
    # (b) deleted node b is gone from node_types but its incident edges survive.
    assert b not in full_idx.node_types
    assert any(r.to_id == b for r in full_idx.neighbors(a, "out"))  # a->b KNOWS kept
    # (c) >= 3 distinct edge types across the live set (multi-bucket topology).
    all_types = {
        rec.edge_type
        for node_map in full_idx.adj["out"].values()
        for bucket in node_map.values()
        for rec in bucket
    }
    assert all_types >= {EdgeTypes.KNOWS, EdgeTypes.OWNED_BY, EdgeTypes.PART_OF}
