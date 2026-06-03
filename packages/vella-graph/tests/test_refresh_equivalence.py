"""Refresh == fresh-fold equivalence (M5): the correctness oracle.

Because ``Cursor`` has no ordering, "did refresh land at the right position?" cannot
be answered by comparing cursors. The only oracle is **full-refold equivalence**:
fold to a position C1, mutate the runtime to C2, ``refresh`` the C1 view to C2;
SEPARATELY do a FRESH fold of the same runtime (now at C2). Every deterministic
sorted-id query — ``neighbors`` / ``bfs`` / ``dfs`` / ``reachable`` /
``shortest_path``, over all nodes and all directions — must be IDENTICAL between the
refreshed view and the fresh-fold view.

This is the load-bearing M5 gate. ``mut-m5-delta-tombstone-skip`` (refresh skips
applying a delta ``delete`` / ``unlink``) leaves a stale edge in the refreshed view
that the fresh fold does not have, so at least one query diverges → RED.
``mut-m5-full-refold`` is caught structurally by ``test_refresh.py``'s ``is``-identity
assertion (a full re-fold would still be equivalent here, by construction).
"""

from __future__ import annotations

from uuid import UUID, uuid4

from vella.core import EdgeTypes
from vella.runtime import Runtime

from vella.graph import GraphProjection, GraphView
from vella.graph._query import QueryDirection

from _fixtures import drive, make_node, thing_registry

_DIRECTIONS: tuple[QueryDirection, ...] = ("out", "in", "both")


async def _all_query_results(
    view: GraphView, nodes: list[UUID]
) -> list[tuple[str, ...]]:
    """A canonical, sorted dump of every query type over every node/direction.

    Topology-only (ids, not bodies), so it is mode-independent and the exact quantity
    refresh-equivalence claims is identical between a refreshed and a fresh-fold view.
    """
    rows: list[tuple[str, ...]] = []
    sorted_nodes = sorted(nodes, key=str)
    for direction in _DIRECTIONS:
        for n in sorted_nodes:
            neigh = await view.neighbors(n, direction=direction)
            rows.append(
                ("neighbors", str(n), direction)
                + tuple(f"{ne.node_id}:{ne.edge_type}:{ne.edge_id}" for ne in neigh)
            )
            bfs = await view.bfs(n, depth=3, direction=direction)
            rows.append(("bfs", str(n), direction) + tuple(str(x) for x in bfs))
            dfs = await view.dfs(n, depth=3, direction=direction)
            rows.append(("dfs", str(n), direction) + tuple(str(x) for x in dfs))
            for target in sorted_nodes:
                reach = await view.reachable(n, target, direction=direction)
                rows.append(("reach", str(n), str(target), direction, str(reach)))
                sp = await view.shortest_path(n, target, direction=direction)
                seq: tuple[str, ...] = (
                    () if sp is None else tuple(str(x) for x in sp.nodes)
                )
                rows.append(("sp", str(n), str(target), direction) + seq)
    return sorted(rows)


def test_refresh_equals_fresh_fold_all_queries() -> None:
    """Every sorted-id query is identical between refreshed and fresh-fold views."""
    drive(_case())


async def _case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"

    # --- Build to C1 and fold. ---
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    for nid in (a, b, c, d):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)
    await rt.link(tenant, b, c, EdgeTypes.OWNED_BY)
    e_cd = await rt.link(tenant, c, d, EdgeTypes.PART_OF)
    await rt.link(tenant, d, a, EdgeTypes.KNOWS)

    view_c1 = await GraphProjection().fold(rt, tenant, mode="full")

    # --- Mutate to C2: add nodes/edges, edit, delete a node, unlink an edge,
    # emit telemetry (observe_only — must be skipped by refresh too), add a
    # dangling edge. ---
    e, f = uuid4(), uuid4()
    await rt.create(make_node(thing, tenant_id=tenant, node_id=e))
    await rt.create(make_node(thing, tenant_id=tenant, node_id=f))
    await rt.link(tenant, a, e, EdgeTypes.OWNED_BY)
    await rt.link(tenant, e, f, EdgeTypes.KNOWS)
    ghost = uuid4()
    await rt.link(tenant, f, ghost, EdgeTypes.REFERENCES)  # dangling endpoint
    await rt.edit(tenant, a, expected_version=1, name="renamed")
    await rt.emit_telemetry(tenant, a, {"note": "ping"})  # observe_only
    await rt.delete(tenant, c)  # c's incident edges become dangling but survive
    await rt.unlink(tenant, e_cd.entity_id)  # c->d removed

    # --- Refresh the C1 view to C2 vs. a FRESH fold of the runtime (now at C2). ---
    refreshed = await view_c1.refresh(rt)
    fresh = await GraphProjection().fold(rt, tenant, mode="full")

    all_nodes = [a, b, c, d, e, f, ghost]
    refreshed_results = await _all_query_results(refreshed, all_nodes)
    fresh_results = await _all_query_results(fresh, all_nodes)

    assert refreshed_results == fresh_results
    # Sanity: the live edge sets agree too (the topology substrate is identical).
    assert refreshed._internal_index().live_edges == fresh._internal_index().live_edges
    # Sanity: the oracle is non-vacuous — the delta genuinely changed the graph.
    c1_live = view_c1._internal_index().live_edges
    assert refreshed._internal_index().live_edges != c1_live


def test_refresh_lean_equals_fresh_fold() -> None:
    """Lean-mode refresh topology equals a fresh lean fold (mode-independent)."""
    drive(_lean_case())


async def _lean_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b, c = uuid4(), uuid4(), uuid4()
    for nid in (a, b, c):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    view_c1 = await GraphProjection().fold(rt, tenant, mode="lean")

    e_bc = await rt.link(tenant, b, c, EdgeTypes.OWNED_BY)
    await rt.link(tenant, c, a, EdgeTypes.PART_OF)
    await rt.unlink(tenant, e_bc.entity_id)

    refreshed = await view_c1.refresh(rt)
    fresh = await GraphProjection().fold(rt, tenant, mode="lean")

    nodes = [a, b, c]
    assert await _all_query_results(refreshed, nodes) == await _all_query_results(
        fresh, nodes
    )
    # Lean refresh holds no resident bodies (a fresh empty LRU; nothing stale).
    assert refreshed._resident_count() == 0
