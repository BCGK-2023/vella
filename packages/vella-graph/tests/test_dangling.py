"""Dangling edges (M2): ids are truth (spec decision #4).

Two cases: (1) an edge whose endpoint node was NEVER created is retained, and its
absent endpoint id is still returned by ``neighbors``; (2) deleting a node leaves
its incident edges present (now dangling). The caller hydrates an id to discover
absence — the index never silently drops an edge for a missing endpoint.
"""

from __future__ import annotations

from uuid import uuid4

from vella.core import EdgeTypes
from vella.runtime import Runtime

from vella.graph import GraphProjection

from _fixtures import drive, make_node, thing_registry


def test_edge_to_absent_node_is_retained() -> None:
    """An edge to a never-created node keeps the absent endpoint id."""
    drive(_absent_endpoint_case())


async def _absent_endpoint_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a = uuid4()
    ghost = uuid4()  # never created
    await rt.create(make_node(thing, tenant_id=tenant, node_id=a))
    e = await rt.link(tenant, a, ghost, EdgeTypes.REFERENCES)

    idx = (await GraphProjection().fold(rt, tenant, mode="full"))._internal_index()

    # The edge is live and a's out-neighbor is the absent ghost id.
    assert e.entity_id in idx.live_edges
    assert [r.to_id for r in idx.neighbors(a, "out")] == [ghost]
    # ghost has no body / type, but its id is truth in the index.
    assert ghost not in idx.node_types


def test_deleting_node_leaves_incident_edges_dangling() -> None:
    """Deleting a node retains its incident edges (now dangling)."""
    drive(_delete_case())


async def _delete_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b = uuid4(), uuid4()
    await rt.create(make_node(thing, tenant_id=tenant, node_id=a))
    await rt.create(make_node(thing, tenant_id=tenant, node_id=b))
    e = await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    # delete b after linking; the a->b edge must survive as dangling.
    await rt.delete(tenant, b)

    idx = (await GraphProjection().fold(rt, tenant, mode="full"))._internal_index()

    assert e.entity_id in idx.live_edges
    assert [r.to_id for r in idx.neighbors(a, "out")] == [b]
    assert [r.from_id for r in idx.neighbors(b, "in")] == [a]
    # b's body is gone (deleted), but b's id remains an endpoint.
    assert b not in idx.node_types
