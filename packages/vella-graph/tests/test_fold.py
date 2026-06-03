"""Fold correctness (M2): create/link/edit/delete/unlink/observe_only.

Feeds a mixed transition sequence through a REAL ``Runtime`` (default in-memory
store) and asserts the folded index reflects each transition: links appear in both
directions, ``edit`` does not change topology, ``observe_only`` is skipped, and
``delete``/``unlink`` remove the entity from the live set.
"""

from __future__ import annotations

from uuid import uuid4

from vella.core import EdgeTypes
from vella.runtime import Runtime

from vella.graph import GraphProjection

from _fixtures import drive, make_node, thing_registry


def test_fold_reflects_each_transition() -> None:
    """A create/link/edit/delete/unlink/observe_only sequence folds correctly."""
    drive(_case())


async def _case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b, c = uuid4(), uuid4(), uuid4()

    # create three nodes
    for nid in (a, b, c):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))

    # link a->b (KNOWS) and a->c (OWNED_BY)
    e_ab = await rt.link(tenant, a, b, EdgeTypes.KNOWS)
    e_ac = await rt.link(tenant, a, c, EdgeTypes.OWNED_BY)

    # edit a node (topology must be unaffected)
    await rt.edit(tenant, a, expected_version=1, name="renamed")

    # emit telemetry on edge e_ab (observe_only — must be skipped, no topo change)
    await rt.emit_telemetry(tenant, e_ab.entity_id, {"note": "ping"})

    # delete node c (its incident edge a->c becomes dangling, but stays)
    await rt.delete(tenant, c)

    # unlink edge a->b (removed from the index)
    await rt.unlink(tenant, e_ab.entity_id)

    view = await GraphProjection().fold(rt, tenant, mode="full")
    idx = view._internal_index()

    # Live edges: only a->c remains (a->b was unlinked).
    assert idx.live_edges == frozenset({e_ac.entity_id})

    # a's out-neighbors: only c remains.
    out_a = [r.to_id for r in idx.neighbors(a, "out")]
    assert out_a == [c]
    # a->b was unlinked: b has no in-edge.
    assert idx.neighbors(b, "in") == ()
    # a->c survives even though c was deleted (dangling kept; ids are truth).
    assert [r.from_id for r in idx.neighbors(c, "in")] == [a]

    # Deleted node c is gone from node_types; a and b remain live nodes.
    assert c not in idx.node_types
    assert a in idx.node_types and b in idx.node_types

    # The OWNED_BY edge kept its authoritative type (read via get(), not payload).
    (rec,) = idx.neighbors(a, "out")
    assert rec.edge_type == EdgeTypes.OWNED_BY


def test_observe_only_does_not_create_topology() -> None:
    """An observe_only entry on a node never adds it to the index topology."""
    drive(_observe_only_case())


async def _observe_only_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a = uuid4()
    await rt.create(make_node(thing, tenant_id=tenant, node_id=a))
    await rt.emit_telemetry(tenant, a, {"k": "v"})

    view = await GraphProjection().fold(rt, tenant, mode="lean")
    idx = view._internal_index()
    # The node is live (from its create), but telemetry added no edge.
    assert idx.neighbors(a, "out") == ()
    assert idx.neighbors(a, "in") == ()
    assert idx.live_edges == frozenset()
