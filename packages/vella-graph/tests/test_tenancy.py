"""Tenancy isolation (M2): a fold for tenant A contains no tenant-B entries.

``observe()`` is the single GLOBAL stream (it carries no tenant filter by design),
so the fold must filter by ``LogEntry.tenant_id`` itself. This feeds two tenants'
nodes and edges through one runtime, folds for tenant A only, and asserts:

1. **Output isolation** — no tenant-B id appears anywhere in the built index.
2. **Input isolation** (non-vacuous gate) — ``get()`` is NEVER called with a
   tenant-B entity id, proving foreign entries are filtered at the ``apply()``
   level BEFORE the authority pass, not saved by the fact that
   ``get("A", b_entity_id)`` returns ``None``.

Non-vacuity for mut-m2-tenant-leak: dropping the ``apply()``-level tenant filter
puts B entity ids into ``live_edges``/``live_nodes``; the fold then calls
``get(tenant_id="A", entity_id=<b_entity_id>)`` — the spy assertion
``b_ids.isdisjoint(entity_ids_passed_to_get)`` fires RED.  The output-only
assertion stays GREEN in that mutation (cross-tenant get returns None so no B
entity leaks into the index), proving the spy assertion is the load-bearing gate.
"""

from __future__ import annotations

from typing import Any, Optional, Union
from uuid import UUID, uuid4

from vella.core import Edge, EdgeTypes, Node
from vella.runtime import Runtime

from vella.graph import GraphProjection

from _fixtures import drive, make_node, thing_registry


class _GetSpyRuntime(Runtime):
    """A ``Runtime`` that records every ``entity_id`` passed to ``get()``.

    Used to assert the fold never calls ``get()`` on a foreign-tenant entity,
    proving the tenant filter lives in ``apply()`` rather than being saved by
    ``get()``'s own tenant scoping.
    """

    def __init__(self) -> None:
        """Initialise with an empty call log."""
        super().__init__()
        self.get_entity_ids: list[UUID] = []

    async def get(
        self, tenant_id: str, entity_id: UUID
    ) -> Optional[Union["Node[Any, Any]", "Edge[Any, Any]"]]:
        """Record ``entity_id`` then delegate to the real read path."""
        self.get_entity_ids.append(entity_id)
        return await super().get(tenant_id, entity_id)


def test_fold_excludes_other_tenants() -> None:
    """Folding tenant A never surfaces any tenant-B entity."""
    drive(_case())


async def _case() -> None:
    rt = _GetSpyRuntime()
    thing = thing_registry()

    a1, a2 = uuid4(), uuid4()  # tenant A nodes
    b1, b2 = uuid4(), uuid4()  # tenant B nodes

    await rt.create(make_node(thing, tenant_id="A", node_id=a1))
    await rt.create(make_node(thing, tenant_id="A", node_id=a2))
    await rt.create(make_node(thing, tenant_id="B", node_id=b1))
    await rt.create(make_node(thing, tenant_id="B", node_id=b2))

    e_a = await rt.link("A", a1, a2, EdgeTypes.KNOWS)
    e_b = await rt.link("B", b1, b2, EdgeTypes.KNOWS)

    # Clear calls made during setup writes; we only care about fold-time get()s.
    rt.get_entity_ids.clear()

    view = await GraphProjection().fold(rt, "A", mode="full")
    idx = view._internal_index()

    b_ids = {b1, b2, e_b.entity_id}

    # --- Output isolation (what the index contains) ---
    # No tenant-B id in node_types, live_edges, or adjacency keys/endpoints.
    assert b_ids.isdisjoint(idx.node_types.keys())
    assert b_ids.isdisjoint(idx.live_edges)
    for direction in ("out", "in"):
        assert b_ids.isdisjoint(idx.adj[direction].keys())
        for node_map in idx.adj[direction].values():
            for bucket in node_map.values():
                for rec in bucket:
                    assert rec.from_id not in {b1, b2}
                    assert rec.to_id not in {b1, b2}
                    assert rec.edge_id != e_b.entity_id

    # --- Input isolation (what get() was called with) — the load-bearing gate ---
    # The fold must NEVER call get() on a tenant-B entity id.  If the apply()-level
    # filter is dropped, B entity ids enter the live set and get() is called for
    # them; this assertion fires RED even though the output looks clean (because
    # get("A", b_entity_id) returns None and nothing leaks into the index).
    called_ids = set(rt.get_entity_ids)
    assert b_ids.isdisjoint(called_ids), (
        f"get() was called with B-tenant entity ids: {b_ids & called_ids}"
    )

    # get() WAS called for the A-side entities (sanity: the authority pass ran).
    a_ids = {a1, a2, e_a.entity_id}
    assert a_ids.issubset(called_ids), (
        f"get() missing expected A-tenant calls: {a_ids - called_ids}"
    )

    # Tenant A's own topology IS present.
    assert e_a.entity_id in idx.live_edges
    assert [r.to_id for r in idx.neighbors(a1, "out")] == [a2]
