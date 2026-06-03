"""observe_only is skipped from tracking but still advances the high-water (M2).

The fold mirrors the reconciler's ``_NON_STATE_CHANGING`` discipline: an
``observe_only`` (telemetry) entry is DRAINED — it advances the internal monotonic
high-water and updates the resume cursor — but it never enters live-set tracking
and never triggers a ``get()``. This module proves all three facets non-vacuously
via two complementary tests.

**Why the original "telemetry on a live edge" test was vacuous** for
mut-m2-observe-only-folds: if telemetry targets an already-live edge id, treating
it as state-changing is an idempotent ``set.add`` on an already-present id, so no
phantom appears in the topology and no extra ``get()`` fires.  The live-edge
topology and get()-count look identical whether ``observe_only`` is skipped or not.

**The non-vacuous fixture** emits telemetry on a PHANTOM entity id — one that was
never created in the runtime.  If the fold treats ``observe_only`` as
state-changing it adds that phantom id to the live set and calls ``get()`` for it.
The spy assertion ``phantom_id not in get_calls`` fires RED for
mut-m2-observe-only-folds.  The topology assertion (phantom absent from the index)
and high-water assertion (cursor DID advance) complete the three facets.
"""

from __future__ import annotations

from typing import Any, Optional, Union
from uuid import UUID, uuid4

from vella.core import Edge, EdgeTypes, Node
from vella.runtime import Runtime

from vella.graph import GraphProjection

from _fixtures import drive, make_node, thing_registry


class _SpyRuntime(Runtime):
    """A ``Runtime`` that records every ``entity_id`` passed to ``get()``."""

    def __init__(self) -> None:
        """Wrap a default runtime and start an empty ``get`` call log."""
        super().__init__()
        self.get_calls: list[UUID] = []

    async def get(
        self, tenant_id: str, entity_id: UUID
    ) -> Optional[Union["Node[Any, Any]", "Edge[Any, Any]"]]:
        """Record ``entity_id`` then delegate to the real read path."""
        self.get_calls.append(entity_id)
        return await super().get(tenant_id, entity_id)


# ---------------------------------------------------------------------------
# Non-vacuous test: telemetry on a PHANTOM (never-created) entity id
# ---------------------------------------------------------------------------

def test_observe_only_phantom_not_folded_not_fetched() -> None:
    """Telemetry on a phantom id: not in topology, no get(), high-water advances.

    Non-vacuity argument: mut-m2-observe-only-folds (treat ``observe_only`` as
    state-changing) adds the phantom id to the live set and calls
    ``get(tenant, phantom_id)`` — the ``phantom_id not in get_calls`` assertion
    fires RED.  The original test (telemetry on a live edge) passes even under
    that mutation because ``set.add`` on an already-present id is a no-op.
    """
    drive(_phantom_case())


async def _phantom_case() -> None:
    rt = _SpyRuntime()
    thing = thing_registry()
    tenant = "t"
    a, b = uuid4(), uuid4()
    await rt.create(make_node(thing, tenant_id=tenant, node_id=a))
    await rt.create(make_node(thing, tenant_id=tenant, node_id=b))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    # phantom_id was NEVER created — it exists only as the target of telemetry.
    phantom_id = uuid4()
    assert phantom_id not in {a, b}

    rt.get_calls.clear()
    await rt.emit_telemetry(tenant, phantom_id, {"note": "ghost"})

    # Capture the true store-stamped cursor of the telemetry entry (the PENDING
    # placeholder on the returned entry is not stamped).
    last_cursor = None
    stream = rt.observe(since=None)
    try:
        async for entry in stream:
            last_cursor = entry.cursor
            if entry.transition == "observe_only":
                break
    finally:
        await stream.aclose()

    view = await GraphProjection().fold(rt, tenant, mode="full")
    idx = view._internal_index()

    # (a) phantom is NOT in the topology.
    assert phantom_id not in idx.live_edges
    assert phantom_id not in idx.live_edges
    assert phantom_id not in idx.node_types
    assert phantom_id not in idx.adj["out"]
    assert phantom_id not in idx.adj["in"]

    # (b) get() was NEVER called for the phantom id.
    # If observe_only were folded as state-changing, the phantom would enter the
    # live set and get(tenant, phantom_id) would be called — the spy fires RED.
    assert phantom_id not in rt.get_calls, (
        f"get() was called for phantom id {phantom_id} — "
        "observe_only was incorrectly treated as state-changing"
    )

    # (c) the high-water DID advance past the telemetry entry (it was drained).
    assert view.high_water == last_cursor, (
        f"high-water {view.high_water!r} != telemetry cursor {last_cursor!r} — "
        "observe_only was not drained at all"
    )


# ---------------------------------------------------------------------------
# Original test retained: telemetry on a live edge (topology + count checks)
# ---------------------------------------------------------------------------

def test_observe_only_on_live_edge_no_topology_change() -> None:
    """Telemetry on a live edge: topology unchanged, get() count exact, cursor advances."""
    drive(_live_edge_case())


async def _live_edge_case() -> None:
    rt = _SpyRuntime()
    thing = thing_registry()
    tenant = "t"
    a, b = uuid4(), uuid4()
    await rt.create(make_node(thing, tenant_id=tenant, node_id=a))
    await rt.create(make_node(thing, tenant_id=tenant, node_id=b))
    e = await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    before = await GraphProjection().fold(rt, tenant, mode="full")
    before_topo = [
        (r.from_id, r.to_id, r.edge_type)
        for r in before._internal_index().neighbors(a, "out")
    ]
    hw_before = before.high_water

    rt.get_calls.clear()
    await rt.emit_telemetry(tenant, e.entity_id, {"note": "ping"})

    last_cursor = None
    stream = rt.observe(since=None)
    try:
        async for entry in stream:
            last_cursor = entry.cursor
            if entry.transition == "observe_only":
                break
    finally:
        await stream.aclose()

    after = await GraphProjection().fold(rt, tenant, mode="full")
    after_idx = after._internal_index()
    after_topo = [
        (r.from_id, r.to_id, r.edge_type)
        for r in after_idx.neighbors(a, "out")
    ]

    # (a) topology unchanged.
    assert after_topo == before_topo
    assert after_idx.live_edges == before._internal_index().live_edges

    # (b) exactly 3 get() calls (a, b, edge) — no re-fetch for the telemetry entry.
    assert len(rt.get_calls) == 3
    assert rt.get_calls.count(e.entity_id) == 1

    # (c) high-water advanced past the telemetry entry.
    assert after.high_water == last_cursor
    assert after.high_water != hw_before
