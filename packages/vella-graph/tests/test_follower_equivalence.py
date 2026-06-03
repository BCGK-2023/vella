"""Follower==fresh-fold equivalence at quiescence (M6).

The follower's incremental fold is correct iff a followed view AT QUIESCENCE is
indistinguishable from a fresh cold fold to the same cursor — every sorted-id query
result byte-identical, in BOTH ``full`` and ``lean`` modes. Equivalence is asserted
at the explicit ``caught_up`` Event (never a sleep). This is the deep oracle behind
``test_follower``'s direct delta checks; ``mut-m6-early-drain`` (a one-entry-short
drain) desynchronises the followed topology and trips it.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

from vella.core import EdgeTypes
from vella.runtime import Runtime

from vella.graph import GraphFollower, GraphProjection, GraphView, ManualClock
from vella.graph.mode import MaterializationMode

from _fixtures import make_node, thing_registry


def _drive(coro: Any, *, timeout: float = 5.0) -> Any:
    """Run ``coro`` under a bounded backstop so a coordination bug fails fast."""
    return asyncio.run(asyncio.wait_for(coro, timeout=timeout))


async def _all_query_results(view: GraphView, nodes: list[UUID]) -> dict[str, Any]:
    """Every sorted-id query result over ``view`` (topology — mode-independent)."""
    out: dict[str, Any] = {}
    out["live_edges"] = sorted(str(e) for e in view._internal_index().live_edges)
    out["node_types"] = sorted(str(n) for n in view._internal_index().node_types)
    for n in nodes:
        out[f"neighbors:{n}"] = [
            (str(x.node_id), x.edge_type, str(x.edge_id))
            for x in await view.neighbors(n, direction="both")
        ]
        out[f"bfs:{n}"] = [str(x) for x in await view.bfs(n, depth=10, direction="both")]
        out[f"dfs:{n}"] = [str(x) for x in await view.dfs(n, depth=10, direction="both")]
    return out


def test_followed_view_equals_fresh_fold_full() -> None:
    """Full-mode: followed view at quiescence == fresh fold (every query result)."""
    _drive(_equivalence_case("full"))


def test_followed_view_equals_fresh_fold_lean() -> None:
    """Lean-mode: followed view at quiescence == fresh fold (every query result)."""
    _drive(_equivalence_case("lean"))


async def _equivalence_case(mode: MaterializationMode) -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    for nid in (a, b, c, d):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)
    await rt.link(tenant, b, c, EdgeTypes.OWNED_BY)
    await rt.link(tenant, a, d, EdgeTypes.PART_OF)
    # A foreign-tenant entry the follower must skip (tenancy preserved under follow).
    other = uuid4()
    await rt.create(make_node(thing, tenant_id="other", node_id=other))

    follower = GraphFollower(rt, tenant, mode=mode, clock=ManualClock())
    task = asyncio.ensure_future(follower.run(max_steps=1))
    try:
        await asyncio.wait_for(follower.caught_up.wait(), timeout=2.0)
        followed = follower.view()
        fresh = await GraphProjection().fold(rt, tenant, mode=mode)

        nodes = [a, b, c, d]
        got = await _all_query_results(followed, nodes)
        want = await _all_query_results(fresh, nodes)
        assert got == want
        # The foreign tenant never entered the followed topology.
        assert str(other) not in got["node_types"]
    finally:
        await asyncio.wait_for(task, timeout=2.0)
    assert task.done()
