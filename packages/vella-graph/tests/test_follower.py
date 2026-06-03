"""The opt-in background follower (M6): backlog fold + live-tail incremental fold.

``GraphFollower`` is the only async-lifecycle surface: a single watch task loops
the generic bounded-drain over ``observe()``'s live tail to keep a ``GraphView``
current. These cases drive it under a :class:`ManualClock` with a bounded
``run(max_steps=)`` (no ``pytest-asyncio``) and assert equivalence AT the explicit
``caught_up`` Event — never after a fixed sleep.

Load-bearing mutation targets proven here:

* **mut-m6-early-drain** — if the drain stops one entry early, the followed view's
  high-water (and topology) lags the runtime, so the live-tail equivalence assertion
  goes RED (a missing edge / a stale high-water cursor).
* **mut-m6-no-quiescence-signal** — if ``bounded_drain``'s ``on_caught_up`` never
  fires the Event, ``await caught_up.wait()`` never returns and the bounded
  ``asyncio.wait_for`` backstop times out (RED).
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from vella.core import EdgeTypes
from vella.runtime import Runtime

from vella.graph import GraphFollower, GraphProjection, ManualClock

from _fixtures import make_node, thing_registry


def _drive(coro: Any, *, timeout: float = 5.0) -> Any:
    """Run ``coro`` under a bounded backstop so a coordination bug fails fast."""
    return asyncio.run(asyncio.wait_for(coro, timeout=timeout))


async def _stop(task: "asyncio.Task[None]") -> None:
    """Cancel a background ``run()`` task and await it to ``done()`` (clean teardown)."""
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.CancelledError:
        pass
    assert task.done()


async def _await_edge(follower: GraphFollower, edge_id: Any, *, steps: int = 500) -> None:
    """Yield until the followed view's live set contains ``edge_id`` (bounded).

    The follower's quiescence Event marks the live edge; we additionally poll the
    view so a spurious early quiescence (or a delayed authority pass) cannot let the
    assertion run against a not-yet-current view. Bounded so a regression fails fast
    under the outer ``asyncio.wait_for`` rather than spinning forever.
    """
    for _ in range(steps):
        if edge_id in follower.view()._internal_index().live_edges:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"follower never folded edge {edge_id} into its view")


def test_follower_folds_backlog_at_caught_up() -> None:
    """The follower folds the pre-existing backlog; equivalence asserted AT the Event."""
    _drive(_backlog_case())


async def _backlog_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b, c = uuid4(), uuid4(), uuid4()
    for nid in (a, b, c):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)
    await rt.link(tenant, a, c, EdgeTypes.OWNED_BY)

    follower = GraphFollower(rt, tenant, mode="full", clock=ManualClock())
    task = asyncio.ensure_future(follower.run(max_steps=1))
    try:
        # Quiescence is the EXPLICIT Event, not a sleep: block on it, then assert.
        await asyncio.wait_for(follower.caught_up.wait(), timeout=2.0)

        followed = follower.view()
        fresh = await GraphProjection().fold(rt, tenant, mode="full")
        # mut-m6-early-drain: a one-entry-short drain drops an edge here.
        assert followed._internal_index().live_edges == fresh._internal_index().live_edges
        assert sorted(str(r.to_id) for r in followed._internal_index().neighbors(a, "out")) == sorted(
            str(r.to_id) for r in fresh._internal_index().neighbors(a, "out")
        )
        # High-water reached the same cursor the cold fold did (the full backlog
        # was drained — mut-m6-early-drain leaves this lagging).
        assert followed.high_water is not None
    finally:
        # max_steps=1 → run() returns after the backlog pass and tears the watch down.
        await asyncio.wait_for(task, timeout=2.0)
    assert task.done()


def test_follower_tracks_live_tail_incrementally() -> None:
    """Entries appended AFTER catch-up are incrementally folded; equiv at re-quiescence."""
    _drive(_live_tail_case())


async def _live_tail_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b = uuid4(), uuid4()
    for nid in (a, b):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    # Unbounded background run: fold the backlog, then track the live tail until we
    # cancel it. The watch task blocks on the live edge between entries.
    follower = GraphFollower(rt, tenant, mode="full", clock=ManualClock())
    task = asyncio.ensure_future(follower.run())
    try:
        # First quiescence: backlog folded.
        await asyncio.wait_for(follower.caught_up.wait(), timeout=2.0)
        first = follower.view()
        assert sorted(str(r.to_id) for r in first._internal_index().neighbors(a, "out")) == [str(b)]

        # Append a NEW live-tail entry while the follower watches.
        c = uuid4()
        await rt.create(make_node(thing, tenant_id=tenant, node_id=c))
        e_ac = await rt.link(tenant, a, c, EdgeTypes.OWNED_BY)

        # The live-tail delta is folded incrementally — wait until the view reflects
        # it (mut-m6-early-drain: the e_ac edge is the one a short drain would drop,
        # so this bounded wait would time out / the equivalence below would fail).
        await _await_edge(follower, e_ac.entity_id)
        followed = follower.view()
        fresh = await GraphProjection().fold(rt, tenant, mode="full")
        assert e_ac.entity_id in followed._internal_index().live_edges
        assert followed._internal_index().live_edges == fresh._internal_index().live_edges
        assert sorted(str(r.to_id) for r in followed._internal_index().neighbors(a, "out")) == sorted(
            [str(b), str(c)]
        )
    finally:
        await _stop(task)
