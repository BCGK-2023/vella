"""Pull ``refresh()`` (M5): delta correctness, COW ``is``-sharing, immutability.

``refresh`` is the incremental fold: it drains ``observe(since=high_water)`` from
the view's stored opaque cursor, applies only the delta, and returns a NEW frozen
view that shares this view's untouched index buckets BY IDENTITY (copy-on-write).
The four load-bearing properties, each with the named mutation it must catch:

* (a) **delta correctness** — create/link/edit/delete/unlink AFTER the fold are
  reflected by ``refresh`` (the equivalence oracle in ``test_refresh_equivalence``
  proves the deep version; this asserts the obvious deltas directly).
* (b) **COW ``is``-identity** — every untouched ``(direction, node_id)`` bucket is
  the SAME OBJECT in the refreshed index; a touched node's bucket is a NEW object.
  ``mut-m5-full-refold`` (re-fold from scratch instead of ``since=high_water``)
  rebuilds every bucket, so the untouched-``is`` assertion goes RED.
* (c) **immutability** — the pre-refresh view's query results are unchanged after
  ``refresh``. ``mut-m5-cow-mutate`` (mutate a shared bucket in place) changes the
  pre-refresh view's results, so this goes RED.
* (d) **cursor-opaque** — ``refresh`` passes ``since=high_water`` straight to
  ``observe`` as an opaque token and NEVER compares cursors (``Cursor`` has no
  ``__lt__`` — any comparison would raise ``TypeError``). Confirmed structurally
  (no comparison operators in ``_refresh.py``) AND behaviourally (the exact stored
  cursor object reaches ``observe``; refresh works regardless of token format).
"""

from __future__ import annotations

import ast
import inspect
from typing import AsyncGenerator, Optional
from uuid import UUID, uuid4

from vella.core import EdgeTypes
from vella.runtime import Cursor, LogEntry, Runtime

from vella.graph import GraphProjection, GraphView
from vella.graph import _refresh as refresh_module
from vella.graph._index import Direction

from _fixtures import drive, make_node, thing_registry

_DIRECTIONS: tuple[Direction, ...] = ("out", "in")


def _bucket_ids(view: GraphView) -> dict[tuple[Direction, UUID], int]:
    """``(direction, node_id) -> id()`` of every adjacency sub-dict in a view.

    The COW sharing claim is over these per-anchor ``edge_type -> bucket`` maps:
    untouched anchors must keep the SAME object identity across a ``refresh``.
    """
    idx = view._internal_index()
    out: dict[tuple[Direction, UUID], int] = {}
    for direction in _DIRECTIONS:
        for node_id, by_type in idx.adj[direction].items():
            out[(direction, node_id)] = id(by_type)
    return out


def test_refresh_reflects_delta() -> None:
    """create/link/edit/delete/unlink after the fold are reflected by refresh (a)."""
    drive(_delta_case())


async def _delta_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b, c = uuid4(), uuid4(), uuid4()
    for nid in (a, b, c):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    e_ab = await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    view = await GraphProjection().fold(rt, tenant, mode="full")
    idx0 = view._internal_index()
    assert [r.to_id for r in idx0.neighbors(a, "out")] == [b]
    assert idx0.live_edges == frozenset({e_ab.entity_id})

    # Delta AFTER the fold: new node d, link a->c and a->d, edit a, delete c,
    # unlink a->b.
    d = uuid4()
    await rt.create(make_node(thing, tenant_id=tenant, node_id=d))
    e_ac = await rt.link(tenant, a, c, EdgeTypes.OWNED_BY)
    e_ad = await rt.link(tenant, a, d, EdgeTypes.KNOWS)
    await rt.edit(tenant, a, expected_version=1, name="renamed")
    await rt.delete(tenant, c)
    await rt.unlink(tenant, e_ab.entity_id)

    refreshed = await view.refresh(rt)
    idx = refreshed._internal_index()

    # a->b unlinked (gone); a->c (dangling, c deleted) and a->d survive.
    assert idx.live_edges == frozenset({e_ac.entity_id, e_ad.entity_id})
    assert sorted(str(r.to_id) for r in idx.neighbors(a, "out")) == sorted(
        [str(c), str(d)]
    )
    # b has no in-edge anymore (a->b unlinked).
    assert idx.neighbors(b, "in") == ()
    # a->c kept though c was deleted (dangling; ids are truth).
    assert [r.from_id for r in idx.neighbors(c, "in")] == [a]
    assert c not in idx.node_types
    # new node d is live and is a's neighbour.
    assert d in idx.node_types
    assert a in {r.from_id for r in idx.neighbors(d, "in")}


def test_refresh_cow_is_identity() -> None:
    """Untouched buckets are shared by identity; touched buckets are new (b)."""
    drive(_cow_case())


async def _cow_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b, c, x, y = uuid4(), uuid4(), uuid4(), uuid4(), uuid4()
    for nid in (a, b, c, x, y):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    # Independent components: a->b, c (isolated), x->y. Refresh only touches a's side.
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)
    await rt.link(tenant, x, y, EdgeTypes.OWNED_BY)

    original = await GraphProjection().fold(rt, tenant, mode="full")
    before = _bucket_ids(original)

    # Delta touches only node a (new out-edge a->c).
    await rt.link(tenant, a, c, EdgeTypes.PART_OF)
    refreshed = await original.refresh(rt)
    after = _bucket_ids(refreshed)
    ref_idx = refreshed._internal_index()
    orig_idx = original._internal_index()

    # Touched anchors: a's out-bucket AND c's in-bucket are NEW objects.
    assert ("out", a) in after
    assert ref_idx.adj["out"][a] is not orig_idx.adj["out"][a]
    assert ("in", c) in after
    assert orig_idx.adj["in"].get(c) is None  # c had no in-edge before
    assert ref_idx.adj["in"][c] is not orig_idx.adj["in"].get(c)

    # Every UNTOUCHED (direction, node_id) bucket is the SAME OBJECT by identity.
    # x->y is wholly untouched; b's in-bucket and y's in-bucket are untouched.
    touched: set[tuple[Direction, UUID]] = {("out", a), ("in", c)}
    untouched = [k for k in before if k not in touched]
    assert untouched  # the fixture has untouched buckets to share
    for direction, node in untouched:
        assert ref_idx.adj[direction][node] is orig_idx.adj[direction][node], (
            f"untouched bucket {(direction, node)} was not shared by identity"
        )
    # Explicit named checks on the wholly-untouched component x->y.
    assert ref_idx.adj["out"][x] is orig_idx.adj["out"][x]
    assert ref_idx.adj["in"][y] is orig_idx.adj["in"][y]


def test_refresh_immutability() -> None:
    """The pre-refresh view's results are unchanged after refresh (c)."""
    drive(_immutability_case())


async def _immutability_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b, c = uuid4(), uuid4(), uuid4()
    for nid in (a, b, c):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    view = await GraphProjection().fold(rt, tenant, mode="full")
    # Snapshot the pre-refresh results.
    before_out = [str(r.to_id) for r in view._internal_index().neighbors(a, "out")]
    before_neighbors = [str(n.node_id) for n in await view.neighbors(a, direction="out")]
    before_hw = view.high_water
    before_live = view._internal_index().live_edges

    # Mutate the runtime and refresh.
    await rt.link(tenant, a, c, EdgeTypes.OWNED_BY)
    await rt.delete(tenant, b)
    refreshed = await view.refresh(rt)

    # The new view saw the delta...
    assert sorted(
        str(r.to_id) for r in refreshed._internal_index().neighbors(a, "out")
    ) == sorted([str(b), str(c)])

    # ...but the pre-refresh view is byte-for-byte unchanged.
    assert [str(r.to_id) for r in view._internal_index().neighbors(a, "out")] == before_out
    assert [str(n.node_id) for n in await view.neighbors(a, direction="out")] == before_neighbors
    assert view.high_water is before_hw
    assert view._internal_index().live_edges == before_live
    # refresh returned a genuinely new object.
    assert refreshed is not view


def test_refresh_cursor_is_opaque_structural() -> None:
    """No ordering comparison in the refresh code AND the token is passed through (d).

    Parses ``_refresh.py`` (so docstrings/comments are excluded — only executable
    code is inspected) and asserts there is NO ordering ``Compare`` node anywhere
    (``Cursor`` has no ``__lt__``; a refresh that compared cursors would raise
    ``TypeError``). The cursor must reach ``observe`` straight as ``since=``.
    """
    tree = ast.parse(inspect.getsource(refresh_module))
    ordering = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                assert not isinstance(op, ordering), (
                    f"refresh uses an ordering comparison {type(op).__name__} "
                    "(cursors must never be compared)"
                )
    # It must pass the opaque token straight through to observe(since=...).
    assert "observe(since=high_water)" in inspect.getsource(refresh_module)


def test_refresh_passes_exact_cursor_token_format_agnostic() -> None:
    """refresh hands the EXACT stored cursor to observe, regardless of token text (d)."""
    drive(_opaque_case())


async def _opaque_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b = uuid4(), uuid4()
    for nid in (a, b):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    view = await GraphProjection().fold(rt, tenant, mode="full")
    hw = view.high_water
    assert hw is not None

    seen: list[Optional[Cursor]] = []

    class _SpyRuntime(Runtime):
        def observe(self, since: Optional[Cursor] = None) -> AsyncGenerator[LogEntry, None]:
            # Record the exact token object refresh passes (opacity proof: it is the
            # stored cursor passed straight through, never reconstructed/compared).
            seen.append(since)
            return super().observe(since=since)

    spy = _SpyRuntime.__new__(_SpyRuntime)
    spy.__dict__.update(rt.__dict__)  # share the same underlying store/log

    c = uuid4()
    await rt.create(make_node(thing, tenant_id=tenant, node_id=c))
    await rt.link(tenant, a, c, EdgeTypes.OWNED_BY)

    refreshed = await view.refresh(spy)

    # The EXACT stored cursor object reached observe (opaque pass-through).
    assert len(seen) == 1
    assert seen[0] is hw
    # And the refresh actually worked (cursor format never inspected/compared).
    assert sorted(
        str(r.to_id) for r in refreshed._internal_index().neighbors(a, "out")
    ) == sorted([str(b), str(c)])


def test_refresh_empty_delta_keeps_high_water_and_shares_all() -> None:
    """An empty delta returns the same high-water and shares EVERY bucket (b/d)."""
    drive(_empty_delta_case())


async def _empty_delta_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b = uuid4(), uuid4()
    for nid in (a, b):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    view = await GraphProjection().fold(rt, tenant, mode="full")
    orig_idx = view._internal_index()

    # No mutation between fold and refresh: the delta is empty.
    refreshed = await view.refresh(rt)
    ref_idx = refreshed._internal_index()

    # High-water unchanged (returned verbatim, never advanced past the live edge).
    assert refreshed.high_water is view.high_water
    # Every bucket shared by identity (nothing touched).
    for direction in _DIRECTIONS:
        for node_id in orig_idx.adj[direction]:
            assert ref_idx.adj[direction][node_id] is orig_idx.adj[direction][node_id]
    assert ref_idx.live_edges == orig_idx.live_edges
