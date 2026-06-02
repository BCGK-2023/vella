"""Observe stream tests for the ``Runtime`` facade (M4).

Covers ``observe`` (catch-up-then-live, resume-from-cursor, total order,
cross-tenant global semantics). Each case uses ``asyncio.run`` — no
async-plugin dependency, matching M2/M3 patterns.

Catch-up-then-live without hanging: after collecting the backlog via a
bounded number of ``anext`` calls we append one more entry and pull it with
a single additional ``anext``. The iterator is never iterated beyond what we
explicitly pull, so the test terminates deterministically.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator
from uuid import uuid4

from vella.core import Node, VellaModel, Registry, node_type

from vella.runtime import Runtime
from vella.runtime.log import Cursor, LogEntry


def _registry() -> tuple[Registry, type]:
    """Isolated registry with one ``doc`` node type (never the global)."""
    reg = Registry()

    @node_type("doc", registry=reg)
    class DocData(VellaModel):
        title: str

    return reg, DocData


def _make_node(
    DocData: type,
    *,
    title: str = "hello",
    tenant_id: str = "t1",
) -> "Node[Any, Any]":
    return Node[DocData, Any](  # type: ignore[valid-type]
        id=uuid4(),
        type="doc",
        name=title,
        created_by=uuid4(),
        data=DocData(title=title),
        tenant_id=tenant_id,
    )


async def _pull_n(
    it: AsyncGenerator[LogEntry, None], n: int
) -> list[LogEntry]:
    """Pull exactly ``n`` entries from an async generator (bounded; never hangs)."""
    results: list[LogEntry] = []
    for _ in range(n):
        results.append(await anext(it))
    return results


# --- catch-up-then-live -----------------------------------------------------
def test_catch_up_then_live() -> None:
    asyncio.run(_case_catch_up_then_live())


async def _case_catch_up_then_live() -> None:
    rt = Runtime()
    _reg, DocData = _registry()

    # Append two entries before starting observe.
    node_a = _make_node(DocData, title="a")
    node_b = _make_node(DocData, title="b")
    await rt.create(node_a)
    await rt.create(node_b)

    # Start the observe iterator (backlog = 2 entries).
    it = rt.observe(since=None)

    # Pull the 2-entry backlog.
    backlog = await _pull_n(it, 2)
    assert len(backlog) == 2
    assert {e.entity_id for e in backlog} == {node_a.id, node_b.id}

    # Append one more entry live, then pull it from the stream.
    node_c = _make_node(DocData, title="c")
    await rt.create(node_c)
    live_entry = await anext(it)
    assert live_entry.entity_id == node_c.id

    await it.aclose()


# --- resume from arbitrary cursor ------------------------------------------
def test_resume_from_arbitrary_cursor() -> None:
    asyncio.run(_case_resume_from_arbitrary_cursor())


async def _case_resume_from_arbitrary_cursor() -> None:
    rt = Runtime()
    _reg, DocData = _registry()

    node_a = _make_node(DocData, title="a")
    node_b = _make_node(DocData, title="b")
    node_c = _make_node(DocData, title="c")
    await rt.create(node_a)
    await rt.create(node_b)
    await rt.create(node_c)

    # Pull the full backlog to get real (store-stamped) cursors.
    it_all = rt.observe(since=None)
    all_entries = await _pull_n(it_all, 3)
    await it_all.aclose()
    cursor_a = all_entries[0].cursor  # real store-assigned cursor for node_a

    # Resume after cursor_a — should only see b and c.
    it = rt.observe(since=cursor_a)
    entries = await _pull_n(it, 2)
    await it.aclose()

    assert len(entries) == 2
    ids = {e.entity_id for e in entries}
    assert node_a.id not in ids
    assert node_b.id in ids
    assert node_c.id in ids


# --- total order stable -----------------------------------------------------
def test_total_order_stable() -> None:
    asyncio.run(_case_total_order_stable())


async def _case_total_order_stable() -> None:
    rt = Runtime()
    _reg, DocData = _registry()

    nodes = [_make_node(DocData, title=f"n{i}") for i in range(5)]
    for node in nodes:
        await rt.create(node)

    it = rt.observe(since=None)
    entries = await _pull_n(it, 5)
    await it.aclose()

    # Cursor tokens are integer offsets; they must be strictly increasing.
    tokens = [int(e.cursor.token) for e in entries]
    assert tokens == sorted(tokens)
    assert tokens == list(range(len(tokens)))  # 0,1,2,3,4


# --- observe is global across tenants ---------------------------------------
def test_observe_is_global_across_tenants() -> None:
    asyncio.run(_case_observe_is_global_across_tenants())


async def _case_observe_is_global_across_tenants() -> None:
    """Writes under two tenants both appear in one observe stream.

    This documents the intentional global semantics: ``observe`` has no
    tenant_id filter so projections (graph/vectorstore/reconciler) can
    rebuild from a single stream and apply their own tenant logic internally.
    """
    rt = Runtime()
    _reg, DocData = _registry()

    node_a = _make_node(DocData, title="tenant-A", tenant_id="A")
    node_b = _make_node(DocData, title="tenant-B", tenant_id="B")
    await rt.create(node_a)
    await rt.create(node_b)

    it = rt.observe(since=None)
    entries = await _pull_n(it, 2)
    await it.aclose()

    tenant_ids = {e.tenant_id for e in entries}
    assert "A" in tenant_ids
    assert "B" in tenant_ids
