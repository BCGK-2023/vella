"""Multi-tenant non-leak invariant (M5), property-style.

Tenancy isolation is a load-bearing security property: a write in tenant A must
never become visible to tenant B through ``get`` or ``history`` (keyed on
``(tenant_id, kind, id)``). A Hypothesis strategy interleaves writes across two
tenants over the SAME entity ids (the adversarial case — id collisions across
tenants must still not leak), then asserts each tenant only ever sees its own
entities.

``observe`` is deliberately NOT asserted isolated: it is the single global
stream every projection replays and filters internally (see ``Runtime.observe``
docstring). Asserting tenant isolation there would contradict the design.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from vella.core import Node, Registry, VellaModel, node_type

from vella.runtime import Runtime
from vella.runtime._inmemory import InMemoryStore


def _registry() -> tuple[Registry, type]:
    reg = Registry()

    @node_type("doc", registry=reg)
    class DocData(VellaModel):
        title: str

    return reg, DocData


def _make_node(
    DocData: type, *, title: str, tenant_id: str, node_id: UUID
) -> "Node[Any, Any]":
    return Node[DocData, Any](  # type: ignore[valid-type]
        id=node_id,
        type="doc",
        name=title,
        created_by=uuid4(),
        data=DocData(title=title),
        tenant_id=tenant_id,
    )


# Each step: which tenant writes, and which of a small shared id pool it targets.
_steps = st.lists(
    st.tuples(st.sampled_from(["t1", "t2"]), st.integers(min_value=0, max_value=2)),
    min_size=1,
    max_size=14,
)


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(steps=_steps)
def test_interleaved_writes_never_leak_across_tenants(
    steps: list[tuple[str, int]],
) -> None:
    """Interleaved A/B writes (even on shared ids) never cross-leak."""
    reg, DocData = _registry()
    rt = Runtime()
    store = rt.store
    assert isinstance(store, InMemoryStore)

    # A small shared id pool, so the two tenants contend on the SAME ids.
    pool = [uuid4(), uuid4(), uuid4()]
    written: dict[str, set[UUID]] = {"t1": set(), "t2": set()}

    async def _drive() -> None:
        for tenant, idx in steps:
            nid = pool[idx]
            if nid in written[tenant]:
                # Already created in this tenant — edit it (version is 1 -> 2 ...).
                row = await store.get(tenant, nid)
                assert row is not None
                await rt.edit(
                    tenant, nid, expected_version=row.version, name=f"{tenant}-{idx}"
                )
            else:
                node = _make_node(
                    DocData, title=f"{tenant}-{idx}", tenant_id=tenant, node_id=nid
                )
                await rt.create(node)
                written[tenant].add(nid)

    asyncio.run(_drive())

    async def _assert() -> None:
        for tenant, others in (("t1", "t2"), ("t2", "t1")):
            for nid in pool:
                row = await store.get(tenant, nid)
                if nid in written[tenant]:
                    assert row is not None
                    assert row.tenant_id == tenant  # never the other tenant's row
                    hist = await store.history(tenant, nid)
                    assert hist  # this tenant has a history
                    assert all(e.tenant_id == tenant for e in hist)
                else:
                    # Not written in this tenant: get is None, history empty —
                    # even if the OTHER tenant wrote the same id.
                    assert row is None
                    assert await store.history(tenant, nid) == []

    asyncio.run(_assert())


def test_cross_tenant_get_and_history_empty() -> None:
    """Concrete anchor: t1's entity is invisible to t2's get and history."""

    async def _case() -> None:
        reg, DocData = _registry()
        rt = Runtime()
        store = rt.store
        assert isinstance(store, InMemoryStore)
        nid = uuid4()
        node = _make_node(DocData, title="secret", tenant_id="t1", node_id=nid)
        await rt.create(node)

        assert await store.get("t2", nid) is None
        assert await store.history("t2", nid) == []
        assert await store.get("t1", nid) is not None
        assert len(await store.history("t1", nid)) == 1

    asyncio.run(_case())
