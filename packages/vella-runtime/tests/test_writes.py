"""Write action-contract tests for the ``Runtime`` facade (M3).

Each case is an ``async def`` driven by a sync wrapper via ``asyncio.run`` on a
fresh ``Runtime`` (no async-plugin dependency, matching the conformance suite).
ALL entity equality is via ``model_dump(mode="json")``, never Python ``==``
(core's ``_vella_registry`` PrivateAttr participates in ``__eq__``).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from uuid import UUID, uuid4

import pytest
from vella.core import (
    Actuator,
    IntegrationBinding,
    Node,
    Registry,
    VellaModel,
    node_type,
)

from vella.runtime import ConcurrencyConflict, Runtime


def _run(case: Callable[[Runtime], Any]) -> None:
    """Build a fresh ``Runtime`` over a fresh in-memory store and run one case."""
    asyncio.run(case(Runtime()))


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
    integrations: list[IntegrationBinding] | None = None,
    node_id: UUID | None = None,
    state: Any | None = None,
) -> "Node[Any, Any]":
    return Node[DocData, Any](  # type: ignore[valid-type]
        id=node_id or uuid4(),
        type="doc",
        name=title,
        created_by=uuid4(),
        data=DocData(title=title),
        tenant_id=tenant_id,
        integrations=integrations or [],
        state=state,
    )


# --- create / get round-trip ------------------------------------------------
def test_create_get_roundtrip() -> None:
    _run(_case_create_get_roundtrip)


async def _case_create_get_roundtrip(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData)
    entry = await rt.create(node)
    assert entry.version == 1  # core's create default

    row = await rt.store.get(node.tenant_id, node.id)
    assert row is not None
    assert row.version == 1
    replayed: Node[Any, Any] = Node.hydrate(**row.payload)
    assert replayed.model_dump(mode="json") == node.model_dump(mode="json")


# --- edit: version bump + history -------------------------------------------
def test_edit_bumps_version_and_history() -> None:
    _run(_case_edit_bumps_version_and_history)


async def _case_edit_bumps_version_and_history(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData, title="v1")
    await rt.create(node)

    edited = await rt.edit(node.tenant_id, node.id, expected_version=1, name="v2")
    assert edited.version == 2
    # The entity's OWN version field agrees with the LogEntry version token.
    assert Node.hydrate(**edited.payload).version == 2

    row = await rt.store.get(node.tenant_id, node.id)
    assert row is not None and row.version == 2
    assert Node.hydrate(**row.payload).name == "v2"

    hist = await rt.store.history(node.tenant_id, node.id)
    assert [e.version for e in hist] == [1, 2]
    assert [e.transition for e in hist] == ["create", "edit"]
    # The prior version is reachable through history.
    prior: Node[Any, Any] = Node.hydrate(**hist[0].payload)
    assert prior.name == "v1" and prior.version == 1


# --- edit: optimistic-concurrency conflict ----------------------------------
def test_edit_wrong_version_raises_conflict() -> None:
    _run(_case_edit_wrong_version_raises_conflict)


async def _case_edit_wrong_version_raises_conflict(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData)
    await rt.create(node)
    await rt.edit(node.tenant_id, node.id, expected_version=1, name="v2")

    # Stale expected_version (1, but current is 2) -> conflict.
    with pytest.raises(ConcurrencyConflict):
        await rt.edit(node.tenant_id, node.id, expected_version=1, name="stale")


# --- upsert idempotency -----------------------------------------------------
def test_double_upsert_single_node() -> None:
    _run(_case_double_upsert_single_node)


async def _case_double_upsert_single_node(rt: Runtime) -> None:
    _reg, DocData = _registry()
    binding = IntegrationBinding(plugin="wordpress", external_id="post-7")
    node = _make_node(DocData, title="first", integrations=[binding])

    first = await rt.upsert("t1", "wordpress", "post-7", node)
    assert first.version == 1

    # Second upsert on the same binding sees the first's node (no new node).
    second = await rt.upsert(
        "t1", "wordpress", "post-7", _make_node(DocData, title="second"), name="updated"
    )
    assert second.entity_id == node.id  # same entity, found by binding
    assert second.version == 2

    hist = await rt.store.history("t1", node.id)
    upserts = [e for e in hist if e.transition == "upsert"]
    assert len(upserts) == 2  # two revisions, one node
    assert Node.hydrate(**upserts[-1].payload).name == "updated"


# --- delete tombstone -------------------------------------------------------
def test_delete_tombstones_get_returns_none() -> None:
    _run(_case_delete_tombstones_get_returns_none)


async def _case_delete_tombstones_get_returns_none(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData, title="doomed")
    await rt.create(node)
    await rt.edit(node.tenant_id, node.id, expected_version=1, name="doomed-v2")

    deleted = await rt.delete(node.tenant_id, node.id)
    assert deleted.transition == "delete"

    # get() returns None for the tombstoned entity.
    assert await rt.store.get(node.tenant_id, node.id) is None

    # history() still contains every entry, including the delete.
    hist = await rt.store.history(node.tenant_id, node.id)
    assert [e.transition for e in hist] == ["create", "edit", "delete"]
    # The delete payload preserves the last-known entity snapshot.
    assert Node.hydrate(**hist[-1].payload).name == "doomed-v2"


# --- set_desired: actuator update -------------------------------------------
def test_set_desired_updates_actuator() -> None:
    _run(_case_set_desired_updates_actuator)


async def _case_set_desired_updates_actuator(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(
        DocData, title="device", state=Actuator(current=DocData(title="off"))
    )
    await rt.create(node)

    entry = await rt.set_desired(
        node.tenant_id, node.id, expected_version=1, title="on"
    )
    assert entry.version == 2

    row = await rt.store.get(node.tenant_id, node.id)
    assert row is not None and row.version == 2
    reconstructed: Node[Any, Any] = Node.hydrate(**row.payload)
    assert reconstructed.state is not None
    assert reconstructed.state.desired.title == "on"  # type: ignore[union-attr]
    assert reconstructed.state.current.title == "off"  # type: ignore[union-attr]


# --- link: edge creation ----------------------------------------------------
def test_link_creates_edge() -> None:
    _run(_case_link_creates_edge)


async def _case_link_creates_edge(rt: Runtime) -> None:
    from_id, to_id = uuid4(), uuid4()
    entry = await rt.link("t1", from_id, to_id, edge_type="references")
    assert entry.entity_kind == "edge"
    assert entry.version == 1

    row = await rt.store.get("t1", entry.entity_id)
    assert row is not None and row.entity_kind == "edge"
    from vella.core import Edge

    edge: Edge[Any, Any] = Edge.hydrate(**row.payload)
    assert edge.from_node_id == from_id
    assert edge.to_node_id == to_id
    assert edge.type == "references"


# --- fold / version-consistency invariant -----------------------------------
def test_create_edit_version_consistency() -> None:
    _run(_case_create_edit_version_consistency)


async def _case_create_edit_version_consistency(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData, title="c")
    await rt.create(node)
    await rt.edit(node.tenant_id, node.id, expected_version=1, name="e")

    # For EVERY entry, the LogEntry.version token equals the reconstructed
    # entity's own version field — replaying the log reconstructs an entity
    # whose version matches the state-table row.
    hist = await rt.store.history(node.tenant_id, node.id)
    for entry in hist:
        reconstructed: Node[Any, Any] = Node.hydrate(**entry.payload)
        assert entry.version == reconstructed.version

    row = await rt.store.get(node.tenant_id, node.id)
    assert row is not None
    assert row.version == Node.hydrate(**row.payload).version


# --- emit_telemetry: no version bump, no state-table change -----------------
def test_emit_telemetry_no_version_bump() -> None:
    _run(_case_emit_telemetry_no_version_bump)


async def _case_emit_telemetry_no_version_bump(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData)
    await rt.create(node)

    entry = await rt.emit_telemetry(node.tenant_id, node.id, {"cpu": 0.9})
    assert entry.transition == "observe_only"
    assert entry.version == 1  # current version, unchanged

    # State table untouched: still the create row at version 1.
    row = await rt.store.get(node.tenant_id, node.id)
    assert row is not None and row.version == 1 and row.transition == "create"

    # The telemetry entry is in the log/history.
    hist = await rt.store.history(node.tenant_id, node.id)
    assert any(e.transition == "observe_only" for e in hist)


# --- tenancy isolation ------------------------------------------------------
def test_create_never_crosses_tenants() -> None:
    _run(_case_create_never_crosses_tenants)


async def _case_create_never_crosses_tenants(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData, tenant_id="t1")
    await rt.create(node)
    # A different tenant cannot see t1's entity.
    assert await rt.store.get("t2", node.id) is None
    assert await rt.store.get("t1", node.id) is not None
