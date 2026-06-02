"""Read surface tests for the ``Runtime`` facade (M4).

Covers ``get`` / ``history`` at the Runtime level (not the raw Store).
Each case is an ``async def`` driven by a sync wrapper via ``asyncio.run`` on a
fresh ``Runtime`` (no async-plugin dependency, matching M2/M3 patterns).
ALL entity equality is via ``model_dump(mode="json")``, never Python ``==``
(core's ``_vella_registry`` PrivateAttr participates in ``__eq__``).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from uuid import UUID, uuid4

from vella.core import Node, VellaModel, Registry, node_type, Edge

from vella.runtime import Runtime


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
    node_id: UUID | None = None,
) -> "Node[Any, Any]":
    return Node[DocData, Any](  # type: ignore[valid-type]
        id=node_id or uuid4(),
        type="doc",
        name=title,
        created_by=uuid4(),
        data=DocData(title=title),
        tenant_id=tenant_id,
    )


# --- get: returns reconstructed entity --------------------------------------
def test_get_returns_reconstructed_entity() -> None:
    _run(_case_get_returns_reconstructed_entity)


async def _case_get_returns_reconstructed_entity(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData, title="alpha")
    await rt.create(node)

    result = await rt.get(node.tenant_id, node.id)
    assert result is not None
    assert result.model_dump(mode="json") == node.model_dump(mode="json")


# --- get: absent entity returns None ----------------------------------------
def test_get_absent_returns_none() -> None:
    _run(_case_get_absent_returns_none)


async def _case_get_absent_returns_none(rt: Runtime) -> None:
    result = await rt.get("t1", uuid4())
    assert result is None


# --- get: deleted entity returns None ---------------------------------------
def test_get_deleted_returns_none() -> None:
    _run(_case_get_deleted_returns_none)


async def _case_get_deleted_returns_none(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData, title="doomed")
    await rt.create(node)
    await rt.delete(node.tenant_id, node.id)

    result = await rt.get(node.tenant_id, node.id)
    assert result is None


# --- history: version order includes delete ---------------------------------
def test_history_in_version_order_includes_delete() -> None:
    _run(_case_history_in_version_order_includes_delete)


async def _case_history_in_version_order_includes_delete(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData, title="v1")
    await rt.create(node)
    await rt.edit(node.tenant_id, node.id, expected_version=1, name="v2")
    await rt.delete(node.tenant_id, node.id)

    hist = await rt.history(node.tenant_id, node.id)
    assert len(hist) == 3
    assert [e.version for e in hist] == [1, 2, 2]
    assert [e.transition for e in hist] == ["create", "edit", "delete"]
    assert hist[-1].transition == "delete"


# --- get: cross-tenant isolation --------------------------------------------
def test_get_cross_tenant_isolation() -> None:
    _run(_case_get_cross_tenant_isolation)


async def _case_get_cross_tenant_isolation(rt: Runtime) -> None:
    _reg, DocData = _registry()
    node = _make_node(DocData, title="secret", tenant_id="A")
    await rt.create(node)

    # Entity exists in tenant A.
    assert await rt.get("A", node.id) is not None
    # Same entity_id under tenant B is invisible.
    assert await rt.get("B", node.id) is None
