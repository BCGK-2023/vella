"""Adapter-agnostic conformance suite for the ``Store``/``StoreTxn`` Protocols.

This module IS the contract. The in-memory adapter must satisfy it today; a
future SQLite/Postgres adapter runs it UNCHANGED. Bind an adapter by subclassing
``StoreConformance`` and supplying a ``store_factory`` (see
``tests/test_inmemory_conforms.py``).

No async test plugin is required (the runtime keeps its dependency surface
minimal): each case is an ``async def`` coroutine driven by a sync wrapper via
``asyncio.run`` on a freshly-built store, so every case is fully isolated.

Determinism rules enforced by these cases:
* In-memory ``LogEntry.payload`` holds actual model-instance fields
  (``{k: getattr(entity, k) for k in type(entity).model_fields}``) — typed
  objects, NOT ``model_dump(mode="python")`` (which dict-ifies nested models and
  breaks ``hydrate``).
* ALL entity comparisons use ``model_dump(mode="json")``, NEVER Python ``==``
  (core's ``_vella_registry`` PrivateAttr participates in ``__eq__``).
* Runtime never re-sorts core model fields (``integrations`` etc. are
  order-semantic).
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Callable, Literal, cast
from uuid import UUID, uuid4

from vella.core import (
    IntegrationBinding,
    Node,
    Registry,
    VellaModel,
    node_type,
    parse_edge,
    parse_node,
    utcnow,
)
from vella.core import Edge as CoreEdge

from vella.runtime import ConcurrencyConflict, Cursor, LogEntry, Store, TransitionKind

EntityKind = Literal["node", "edge"]


# --- Test entity type (registered into an isolated Registry) -----------------
def _registry() -> tuple[Registry, type]:
    """Build an isolated registry with one ``doc`` node type (never the global)."""
    reg = Registry()

    @node_type("doc", registry=reg)
    class DocData(VellaModel):
        title: str

    return reg, DocData


def _payload(entity: Any) -> dict[str, Any]:
    """Model-instance fields — the in-memory payload shape (typed, not dict-ified)."""
    return {k: getattr(entity, k) for k in type(entity).model_fields}


def _entry(
    entity: Any,
    *,
    entity_kind: EntityKind,
    transition: TransitionKind,
    version: int,
) -> LogEntry:
    """Build a ``LogEntry`` for ``entity`` (cursor is re-stamped by the store)."""
    return LogEntry(
        cursor=Cursor(token="pending"),
        tenant_id=entity.tenant_id,
        entity_kind=entity_kind,
        entity_id=entity.id,
        version=version,
        transition=transition,
        payload=_payload(entity),
        recorded_at=utcnow(),
    )


def _make_node(
    DocData: type,
    *,
    title: str = "hello",
    tenant_id: str = "t1",
    integrations: list[IntegrationBinding] | None = None,
    node_id: UUID | None = None,
    version: int = 1,
) -> "Node[Any, Any]":
    return Node[DocData, Any](  # type: ignore[valid-type]
        id=node_id or uuid4(),
        type="doc",
        name=title,
        created_by=uuid4(),
        data=DocData(title=title),
        tenant_id=tenant_id,
        version=version,
        integrations=integrations or [],
    )


class StoreConformance:
    """Conformance cases; a subclass supplies ``store_factory`` to bind an adapter.

    Each ``test_*`` is a thin sync wrapper that builds a fresh store and runs the
    corresponding ``async def _case_*`` under ``asyncio.run`` — giving full
    per-case isolation with no async test plugin.
    """

    store_factory: Callable[[], Store]

    def _run(self, case: Callable[[Store], Any]) -> None:
        store = type(self).store_factory()
        asyncio.run(case(store))

    # ---- basic contract anchors ---------------------------------------------
    def test_append_get_roundtrip(self) -> None:
        self._run(self._case_append_get_roundtrip)

    async def _case_append_get_roundtrip(self, store: Store) -> None:
        _reg, DocData = _registry()
        node = _make_node(DocData)
        async with store.transaction() as txn:
            cursor = await txn.append(
                [_entry(node, entity_kind="node", transition="create", version=1)]
            )
        assert isinstance(cursor, Cursor)

        got = await store.get(node.tenant_id, node.id)
        assert got is not None
        assert got.version == 1
        replayed: Node[Any, Any] = Node.hydrate(**got.payload)
        assert replayed.model_dump(mode="json") == node.model_dump(mode="json")

    def test_history_in_order(self) -> None:
        self._run(self._case_history_in_order)

    async def _case_history_in_order(self, store: Store) -> None:
        _reg, DocData = _registry()
        node = _make_node(DocData, title="v1")
        async with store.transaction() as txn:
            await txn.append(
                [_entry(node, entity_kind="node", transition="create", version=1)]
            )
        edited = node.evolve(name="v2")
        async with store.transaction() as txn:
            await txn.append(
                [_entry(edited, entity_kind="node", transition="edit", version=2)],
                expected_version=1,
            )

        hist = await store.history(node.tenant_id, node.id)
        assert [e.version for e in hist] == [1, 2]
        assert [e.transition for e in hist] == ["create", "edit"]

    # ---- concurrency races --------------------------------------------------
    def test_concurrent_upsert_race(self) -> None:
        self._run(self._case_concurrent_upsert_race)

    async def _case_concurrent_upsert_race(self, store: Store) -> None:
        _reg, DocData = _registry()
        binding = IntegrationBinding(plugin="wordpress", external_id="post-7")

        async def upsert() -> UUID:
            async with store.transaction() as txn:
                existing = await txn.find_by_binding("t1", "wordpress", "post-7")
                if existing is not None:
                    return existing.entity_id
                node = _make_node(DocData, integrations=[binding])
                await txn.append(
                    [_entry(node, entity_kind="node", transition="upsert", version=1)]
                )
                return node.id

        ids = await asyncio.gather(upsert(), upsert())
        assert ids[0] == ids[1]  # exactly one entity — the second saw the first

        hist = await store.history("t1", ids[0])
        creates = [e for e in hist if e.transition == "upsert"]
        assert len(creates) == 1  # only one transaction actually created

    def test_concurrent_edit_race(self) -> None:
        self._run(self._case_concurrent_edit_race)

    async def _case_concurrent_edit_race(self, store: Store) -> None:
        _reg, DocData = _registry()
        node = _make_node(DocData, title="base")
        async with store.transaction() as txn:
            await txn.append(
                [_entry(node, entity_kind="node", transition="create", version=1)]
            )

        # Both readers capture the SAME base version (v1) BEFORE either writes —
        # the classic optimistic-concurrency race. The transaction then contends
        # on append: the first commits v2, the second's stale expected_version=1
        # no longer matches the now-current v2 and must raise.
        snapshot = await store.get("t1", node.id)
        assert snapshot is not None
        base = snapshot.version
        updated: Node[Any, Any] = Node.hydrate(**snapshot.payload)

        async def edit(new_title: str) -> str:
            async with store.transaction() as txn:
                evolved = updated.evolve(name=new_title)
                await txn.append(
                    [_entry(evolved, entity_kind="node", transition="edit", version=base + 1)],
                    expected_version=base,
                )
                return "ok"

        results = await asyncio.gather(edit("a"), edit("b"), return_exceptions=True)

        oks = [r for r in results if r == "ok"]
        conflicts = [r for r in results if isinstance(r, ConcurrencyConflict)]
        assert len(oks) == 1
        assert len(conflicts) == 1

        final = await store.get("t1", node.id)
        assert final is not None and final.version == 2

    # ---- telemetry (observe_only) -------------------------------------------
    def test_observe_only_telemetry(self) -> None:
        self._run(self._case_observe_only_telemetry)

    async def _case_observe_only_telemetry(self, store: Store) -> None:
        _reg, DocData = _registry()
        node = _make_node(DocData, title="state-v1")
        async with store.transaction() as txn:
            await txn.append(
                [_entry(node, entity_kind="node", transition="create", version=1)]
            )

        pre = await store.get("t1", node.id)
        assert pre is not None and pre.version == 1

        # Live observer attached before telemetry is emitted.
        stream = cast("AsyncGenerator[LogEntry, None]", store.observe(since=pre.cursor))

        telemetry = _entry(node, entity_kind="node", transition="observe_only", version=1)
        async with store.transaction() as txn:
            await txn.append([telemetry])

        # Delivered to the live observer.
        delivered = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        assert delivered.transition == "observe_only"
        await stream.aclose()

        # Appears in the log / history.
        hist = await store.history("t1", node.id)
        assert any(e.transition == "observe_only" for e in hist)

        # State table unchanged: same pre-telemetry row, version NOT bumped.
        post = await store.get("t1", node.id)
        assert post is not None
        assert post.version == 1
        assert post.transition == "create"

    # ---- portable replay (JSON-mode reconstruction via parse_*) -------------
    def test_portable_replay(self) -> None:
        self._run(self._case_portable_replay)

    async def _case_portable_replay(self, store: Store) -> None:
        reg, DocData = _registry()
        node = _make_node(DocData, title="portable")
        node_json = node.model_dump(mode="json")
        reconstructed = parse_node(node_json, registry=reg)
        assert reconstructed.model_dump(mode="json") == node_json

        edge = CoreEdge[DocData, Any](  # type: ignore[valid-type]
            type="references",
            from_node_id=uuid4(),
            to_node_id=uuid4(),
            created_by=uuid4(),
            data=DocData(title="link"),
            tenant_id="t1",
        )
        edge_json = edge.model_dump(mode="json")
        reconstructed_edge = parse_edge(edge_json, registry=reg)
        assert reconstructed_edge.model_dump(mode="json") == edge_json

    # ---- list-field order preservation through write->log->replay -----------
    def test_list_field_order_preserved(self) -> None:
        self._run(self._case_list_field_order_preserved)

    async def _case_list_field_order_preserved(self, store: Store) -> None:
        _reg, DocData = _registry()
        integrations = [
            IntegrationBinding(
                plugin="wordpress", external_id="a", contributes_to=["state", "data", "embedding"]
            ),
            IntegrationBinding(plugin="ga4", external_id="b", contributes_to=["embedding", "data"]),
            IntegrationBinding(plugin="hubspot", external_id="c"),
        ]
        node = _make_node(DocData, integrations=integrations)
        async with store.transaction() as txn:
            await txn.append(
                [_entry(node, entity_kind="node", transition="create", version=1)]
            )

        got = await store.get("t1", node.id)
        assert got is not None
        replayed: Node[Any, Any] = Node.hydrate(**got.payload)
        assert replayed.model_dump(mode="json") == node.model_dump(mode="json")

        # Explicit order assertions on the order-semantic list fields.
        dumped = replayed.model_dump(mode="json")
        assert [b["plugin"] for b in dumped["integrations"]] == ["wordpress", "ga4", "hubspot"]
        assert dumped["integrations"][0]["contributes_to"] == ["state", "data", "embedding"]
        assert dumped["integrations"][1]["contributes_to"] == ["embedding", "data"]

    # ---- observe: catch-up-then-live ----------------------------------------
    def test_observe_catch_up_then_live(self) -> None:
        self._run(self._case_observe_catch_up_then_live)

    async def _case_observe_catch_up_then_live(self, store: Store) -> None:
        _reg, DocData = _registry()
        n1 = _make_node(DocData, title="first")
        async with store.transaction() as txn:
            await txn.append(
                [_entry(n1, entity_kind="node", transition="create", version=1)]
            )

        # Observe from the start: must replay the historical entry, then go live.
        stream = cast("AsyncGenerator[LogEntry, None]", store.observe(since=None))
        caught_up = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        assert caught_up.entity_id == n1.id

        n2 = _make_node(DocData, title="second")
        async with store.transaction() as txn:
            await txn.append(
                [_entry(n2, entity_kind="node", transition="create", version=1)]
            )

        live = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        assert live.entity_id == n2.id
        await stream.aclose()
