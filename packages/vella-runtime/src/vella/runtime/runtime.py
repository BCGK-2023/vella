"""The ``Runtime`` facade — the async write action-contract over a ``Store``.

Where ``Store``/``StoreTxn`` are the raw persistence boundary, ``Runtime`` is the
front door consumers use to move graph state forward: ``create`` / ``edit`` /
``set_desired`` / ``upsert`` / ``delete`` / ``link`` / ``unlink`` plus the
``emit_telemetry`` side-channel. Each verb validates through a core door
(``Node``/``Edge`` construction, ``hydrate``, ``evolve``, ``update_desired``),
opens a transactional scope, and appends exactly one ``LogEntry``.

Two invariants the verbs uphold, which the fold/replay path depends on:

* **Version consistency.** Each entity's own ``version`` field and the
  ``LogEntry.version`` token agree after every write. On ``create`` both equal the
  entity's create version; on ``edit``/``set_desired`` both equal
  ``expected_version + 1`` (stamped onto the entity via ``evolve(version=...)``).
  Replaying the log via ``hydrate`` therefore reconstructs an entity whose
  ``version`` matches the state-table row.
* **Timestamp fidelity.** Core's ``evolve`` never touches ``created_at`` /
  ``updated_at``; the ``LogEntry`` payload captures the post-write snapshot, so
  replay reproduces the exact stored timestamps rather than minting new ones.

The ``LogEntry.payload`` is always model-instance fields
(``{k: getattr(entity, k) for k in type(entity).model_fields}``) — typed objects
``hydrate`` can consume directly, never ``model_dump(mode="python")``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union, cast
from uuid import UUID

from vella.core import (
    Edge,
    EdgeTypes,
    Node,
    UnresolvedRef,
    utcnow,
)

from ._inmemory import InMemoryStore
from .errors import ConcurrencyConflict
from .log import Cursor, LogEntry, TransitionKind
from .store import Store

_Entity = Union["Node[Any, Any]", "Edge[Any, Any]"]
"""A core ``Node`` or ``Edge`` — the two entity kinds the runtime records."""

_PENDING = Cursor(token="pending")
"""Placeholder cursor on a freshly-built entry; the store re-stamps the real one."""


def _payload(entity: _Entity) -> dict[str, Any]:
    """Model-instance fields of ``entity`` — the in-memory ``LogEntry`` payload.

    Returns ``{k: getattr(entity, k) for k in type(entity).model_fields}``: real
    typed objects (UUID, datetime, nested models) that core's ``hydrate`` door
    consumes directly. NEVER ``model_dump(mode="python")`` (which dict-ifies
    nested models and breaks ``hydrate``). Core fields keep core's own order.
    """
    return {k: getattr(entity, k) for k in type(entity).model_fields}


def _entry(
    entity: _Entity,
    *,
    entity_kind: Literal["node", "edge"],
    transition: TransitionKind,
    version: int,
) -> LogEntry:
    """Build a single ``LogEntry`` for ``entity`` (the store re-stamps the cursor).

    ``version`` is the post-transition concurrency token; it MUST equal the
    entity's own ``version`` field for every non-telemetry write (the
    version-consistency invariant).
    """
    return LogEntry(
        cursor=_PENDING,
        tenant_id=entity.tenant_id,
        entity_kind=entity_kind,
        entity_id=entity.id,
        version=version,
        transition=transition,
        payload=_payload(entity),
        recorded_at=utcnow(),
    )


def _reconstruct(row: LogEntry) -> _Entity:
    """Rebuild the typed entity from a state-table row's payload via ``hydrate``.

    Branches on ``row.entity_kind`` so an edit/set_desired/delete reads back the
    same kind it wrote. ``hydrate`` is the trusted fast path: the payload already
    holds validated, typed objects.
    """
    # ``hydrate`` returns ``Self`` over erased generics: pyright --strict widens
    # the unparametrized args to ``BaseModel`` (reportUnknownVariableType) without
    # the cast, while mypy infers the precise type and would call the cast
    # redundant — the two checkers disagree, so the cast carries a mypy-side
    # ignore. This keeps both type gates green over the same line.
    if row.entity_kind == "edge":
        return cast("Edge[Any, Any]", Edge.hydrate(**row.payload))  # type: ignore[redundant-cast]
    return cast("Node[Any, Any]", Node.hydrate(**row.payload))  # type: ignore[redundant-cast]


class Runtime:
    """Async write action-contract over a ``Store``.

    Holds an injectable ``Store`` (defaulting to the in-memory reference adapter)
    and exposes the transition verbs. Every verb requires ``tenant_id`` and never
    crosses tenants; ``edit``/``set_desired`` enforce optimistic concurrency via
    ``expected_version`` (raising ``ConcurrencyConflict`` on mismatch).
    """

    def __init__(self, store: Optional[Store] = None) -> None:
        """Bind a ``Store`` (defaulting to a fresh ``InMemoryStore``)."""
        self._store: Store = store if store is not None else InMemoryStore()

    @property
    def store(self) -> Store:
        """The underlying ``Store`` this runtime writes through (read-only)."""
        return self._store

    async def create(self, entity: _Entity) -> LogEntry:
        """Record the creation of an already-validated ``Node`` or ``Edge``.

        The entity is built through a core door (``Node(...)`` / ``Edge(...)`` /
        ``parse_node``) by the caller; the runtime appends a ``create`` entry at
        the entity's own ``version`` (core's create default is ``1``). Returns the
        appended entry (with its store-assigned cursor).
        """
        kind: Literal["node", "edge"] = "edge" if isinstance(entity, Edge) else "node"
        entry = _entry(
            entity, entity_kind=kind, transition="create", version=entity.version
        )
        return await self._append_one(entry)

    async def edit(
        self,
        tenant_id: str,
        entity_id: UUID,
        expected_version: int,
        **updates: Any,
    ) -> LogEntry:
        """Optimistically edit an entity's fields, bumping its version by one.

        Reads the current state-table row, reconstructs the entity via
        ``hydrate``, applies ``updates`` through core's re-validating ``evolve``
        (stamping ``version=expected_version + 1`` so the entity's version and the
        ``LogEntry.version`` agree), and appends with ``expected_version`` — the
        adapter raises ``ConcurrencyConflict`` on a stale version.
        """
        async with self._store.transaction() as txn:
            row = await txn.get(tenant_id, entity_id)
            if row is None:
                raise ConcurrencyConflict(
                    f"cannot edit absent entity {entity_id} in tenant {tenant_id!r}."
                )
            new_version = expected_version + 1
            entity = _reconstruct(row)
            evolved = entity.evolve(**updates, version=new_version)
            entry = _entry(
                evolved,
                entity_kind=row.entity_kind,
                transition="edit",
                version=new_version,
            )
            await txn.append([entry], expected_version=expected_version)
            return entry

    async def set_desired(
        self,
        tenant_id: str,
        entity_id: UUID,
        expected_version: int,
        **partial: Any,
    ) -> LogEntry:
        """Optimistically merge ``partial`` into an actuator's desired state.

        Reads the current row, reconstructs via ``hydrate``, applies core's
        ``update_desired`` (a declarative, level-triggered target merge), then
        stamps ``version=expected_version + 1`` via ``evolve`` so the entity and
        ``LogEntry`` versions agree. Appends with ``expected_version``; the
        adapter raises ``ConcurrencyConflict`` on a stale version.
        """
        async with self._store.transaction() as txn:
            row = await txn.get(tenant_id, entity_id)
            if row is None:
                raise ConcurrencyConflict(
                    f"cannot set_desired on absent entity {entity_id} in tenant "
                    f"{tenant_id!r}."
                )
            new_version = expected_version + 1
            entity = _reconstruct(row)
            evolved = entity.update_desired(**partial).evolve(version=new_version)
            entry = _entry(
                evolved,
                entity_kind=row.entity_kind,
                transition="set_desired",
                version=new_version,
            )
            await txn.append([entry], expected_version=expected_version)
            return entry

    async def upsert(
        self,
        tenant_id: str,
        plugin: str,
        external_id: str,
        entity: "Node[Any, Any]",
        **updates: Any,
    ) -> LogEntry:
        """Idempotent find-or-create on ``(tenant_id, plugin, external_id)``.

        Inside one transaction (the lock serializes the find-or-create, so a
        concurrent second upsert sees the first's node — no ``expected_version``):

        * No existing binding -> append the supplied ``entity`` as an ``upsert``
          at its own ``version``.
        * Existing binding -> reconstruct it, apply ``updates`` via ``evolve``
          (bumping its version by one), and append the new revision.
        """
        async with self._store.transaction() as txn:
            existing = await txn.find_by_binding(tenant_id, plugin, external_id)
            if existing is None:
                entry = _entry(
                    entity,
                    entity_kind="node",
                    transition="upsert",
                    version=entity.version,
                )
            else:
                current = _reconstruct(existing)
                new_version = existing.version + 1
                evolved = current.evolve(**updates, version=new_version)
                entry = _entry(
                    evolved,
                    entity_kind="node",
                    transition="upsert",
                    version=new_version,
                )
            await txn.append([entry])
            return entry

    async def delete(self, tenant_id: str, entity_id: UUID) -> LogEntry:
        """Tombstone an entity: ``get`` returns ``None``, history keeps the trail.

        Appends a ``delete`` entry whose payload is the last-known entity snapshot
        (so history stays replayable) at the entity's current version; the adapter
        marks the state-table row deleted. Core has no delete concept — the
        tombstone is runtime-side state-table metadata, not a model field.
        """
        async with self._store.transaction() as txn:
            row = await txn.get(tenant_id, entity_id)
            if row is None:
                raise ConcurrencyConflict(
                    f"cannot delete absent entity {entity_id} in tenant "
                    f"{tenant_id!r}."
                )
            entity = _reconstruct(row)
            entry = _entry(
                entity,
                entity_kind=row.entity_kind,
                transition="delete",
                version=row.version,
            )
            await txn.append([entry])
            return entry

    async def link(
        self,
        tenant_id: str,
        from_node_id: UUID,
        to_node_id: UUID,
        edge_type: str = EdgeTypes.REFERENCES,
        created_by: Optional[Union[UUID, UnresolvedRef]] = None,
        **fields: Any,
    ) -> LogEntry:
        """Create a typed, directed edge between two nodes.

        Builds the ``Edge`` through core's validated constructor and appends a
        ``link`` entry at the edge's own ``version``. ``created_by`` defaults to an
        ``UnresolvedRef`` (``vella:runtime``) when the caller supplies none.
        """
        edge: Edge[Any, Any] = Edge(
            type=edge_type,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            created_by=created_by or UnresolvedRef(identifier="vella:runtime"),
            tenant_id=tenant_id,
            **fields,
        )
        entry = _entry(
            edge, entity_kind="edge", transition="link", version=edge.version
        )
        return await self._append_one(entry)

    async def unlink(self, tenant_id: str, edge_id: UUID) -> LogEntry:
        """Retire an edge by tombstoning it (history keeps the ``unlink`` trail).

        Reads the edge's current state-table row, reconstructs it, and appends an
        ``unlink`` entry whose payload is the last-known edge snapshot; the adapter
        marks the row deleted so a later ``get`` returns ``None``.
        """
        async with self._store.transaction() as txn:
            row = await txn.get(tenant_id, edge_id)
            if row is None:
                raise ConcurrencyConflict(
                    f"cannot unlink absent edge {edge_id} in tenant {tenant_id!r}."
                )
            edge = _reconstruct(row)
            entry = _entry(
                edge, entity_kind="edge", transition="unlink", version=row.version
            )
            await txn.append([entry])
            return entry

    async def emit_telemetry(
        self, tenant_id: str, entity_id: UUID, payload: dict[str, Any]
    ) -> LogEntry:
        """Emit an ``observe_only`` telemetry entry — no version bump, no RMW.

        Telemetry has no read-modify-write: the entry reaches the log and live
        observers but never touches the state-table or bumps a version. It is
        appended through a trivial transaction that only appends (there is no
        ``Store.append``; "outside a transaction scope" means no RMW, not no
        transaction object). The entry's ``version`` carries the entity's current
        version unchanged, read once inside the same scope.
        """
        async with self._store.transaction() as txn:
            row = await txn.get(tenant_id, entity_id)
            current_version = row.version if row is not None else 0
            entity_kind = row.entity_kind if row is not None else "node"
            entry = LogEntry(
                cursor=_PENDING,
                tenant_id=tenant_id,
                entity_kind=entity_kind,
                entity_id=entity_id,
                version=current_version,
                transition="observe_only",
                payload=payload,
                recorded_at=utcnow(),
            )
            await txn.append([entry])
            return entry

    async def _append_one(self, entry: LogEntry) -> LogEntry:
        """Append a single entry through a fresh transaction; return it.

        The transaction lock keeps the append atomic with respect to other
        scopes. Used by the non-RMW verbs (``create``/``link``) that build their
        entity outside any prior read.
        """
        async with self._store.transaction() as txn:
            await txn.append([entry])
        return entry


__all__ = ["Runtime"]
