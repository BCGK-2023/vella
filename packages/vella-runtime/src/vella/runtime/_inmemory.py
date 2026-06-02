"""In-memory reference ``Store`` adapter — the v0.1 persistence implementation.

Private module (``_``-prefixed, per core's convention; excluded from doctest
collection by the root conftest's ``collect_ignore_glob``). The public surface is
the ``Store``/``StoreTxn`` Protocols in ``store.py``; this is one impl that
satisfies them structurally.

Data structures
---------------
* ``_log`` — an ordered ``list[LogEntry]``: the global, append-only stream. List
  index == monotonically increasing offset, which is also the cursor token.
* ``_state`` — ``{(tenant_id, kind, entity_id): _StateRow}``: the derived
  state-table (a fold of the log). Each row carries the latest ``LogEntry`` plus
  a runtime-side ``deleted`` flag (OUTSIDE the core model payload — core has no
  delete concept).
* ``_bindings`` — ``{(tenant_id, plugin, external_id): entity_id}``: the
  idempotency index for upsert.
* ``_observers`` — a set of live ``asyncio.Queue`` objects, one per active
  ``observe()`` iterator, fed every appended entry for catch-up-then-live.

A single ``asyncio.Lock`` serializes the read-modify-write transactional scope:
reads and appends inside ``async with store.transaction()`` are atomic with
respect to other transactions. The optimistic-concurrency check (and the
``ConcurrencyConflict`` it raises) happens INSIDE that lock.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator, AsyncIterator, Optional, Sequence
from uuid import UUID

from .errors import ConcurrencyConflict
from .log import Cursor, LogEntry
from .store import Store

_StateKey = tuple[str, str, UUID]
_BindingKey = tuple[str, str, str]


@dataclass
class _StateRow:
    """A single derived state-table row: the latest entry plus a tombstone flag.

    ``deleted`` is runtime-side metadata living OUTSIDE the core model payload —
    core (``NodeFlags``) has no tombstone field, so a delete is recorded here,
    not by mutating the entity.
    """

    entry: LogEntry
    deleted: bool = False


@dataclass
class _Index:
    """The mutable in-memory state shared between the store and its transactions.

    Holds the ordered log, the derived state table, the binding index, and the
    set of live observer queues. The transaction operates directly on this same
    instance under the store's lock, so reads and appends within a scope see a
    consistent snapshot.
    """

    log: list[LogEntry] = field(default_factory=list[LogEntry])
    state: dict[_StateKey, _StateRow] = field(default_factory=dict[_StateKey, _StateRow])
    bindings: dict[_BindingKey, UUID] = field(default_factory=dict[_BindingKey, UUID])
    observers: set[asyncio.Queue[LogEntry]] = field(
        default_factory=set["asyncio.Queue[LogEntry]"]
    )


def _parse_token(cursor: Cursor) -> int:
    """Resolve a ``Cursor`` to this adapter's internal integer offset.

    The in-memory adapter encodes its ordering key as ``str(offset)``; this is
    the inverse. A SQL adapter would parse its own token form (BIGSERIAL, LSN,
    base64) here instead — the token is opaque to callers.
    """
    return int(cursor.token)


def _apply(index: _Index, entry: LogEntry) -> None:
    """Append one entry to the log, fold it into the state table, fan it out.

    Branches on ``entry.transition``:

    * ``observe_only`` — append to the log + notify observers ONLY. Skips the
      state-table update and does NOT track/bump version (telemetry).
    * ``delete`` — append, then mark the state-table row deleted (``get``
      returns None) while keeping the entry in history.
    * everything else — append, then upsert the state-table row.
    """
    index.log.append(entry)

    if entry.transition != "observe_only":
        key: _StateKey = (entry.tenant_id, entry.entity_kind, entry.entity_id)
        if entry.transition == "delete":
            row = index.state.get(key)
            if row is not None:
                row.entry = entry
                row.deleted = True
            else:
                index.state[key] = _StateRow(entry=entry, deleted=True)
        else:
            index.state[key] = _StateRow(entry=entry, deleted=False)

    for queue in index.observers:
        queue.put_nowait(entry)


def _latest(index: _Index, tenant_id: str, entity_id: UUID) -> Optional[LogEntry]:
    """Latest live (non-deleted) state-table entry, scanning node then edge kind."""
    for kind in ("node", "edge"):
        row = index.state.get((tenant_id, kind, entity_id))
        if row is not None:
            return None if row.deleted else row.entry
    return None


def _find_binding(
    index: _Index, tenant_id: str, plugin: str, external_id: str
) -> Optional[LogEntry]:
    """Resolve a binding key to its current live state-table entry, if any."""
    entity_id = index.bindings.get((tenant_id, plugin, external_id))
    if entity_id is None:
        return None
    return _latest(index, tenant_id, entity_id)


def _append(
    index: _Index,
    entries: Sequence[LogEntry],
    expected_version: Optional[int],
) -> Cursor:
    """Core append: version-check, assign offsets/cursors, fold, fan out.

    Shared by the transactional path and the lock-held direct telemetry append.
    Caller MUST hold the store lock. Returns the new high-water cursor.
    """
    if not entries:
        raise ValueError("append requires at least one entry")

    if expected_version is not None:
        first = entries[0]
        current = _latest(index, first.tenant_id, first.entity_id)
        current_version = current.version if current is not None else 0
        if current_version != expected_version:
            raise ConcurrencyConflict(
                f"version mismatch for entity {first.entity_id}: expected "
                f"{expected_version}, found {current_version}."
            )

    last_cursor: Optional[Cursor] = None
    for entry in entries:
        offset = len(index.log)
        stamped = entry.model_copy(update={"cursor": Cursor(token=str(offset))})
        _apply(index, stamped)
        last_cursor = stamped.cursor

    assert last_cursor is not None  # guaranteed: entries is non-empty
    return last_cursor


class _InMemoryTxn:
    """Transactional scope (a ``StoreTxn``) over the shared ``_Index``.

    Created by ``InMemoryStore.transaction()`` while the store lock is held, so
    its reads and its ``append`` are atomic with respect to other transactions.
    """

    def __init__(self, index: _Index) -> None:
        self._index = index

    async def get(self, tenant_id: str, entity_id: UUID) -> Optional[LogEntry]:
        """Latest live state-table row within this transaction's snapshot."""
        return _latest(self._index, tenant_id, entity_id)

    async def find_by_binding(
        self, tenant_id: str, plugin: str, external_id: str
    ) -> Optional[LogEntry]:
        """Idempotency lookup within this transaction's snapshot."""
        return _find_binding(self._index, tenant_id, plugin, external_id)

    async def append(
        self,
        entries: Sequence[LogEntry],
        *,
        expected_version: Optional[int] = None,
    ) -> Cursor:
        """Append atomically; optimistic-concurrency check runs under the lock.

        Registers any new ``(tenant, plugin, external_id)`` bindings carried by
        the appended entities' ``integrations`` so a later ``find_by_binding``
        (the upsert idempotency path) resolves them.
        """
        cursor = _append(self._index, entries, expected_version)
        for entry in entries:
            if entry.transition == "observe_only":
                continue
            self._register_bindings(entry)
        return cursor

    def _register_bindings(self, entry: LogEntry) -> None:
        for binding in entry.payload.get("integrations", []):
            plugin = getattr(binding, "plugin", None)
            external_id = getattr(binding, "external_id", None)
            if isinstance(plugin, str) and isinstance(external_id, str):
                self._index.bindings[(entry.tenant_id, plugin, external_id)] = (
                    entry.entity_id
                )


class InMemoryStore:
    """In-memory reference ``Store`` — ordered log + derived state + observers.

    Structurally satisfies the ``Store`` Protocol. Read-modify-write goes
    through ``transaction()`` (a single ``asyncio.Lock`` serializes scopes);
    read-only ``get``/``history``/``find_by_binding``/``observe`` are direct.
    """

    def __init__(self) -> None:
        """Initialize an empty store: fresh index and a single transaction lock."""
        self._index = _Index()
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[_InMemoryTxn, None]:
        """Open an atomic read-modify-write scope guarded by the store lock."""
        async with self._lock:
            yield _InMemoryTxn(self._index)

    async def get(self, tenant_id: str, entity_id: UUID) -> Optional[LogEntry]:
        """Latest live state-table row, or None for deleted/absent entities."""
        return _latest(self._index, tenant_id, entity_id)

    async def history(
        self, tenant_id: str, entity_id: UUID
    ) -> Sequence[LogEntry]:
        """All log entries for one entity, in offset order (includes deletes)."""
        return [
            entry
            for entry in self._index.log
            if entry.tenant_id == tenant_id and entry.entity_id == entity_id
        ]

    async def find_by_binding(
        self, tenant_id: str, plugin: str, external_id: str
    ) -> Optional[LogEntry]:
        """Idempotency lookup on ``(tenant_id, plugin, external_id)``."""
        return _find_binding(self._index, tenant_id, plugin, external_id)

    async def observe(
        self, since: Optional[Cursor] = None
    ) -> AsyncIterator[LogEntry]:
        """Drain the historical slice after ``since``, then yield live entries.

        Catch-up-then-live: a snapshot of the log after the cursor is replayed
        first (in total order), then a fresh queue — registered before the
        snapshot is taken so nothing between snapshot and registration is lost —
        delivers subsequent appends. No acks.
        """
        queue: asyncio.Queue[LogEntry] = asyncio.Queue()
        self._index.observers.add(queue)
        try:
            start = 0 if since is None else _parse_token(since) + 1
            backlog = list(self._index.log[start:])
            backlog_ids = {id(entry) for entry in backlog}
            for entry in backlog:
                yield entry
            while True:
                entry = await queue.get()
                # Skip anything already delivered from the backlog snapshot
                # (an append between registration and slicing lands in both).
                if id(entry) in backlog_ids:
                    backlog_ids.discard(id(entry))
                    continue
                yield entry
        finally:
            self._index.observers.discard(queue)


# Structural conformance: assigning the concrete adapter to a ``Store``-typed
# name forces mypy --strict / pyright to prove InMemoryStore satisfies the
# Protocol by shape. A mismatch fails the type gate here, at definition time.
_assert_conforms: Store = InMemoryStore()


__all__ = ["InMemoryStore"]
