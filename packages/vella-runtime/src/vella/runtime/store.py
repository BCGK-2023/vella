"""The async ``Store`` persistence boundary and its transactional scope.

Two Protocols, structurally typed (no inheritance required of adapters):

* ``Store`` — the persistence boundary. Read-only verbs (``get``, ``history``,
  ``find_by_binding``, ``observe``) are callable directly; read-modify-write
  goes through ``transaction()``.
* ``StoreTxn`` — the atomic scope opened by ``transaction()``.

The in-memory adapter (``_inmemory.InMemoryStore``) is the v0.1 reference
implementation; SQLite/Postgres adapters added later pass the SAME conformance
suite unchanged. Because the Protocols are structural, an adapter satisfies them
by shape alone — verified by ``mypy --strict`` via an explicit assignment in the
conformance binding.
"""

from __future__ import annotations

from typing import (
    AsyncContextManager,
    AsyncIterator,
    Optional,
    Protocol,
    Sequence,
)
from uuid import UUID

from .log import Cursor, LogEntry


class StoreTxn(Protocol):
    """Transactional scope for read-modify-write atomicity.

    Opened via ``Store.transaction()``. All reads and appends within the scope
    are atomic — the in-memory adapter serializes via ``asyncio.Lock``; a SQL
    adapter maps to ``BEGIN ... COMMIT``. Verbs that do read-modify-write
    (edit, set_desired, upsert) MUST use this scope.

    Optimistic concurrency: ``edit``/``set_desired`` pass ``expected_version``
    to ``append``; the adapter checks ``version == expected`` inside the scope
    before appending. On mismatch: raise ``ConcurrencyConflict``. ``upsert``
    uses the lock alone (no version check) — it passes no ``expected_version``.
    """

    async def get(self, tenant_id: str, entity_id: UUID) -> Optional[LogEntry]:
        """Latest state-table row within this transaction's snapshot."""
        ...

    async def find_by_binding(
        self, tenant_id: str, plugin: str, external_id: str
    ) -> Optional[LogEntry]:
        """Idempotency lookup for upsert on ``(tenant_id, plugin, external_id)``."""
        ...

    async def append(
        self,
        entries: Sequence[LogEntry],
        *,
        expected_version: Optional[int] = None,
    ) -> Cursor:
        """Atomically append entries (in order), update the derived state-table.

        Also notifies live observers and returns the new high-water ``Cursor``.

        When ``expected_version`` is given (the optimistic-concurrency path for
        ``edit``/``set_desired``), the adapter asserts the current state-table
        version of the first entry's entity equals it BEFORE appending, raising
        ``ConcurrencyConflict`` on mismatch. ``upsert`` omits it (lock-only).

        For ``observe_only`` entries: appends to the log and notifies observers
        but does NOT update the state-table or bump version.
        """
        ...


class Store(Protocol):
    """Persistence boundary; the in-memory adapter is the v0.1 reference impl.

    SQLite/Postgres adapters later pass the SAME conformance suite unchanged.

    Read-modify-write verbs use ``transaction()`` for atomicity. Read-only
    operations (get, history, find_by_binding, observe) can be called directly
    on the Store.

    The single ``append`` method (on ``StoreTxn``) handles all transition kinds
    including ``observe_only`` (telemetry) — there is no separate
    ``append_telemetry``. The adapter branches internally on
    ``entry.transition``.
    """

    def transaction(self) -> AsyncContextManager[StoreTxn]:
        """Open a transactional scope for atomic read-modify-write operations.

        In-memory: ``asyncio.Lock``. SQL: ``BEGIN ... COMMIT``.
        """
        ...

    async def get(self, tenant_id: str, entity_id: UUID) -> Optional[LogEntry]:
        """Return the latest state-table row for ``(tenant_id, entity_id)``, or None.

        Returns None for deleted entities. Never crosses tenants.
        """
        ...

    async def history(
        self, tenant_id: str, entity_id: UUID
    ) -> Sequence[LogEntry]:
        """Return all log entries for one entity, in version/offset order.

        Includes delete transitions.
        """
        ...

    async def find_by_binding(
        self, tenant_id: str, plugin: str, external_id: str
    ) -> Optional[LogEntry]:
        """Idempotency lookup for upsert on ``(tenant_id, plugin, external_id)``.

        Available both on Store (read-only) and StoreTxn (transactional).
        """
        ...

    def observe(self, since: Optional[Cursor] = None) -> AsyncIterator[LogEntry]:
        """Replay-from-offset, then continue live in total, stable order.

        Drains the historical slice after ``since`` (or from the start when
        None), then continues live. Includes ``observe_only`` (telemetry)
        entries. No acks / consumer-groups / redelivery (deferred Non-Goal).

        Expressible by: ``SELECT ... WHERE offset > $1 ORDER BY offset`` +
        LISTEN/NOTIFY (Postgres); in-memory: ordered-list slice + asyncio queue.
        """
        ...


__all__ = ["Store", "StoreTxn"]
