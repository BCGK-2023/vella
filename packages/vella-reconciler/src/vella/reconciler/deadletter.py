"""Dead-letter seam: where keys the loop has given up on are recorded.

When a key exhausts its backoff budget the worker dead-letters it: it stops being
re-enqueued by the resync ticker until an explicit :meth:`DeadLetterStore.drain`
re-entry. :class:`DeadLetterStore` is the abstract seam,
:class:`InMemoryDeadLetterStore` the process-local v0.1 implementation, and
:class:`DeadLetterRecord` the frozen record it holds.

Determinism: :meth:`DeadLetterStore.all` and :meth:`DeadLetterStore.drain` return
records in a stable ``sorted()`` order keyed by ``(tenant_id, str(entity_id))`` —
iteration must never be hash-seed dependent (the M6 determinism artifact depends on
this). The seam is async to match the runtime's async-first contract; a future
durable store does real I/O and the driver awaits inline.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DeadLetterRecord(BaseModel):
    """A single given-up key and why the loop stopped reconciling it.

    Attributes:
        tenant_id: The tenant the entity belongs to.
        entity_id: The entity that was given up on.
        reason: A human-readable explanation (e.g. the final error).
        attempts: How many reconcile attempts were made before giving up.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    entity_id: UUID
    reason: str
    attempts: int


@runtime_checkable
class DeadLetterStore(Protocol):
    """Records and retrieves :class:`DeadLetterRecord` entries by key.

    The resync ticker consults this store (via :meth:`get`) to SKIP keys already
    given up on, so a deliberate give-up is not silently defeated by the next tick.
    :meth:`drain` is the explicit, documented re-entry path: it empties the store
    and returns the drained records so the driver can re-enqueue them.
    """

    async def record(self, record: DeadLetterRecord) -> None:
        """Store ``record``, replacing any prior record for the same key.

        Args:
            record: The give-up record, keyed by ``(tenant_id, entity_id)``.
        """
        ...

    async def get(
        self, tenant_id: str, entity_id: UUID
    ) -> Optional[DeadLetterRecord]:
        """Return the record for a key, or ``None`` if it is not dead-lettered.

        Args:
            tenant_id: The tenant to look up under.
            entity_id: The entity to look up.

        Returns:
            The stored record, or ``None`` if the key was never given up on.
        """
        ...

    async def all(self) -> Sequence[DeadLetterRecord]:
        """Return every dead-lettered record in deterministic sorted order.

        Returns:
            All records, sorted by ``(tenant_id, str(entity_id))``. The store is
            left unchanged.
        """
        ...

    async def drain(self) -> Sequence[DeadLetterRecord]:
        """Empty the store and return the drained records in sorted order.

        Returns:
            Every record the store held, sorted by ``(tenant_id, str(entity_id))``,
            after which the store is empty. This is the explicit re-entry path for
            dead-lettered keys.
        """
        ...


def _sorted(records: Sequence[DeadLetterRecord]) -> list[DeadLetterRecord]:
    """Order records deterministically by ``(tenant_id, str(entity_id))``.

    Args:
        records: The records to order.

    Returns:
        A new list in stable, hash-seed-independent order — ``UUID`` is keyed via
        ``str()`` since UUID objects are not orderable against the tuple's other
        (string) element.
    """
    return sorted(records, key=lambda r: (r.tenant_id, str(r.entity_id)))


class InMemoryDeadLetterStore:
    """Process-local :class:`DeadLetterStore` keyed by ``(tenant_id, entity_id)``.

    A re-record of an existing key replaces the prior record (the worker bumps
    ``attempts`` on the new record). :meth:`all` and :meth:`drain` return records
    in deterministic ``sorted()`` order so iteration never depends on the hash seed.
    """

    def __init__(self) -> None:
        """Create an empty in-memory dead-letter store."""
        self._records: dict[tuple[str, UUID], DeadLetterRecord] = {}

    async def record(self, record: DeadLetterRecord) -> None:
        """Store ``record`` under its ``(tenant_id, entity_id)`` key.

        Args:
            record: The give-up record to retain (replaces any prior record for
                the same key).
        """
        self._records[(record.tenant_id, record.entity_id)] = record

    async def get(
        self, tenant_id: str, entity_id: UUID
    ) -> Optional[DeadLetterRecord]:
        """Return the record for ``(tenant_id, entity_id)``, or ``None``.

        Args:
            tenant_id: The tenant to look up under.
            entity_id: The entity to look up.

        Returns:
            The stored record, or ``None``.
        """
        return self._records.get((tenant_id, entity_id))

    async def all(self) -> Sequence[DeadLetterRecord]:
        """Return every record in deterministic sorted order; leave the store intact.

        Returns:
            All records, sorted by ``(tenant_id, str(entity_id))``.
        """
        return _sorted(list(self._records.values()))

    async def drain(self) -> Sequence[DeadLetterRecord]:
        """Empty the store and return its records in deterministic sorted order.

        Returns:
            Every record the store held, sorted by ``(tenant_id, str(entity_id))``;
            the store is empty afterward.
        """
        drained = _sorted(list(self._records.values()))
        self._records.clear()
        return drained
