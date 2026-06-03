"""Dead-letter seam: where keys the loop has given up on are recorded.

When a key exhausts its backoff budget the worker dead-letters it: it stops being
re-enqueued by the resync ticker until an explicit ``drain()`` re-entry.
:class:`DeadLetterStore` is the abstract seam, :class:`InMemoryDeadLetterStore`
the process-local v0.1 implementation, and :class:`DeadLetterRecord` the frozen
record it holds.
"""

from __future__ import annotations

from typing import Optional, Protocol
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


class DeadLetterStore(Protocol):
    """Records and retrieves :class:`DeadLetterRecord` entries by key.

    The resync ticker consults this store to SKIP keys already given up on, so a
    deliberate give-up is not silently defeated by the next tick.
    """

    def record(self, record: DeadLetterRecord) -> None:
        """Store ``record``, replacing any prior record for the same key.

        Args:
            record: The give-up record, keyed by ``(tenant_id, entity_id)``.
        """
        ...

    def get(self, tenant_id: str, entity_id: UUID) -> Optional[DeadLetterRecord]:
        """Return the record for a key, or ``None`` if it is not dead-lettered.

        Args:
            tenant_id: The tenant to look up under.
            entity_id: The entity to look up.

        Returns:
            The stored record, or ``None`` if the key was never given up on.
        """
        ...


class InMemoryDeadLetterStore:
    """Process-local :class:`DeadLetterStore` keyed by ``(tenant_id, entity_id)``."""

    def __init__(self) -> None:
        """Create an empty in-memory dead-letter store."""
        self._records: dict[tuple[str, UUID], DeadLetterRecord] = {}

    def record(self, record: DeadLetterRecord) -> None:
        """Store ``record`` under its ``(tenant_id, entity_id)`` key.

        Args:
            record: The give-up record to retain.
        """
        self._records[(record.tenant_id, record.entity_id)] = record

    def get(self, tenant_id: str, entity_id: UUID) -> Optional[DeadLetterRecord]:
        """Return the record for ``(tenant_id, entity_id)``, or ``None``.

        Args:
            tenant_id: The tenant to look up under.
            entity_id: The entity to look up.

        Returns:
            The stored record, or ``None``.
        """
        return self._records.get((tenant_id, entity_id))
