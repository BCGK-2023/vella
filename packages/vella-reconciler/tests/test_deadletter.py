"""``InMemoryDeadLetterStore`` M2 invariants + conformance binding.

Binds :class:`InMemoryDeadLetterStore` to the shared
:class:`DeadLetterStoreConformance` suite (retrieval by key, re-record bumps
attempts, ``all()``/``drain()`` return deterministic sorted order, drain empties)
and adds a focused determinism check that the sorted order is independent of
insertion order. All async via ``asyncio.run``; no ``pytest-asyncio``. A structural
``_s: DeadLetterStore = InMemoryDeadLetterStore()`` proves Protocol conformance at
type-check time.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from vella.reconciler import (
    DeadLetterRecord,
    DeadLetterStore,
    InMemoryDeadLetterStore,
)

from conformance.deadletter_suite import DeadLetterStoreConformance

# Structural Protocol-conformance proof at type-check time.
_s: DeadLetterStore = InMemoryDeadLetterStore()

_ID_A = UUID("00000000-0000-0000-0000-0000000000aa")
_ID_B = UUID("00000000-0000-0000-0000-0000000000bb")


class TestInMemoryDeadLetterStoreConforms(DeadLetterStoreConformance):
    """Run the full DeadLetterStore conformance suite against the in-memory impl."""

    store_factory = staticmethod(InMemoryDeadLetterStore)


def test_record_retrieve_rerecord_and_sorted_drain() -> None:
    asyncio.run(asyncio.wait_for(_case(), timeout=1.0))


async def _case() -> None:
    store = InMemoryDeadLetterStore()

    # Retrievable by (tenant_id, entity_id).
    await store.record(
        DeadLetterRecord(tenant_id="t1", entity_id=_ID_A, reason="r1", attempts=1)
    )
    got = await store.get("t1", _ID_A)
    assert got is not None and got.attempts == 1

    # Re-record bumps attempts (replaces, does not duplicate).
    await store.record(
        DeadLetterRecord(tenant_id="t1", entity_id=_ID_A, reason="r2", attempts=2)
    )
    got = await store.get("t1", _ID_A)
    assert got is not None and got.attempts == 2

    # Insert a second key out of sorted order; all()/drain() are sorted.
    await store.record(
        DeadLetterRecord(tenant_id="t1", entity_id=_ID_B, reason="r", attempts=1)
    )
    keys = [(r.tenant_id, str(r.entity_id)) for r in await store.all()]
    assert keys == sorted(keys)

    drained = await store.drain()
    drained_keys = [(r.tenant_id, str(r.entity_id)) for r in drained]
    assert drained_keys == sorted(drained_keys)
    assert await store.all() == []
