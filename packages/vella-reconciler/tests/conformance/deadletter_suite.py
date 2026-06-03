"""Adapter-agnostic conformance suite for the :class:`~vella.reconciler.DeadLetterStore`.

This module IS the contract any ``DeadLetterStore`` must satisfy: records are
retrievable by ``(tenant_id, entity_id)``; a re-record of an existing key replaces
the prior record (so the worker can bump ``attempts``); and ``all()``/``drain()``
return records in deterministic ``sorted()`` order keyed by
``(tenant_id, str(entity_id))`` — iteration must never be hash-seed dependent.
``drain()`` empties the store.

Bind an implementation by subclassing :class:`DeadLetterStoreConformance` and
supplying a ``store_factory``; the reference :class:`InMemoryDeadLetterStore` is
bound in ``tests/test_deadletter.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable
from uuid import UUID

from vella.reconciler import DeadLetterRecord, DeadLetterStore

# Fixed UUIDs whose sorted-by-str order (id_a < id_b) is independent of the hash
# seed — the determinism property the suite pins.
_ID_A = UUID("00000000-0000-0000-0000-0000000000aa")
_ID_B = UUID("00000000-0000-0000-0000-0000000000bb")


class DeadLetterStoreConformance:
    """Conformance cases; a subclass supplies ``store_factory`` to bind a store."""

    store_factory: Callable[[], DeadLetterStore]

    def _run(self, case: Callable[[DeadLetterStore], Awaitable[Any]]) -> None:
        store = type(self).store_factory()
        asyncio.run(asyncio.wait_for(case(store), timeout=1.0))

    def test_retrievable_by_key(self) -> None:
        self._run(self._case_retrievable_by_key)

    async def _case_retrievable_by_key(self, store: DeadLetterStore) -> None:
        rec = DeadLetterRecord(
            tenant_id="t1", entity_id=_ID_A, reason="boom", attempts=3
        )
        await store.record(rec)
        got = await store.get("t1", _ID_A)
        assert got is not None
        assert got.reason == "boom"
        assert got.attempts == 3
        # A key never recorded is absent.
        assert await store.get("t1", _ID_B) is None
        # Tenant is part of the key.
        assert await store.get("other", _ID_A) is None

    def test_rerecord_replaces_and_bumps_attempts(self) -> None:
        self._run(self._case_rerecord_replaces_and_bumps_attempts)

    async def _case_rerecord_replaces_and_bumps_attempts(
        self, store: DeadLetterStore
    ) -> None:
        await store.record(
            DeadLetterRecord(tenant_id="t1", entity_id=_ID_A, reason="r1", attempts=1)
        )
        await store.record(
            DeadLetterRecord(tenant_id="t1", entity_id=_ID_A, reason="r2", attempts=2)
        )
        got = await store.get("t1", _ID_A)
        assert got is not None
        assert got.attempts == 2
        assert got.reason == "r2"
        # Re-record did not create a second entry.
        assert len(await store.all()) == 1

    def test_all_sorted_order(self) -> None:
        self._run(self._case_all_sorted_order)

    async def _case_all_sorted_order(self, store: DeadLetterStore) -> None:
        # Insert out of sorted order across distinct tenants + ids.
        await store.record(
            DeadLetterRecord(tenant_id="t2", entity_id=_ID_B, reason="x", attempts=1)
        )
        await store.record(
            DeadLetterRecord(tenant_id="t1", entity_id=_ID_B, reason="x", attempts=1)
        )
        await store.record(
            DeadLetterRecord(tenant_id="t1", entity_id=_ID_A, reason="x", attempts=1)
        )
        keys = [(r.tenant_id, str(r.entity_id)) for r in await store.all()]
        assert keys == sorted(keys)
        assert keys[0] == ("t1", str(_ID_A))

    def test_drain_returns_sorted_and_empties(self) -> None:
        self._run(self._case_drain_returns_sorted_and_empties)

    async def _case_drain_returns_sorted_and_empties(
        self, store: DeadLetterStore
    ) -> None:
        await store.record(
            DeadLetterRecord(tenant_id="t2", entity_id=_ID_A, reason="x", attempts=1)
        )
        await store.record(
            DeadLetterRecord(tenant_id="t1", entity_id=_ID_A, reason="x", attempts=1)
        )
        drained = await store.drain()
        keys = [(r.tenant_id, str(r.entity_id)) for r in drained]
        assert keys == sorted(keys)
        assert keys[0][0] == "t1"
        # Store is empty after drain.
        assert await store.all() == []
        assert await store.get("t1", _ID_A) is None
