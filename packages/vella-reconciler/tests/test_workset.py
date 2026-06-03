"""Work-set fold invariants (M3).

The fold consumes a stream of real :class:`vella.runtime.LogEntry` values and
maintains a work-set, a dedup work-queue, a monotonic integer high-water, and a
backlog-drained :class:`asyncio.Event`. These tests pin:

1. folding N entries for the SAME key dedups to exactly ONE queue item (TRAP-3);
2. distinct keys produce distinct queue items;
3. the high-water advances by exactly one per entry, INCLUDING ``observe_only``;
4. ``observe_only`` entries do NOT enter the work-set and do NOT enqueue (the
   self-re-enqueue guard);
5. the backlog-drained Event is set only after the known backlog is consumed (per
   :func:`vella.reconciler.workset.fold_available`'s documented contract);
6. the fold reads only typed top-level ``LogEntry`` fields — never ``.payload``
   (enforced by a payload that raises on item access).

Fixtures construct real ``LogEntry`` objects via the public constructor. Async is
driven by ``asyncio.run`` + a bounded ``asyncio.wait_for`` backstop and an injected
:class:`~vella.reconciler.ManualClock`; no ``pytest-asyncio``.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from vella.core import utcnow
from vella.runtime import Cursor, LogEntry, TransitionKind

from vella.reconciler import ManualClock
from vella.reconciler.workset import WorkSet, fold_available

_offset = 0


def _entry(
    *,
    tenant_id: str = "t1",
    entity_id: UUID,
    version: int,
    transition: TransitionKind,
) -> LogEntry:
    """Build a real ``LogEntry`` via its public constructor (monotonic cursor)."""
    global _offset
    cursor = Cursor(token=str(_offset))
    _offset += 1
    return LogEntry(
        cursor=cursor,
        tenant_id=tenant_id,
        entity_kind="node",
        entity_id=entity_id,
        version=version,
        transition=transition,
        payload={"unused": "the fold must never read this"},
        recorded_at=utcnow(),
    )


def _drive(coro: object) -> None:
    """Run ``coro`` under a bounded backstop so a hung test fails fast."""
    asyncio.run(asyncio.wait_for(coro, timeout=1.0))  # type: ignore[arg-type]


def test_same_key_dedups_to_one_queue_item() -> None:
    """N edits to one key enqueue exactly ONCE (TRAP-3 / O(entities))."""
    ws = WorkSet()
    eid = uuid4()
    returned = [
        ws.apply(_entry(entity_id=eid, version=v, transition="edit"))
        for v in range(5)
    ]

    # Only the first apply newly-enqueues; the rest are deduped (return None).
    key = ("t1", eid)
    assert returned[0] == key
    assert all(r is None for r in returned[1:])
    assert ws.queue_depth() == 1

    # Exactly one pop yields the key; the queue is then empty.
    assert ws.pop() == key
    assert ws.pop() is None


def test_distinct_keys_produce_distinct_items() -> None:
    """Different keys each enqueue once, in order."""
    ws = WorkSet()
    a, b, c = uuid4(), uuid4(), uuid4()
    for eid in (a, b, c):
        ws.apply(_entry(entity_id=eid, version=1, transition="create"))

    assert ws.queue_depth() == 3
    popped = [ws.pop(), ws.pop(), ws.pop()]
    assert popped == [("t1", a), ("t1", b), ("t1", c)]
    assert ws.pop() is None


def test_same_entity_id_distinct_tenants_are_distinct_keys() -> None:
    """The key is (tenant_id, entity_id): same id under two tenants is two keys."""
    ws = WorkSet()
    eid = uuid4()
    ws.apply(_entry(tenant_id="A", entity_id=eid, version=1, transition="create"))
    ws.apply(_entry(tenant_id="B", entity_id=eid, version=1, transition="create"))
    assert ws.queue_depth() == 2
    assert {ws.pop(), ws.pop()} == {("A", eid), ("B", eid)}


def test_high_water_advances_by_exactly_one_per_entry() -> None:
    """The monotonic counter increments by one per entry, observe_only included."""
    ws = WorkSet()
    eid = uuid4()
    assert ws.high_water == 0

    ws.apply(_entry(entity_id=eid, version=1, transition="create"))
    assert ws.high_water == 1

    # An observe_only entry still advances the high-water (it was drained).
    ws.apply(_entry(entity_id=eid, version=1, transition="observe_only"))
    assert ws.high_water == 2

    ws.apply(_entry(entity_id=eid, version=2, transition="edit"))
    assert ws.high_water == 3


def test_observe_only_does_not_enter_workset_or_enqueue() -> None:
    """observe_only is skipped from the work-set and queue (self-re-enqueue guard)."""
    ws = WorkSet()
    eid = uuid4()
    key = ("t1", eid)

    result = ws.apply(_entry(entity_id=eid, version=7, transition="observe_only"))
    assert result is None
    assert ws.queue_depth() == 0
    assert ws.version(key) is None  # never entered the work-set
    assert ws.high_water == 1  # but it WAS counted


def test_workset_records_last_seen_version() -> None:
    """State-changing entries upsert the key's version (a staleness index)."""
    ws = WorkSet()
    eid = uuid4()
    key = ("t1", eid)
    ws.apply(_entry(entity_id=eid, version=1, transition="create"))
    ws.apply(_entry(entity_id=eid, version=4, transition="edit"))
    assert ws.version(key) == 4


def test_delete_is_recorded_like_any_state_change() -> None:
    """delete is folded normally — no payload read, no special-casing in the fold."""
    ws = WorkSet()
    eid = uuid4()
    result = ws.apply(_entry(entity_id=eid, version=2, transition="delete"))
    assert result == ("t1", eid)
    assert ws.version(("t1", eid)) == 2
    assert ws.queue_depth() == 1


def test_fold_never_reads_payload() -> None:
    """The fold reads only typed top-level fields — a poisoned payload proves it.

    ``LogEntry.payload`` is a plain dict; we wrap it in a mapping that raises on
    ANY item access, so if the fold ever did ``entry.payload[...]`` (or iterated
    it) the apply would raise. It does not — the fold only reads ``transition``,
    ``tenant_id``, ``entity_id``, ``version``.
    """

    class _ExplodingPayload(dict[str, Any]):
        def __getitem__(self, key: object) -> Any:
            raise AssertionError("the fold must never read entry.payload")

        def __iter__(self) -> Any:
            raise AssertionError("the fold must never iterate entry.payload")

    eid = uuid4()
    # Construct via the public constructor, then validate-assign the poisoned
    # payload (the field is `dict[str, Any]`; the subclass satisfies it).
    entry = _entry(entity_id=eid, version=1, transition="edit")
    poisoned = entry.model_copy(update={"payload": _ExplodingPayload()})

    ws = WorkSet()
    assert ws.apply(poisoned) == ("t1", eid)  # no payload access -> no raise


# --- backlog-drained contract ----------------------------------------------
def test_backlog_drained_set_only_after_backlog_consumed() -> None:
    """The Event fires only once the known backlog is folded, not before."""
    _drive(_case_backlog_drained())


async def _case_backlog_drained() -> None:
    ws = WorkSet()
    _clock = ManualClock()  # the injected deterministic seam (no pytest-asyncio)
    a, b = uuid4(), uuid4()
    backlog = [
        _entry(entity_id=a, version=1, transition="create"),
        _entry(entity_id=b, version=1, transition="create"),
    ]

    async def finite_stream() -> AsyncIterator[LogEntry]:
        for e in backlog:
            yield e

    assert not ws.backlog_drained.is_set()  # unset before draining
    await fold_available(ws, finite_stream())

    # Both backlog entries folded, high-water reflects them, THEN the Event fired.
    assert ws.high_water == 2
    assert ws.queue_depth() == 2
    assert ws.backlog_drained.is_set()


def test_backlog_drained_sets_at_live_edge_with_pending_stream() -> None:
    """A stream that blocks at the live edge still marks caught-up after the backlog.

    The stream yields two entries, then parks forever (the live edge). The contract
    is that ``fold_available`` folds the two available entries and sets the Event
    the first time a pull would block — it must NOT hang waiting for a third entry.
    """
    _drive(_case_live_edge())


async def _case_live_edge() -> None:
    ws = WorkSet()
    a, b = uuid4(), uuid4()
    blocked: asyncio.Event = asyncio.Event()  # never set: simulates the live edge

    async def live_stream() -> AsyncIterator[LogEntry]:
        yield _entry(entity_id=a, version=1, transition="create")
        yield _entry(entity_id=b, version=1, transition="edit")
        await blocked.wait()  # park at the live edge, like InMemoryStore.observe
        raise AssertionError("unreachable: the live edge never unblocks here")

    await fold_available(ws, live_stream())

    assert ws.high_water == 2
    assert ws.queue_depth() == 2
    assert ws.backlog_drained.is_set()


def test_backlog_drained_with_empty_backlog() -> None:
    """An empty backlog marks caught-up immediately with a zero high-water."""
    _drive(_case_empty_backlog())


async def _case_empty_backlog() -> None:
    ws = WorkSet()
    blocked: asyncio.Event = asyncio.Event()

    async def empty_then_live() -> AsyncIterator[LogEntry]:
        await blocked.wait()  # park at the live edge with an empty backlog
        yield _entry(entity_id=uuid4(), version=1, transition="create")  # unreached

    await fold_available(ws, empty_then_live())
    assert ws.high_water == 0
    assert ws.queue_depth() == 0
    assert ws.backlog_drained.is_set()
