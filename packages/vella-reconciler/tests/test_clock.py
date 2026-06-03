"""``ManualClock`` M2 invariants + conformance binding.

Binds :class:`ManualClock` to the shared :class:`ClockConformance` suite (the
ordering contract any deterministic clock must satisfy) and adds the M2-specific
invariants spelled out in the plan:

1. two sleepers scheduled for the SAME wake time resolve in INSERTION order;
2. a later-scheduled-but-earlier-wake sleeper resolves FIRST;
3. after ``advance()`` returns, all woken coroutines have observably progressed —
   asserted via a list of side-effect markers recorded in fire order.

All async is driven by ``asyncio.run`` + a bounded ``asyncio.wait_for`` backstop;
no ``pytest-asyncio``. A structural ``_c: Clock = ManualClock()`` assignment makes
mypy/pyright prove Protocol conformance at definition time.
"""

from __future__ import annotations

import asyncio

from vella.reconciler import Clock, ManualClock
from vella.reconciler.clock import SystemClock

from conformance.clock_suite import ClockConformance

# Structural Protocol-conformance proof at type-check time (mirrors runtime).
_c: Clock = ManualClock()
_s: Clock = SystemClock()


class TestManualClockConforms(ClockConformance):
    """Run the full Clock conformance suite against ``ManualClock``."""

    clock_factory = staticmethod(ManualClock)


def _drive(coro: object) -> None:
    """Run ``coro`` to completion under a bounded backstop (never hangs)."""
    asyncio.run(asyncio.wait_for(coro, timeout=1.0))  # type: ignore[arg-type]


def test_same_wake_time_resolves_in_insertion_order() -> None:
    _drive(_case_insertion_order())


async def _case_insertion_order() -> None:
    clock = ManualClock()
    fired: list[str] = []

    async def sleeper(label: str) -> None:
        await clock.sleep(1.0)
        fired.append(label)

    # Insertion order: a, b, c — all at wake time 1.0.
    tasks = [asyncio.ensure_future(sleeper(label)) for label in ("a", "b", "c")]
    await asyncio.sleep(0)  # let waiters register
    await clock.advance(1.0)
    assert fired == ["a", "b", "c"]
    await asyncio.gather(*tasks)


def test_earlier_wake_resolves_before_later() -> None:
    _drive(_case_earlier_wake_first())


async def _case_earlier_wake_first() -> None:
    clock = ManualClock()
    fired: list[str] = []

    async def sleeper(label: str, delay: float) -> None:
        await clock.sleep(delay)
        fired.append(label)

    # "late" registered FIRST (insertion seq 0) but wakes at 5.0; "early"
    # registered SECOND (seq 1) but wakes at 1.0. By wake time, early fires first,
    # proving the primary sort is wake-time, not insertion order.
    t_late = asyncio.ensure_future(sleeper("late", 5.0))
    t_early = asyncio.ensure_future(sleeper("early", 1.0))
    await asyncio.sleep(0)
    await clock.advance(5.0)
    assert fired == ["early", "late"]
    await asyncio.gather(t_late, t_early)


def test_woken_coroutines_progressed_when_advance_returns() -> None:
    _drive(_case_observable_progress())


async def _case_observable_progress() -> None:
    clock = ManualClock()
    fired: list[str] = []

    async def sleeper(label: str, delay: float) -> None:
        await clock.sleep(delay)
        fired.append(label)

    tasks = [
        asyncio.ensure_future(sleeper("first", 1.0)),
        asyncio.ensure_future(sleeper("second", 2.0)),
    ]
    await asyncio.sleep(0)

    # After advancing to 1.0, ONLY "first" is due and has observably fired the
    # moment advance() returns — no extra awaits needed by the caller.
    await clock.advance(1.0)
    assert fired == ["first"]

    # Advancing the rest wakes "second"; observable immediately on return.
    await clock.advance(1.0)
    assert fired == ["first", "second"]
    await asyncio.gather(*tasks)
