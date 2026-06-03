"""Adapter-agnostic conformance suite for the :class:`~vella.reconciler.Clock`.

This module IS the ordering contract any deterministic ``Clock`` must satisfy.
Bind an implementation by subclassing :class:`ClockConformance` and supplying a
``clock_factory`` returning a clock that also exposes an awaitable
``advance(dt)`` (the test-driving seam). The reference :class:`ManualClock` is
bound in ``tests/test_clock.py``.

No async plugin is required: each case is an ``async def`` coroutine driven by a
sync wrapper via ``asyncio.run``, with a bounded ``asyncio.wait_for`` backstop so
a regression manifests as a failure, never a hang.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Protocol


class _AdvanceableClock(Protocol):
    """A ``Clock`` that also exposes the deterministic ``advance`` test seam."""

    def now(self) -> float: ...

    async def sleep(self, delay: float) -> None: ...

    async def advance(self, dt: float) -> None: ...


class ClockConformance:
    """Conformance cases; a subclass supplies ``clock_factory`` to bind a clock.

    Each ``test_*`` is a thin sync wrapper that runs the corresponding
    ``async def _case_*`` under ``asyncio.run`` on a freshly-built clock, giving
    full isolation with no async-plugin dependency.
    """

    clock_factory: Callable[[], "_AdvanceableClock"]

    def _run(self, case: Callable[["_AdvanceableClock"], Awaitable[Any]]) -> None:
        clock = type(self).clock_factory()
        asyncio.run(asyncio.wait_for(case(clock), timeout=1.0))

    # --- now() advances only via advance() ----------------------------------
    def test_now_advances_only_on_advance(self) -> None:
        self._run(self._case_now_advances_only_on_advance)

    async def _case_now_advances_only_on_advance(
        self, clock: "_AdvanceableClock"
    ) -> None:
        assert clock.now() == 0.0
        await clock.advance(2.5)
        assert clock.now() == 2.5
        await clock.advance(0.5)
        assert clock.now() == 3.0

    # --- a sleeper wakes once its wake time is reached -----------------------
    def test_sleeper_wakes_at_wake_time(self) -> None:
        self._run(self._case_sleeper_wakes_at_wake_time)

    async def _case_sleeper_wakes_at_wake_time(
        self, clock: "_AdvanceableClock"
    ) -> None:
        fired: list[str] = []

        async def sleeper() -> None:
            await clock.sleep(1.0)
            fired.append("woke")

        task = asyncio.ensure_future(sleeper())
        await asyncio.sleep(0)  # let the sleeper register its waiter
        await clock.advance(0.5)  # not yet due
        assert fired == []
        await clock.advance(0.5)  # now due (total 1.0)
        assert fired == ["woke"]
        await task

    # --- same wake time resolves in insertion order --------------------------
    def test_same_wake_time_insertion_order(self) -> None:
        self._run(self._case_same_wake_time_insertion_order)

    async def _case_same_wake_time_insertion_order(
        self, clock: "_AdvanceableClock"
    ) -> None:
        fired: list[str] = []

        async def sleeper(label: str) -> None:
            await clock.sleep(1.0)
            fired.append(label)

        # Register first, second, third — all at the same wake time (1.0).
        tasks = [asyncio.ensure_future(sleeper(label)) for label in ("a", "b", "c")]
        await asyncio.sleep(0)
        await clock.advance(1.0)
        assert fired == ["a", "b", "c"]
        await asyncio.gather(*tasks)

    # --- later-scheduled-but-earlier-wake fires first ------------------------
    def test_earlier_wake_fires_first(self) -> None:
        self._run(self._case_earlier_wake_fires_first)

    async def _case_earlier_wake_fires_first(
        self, clock: "_AdvanceableClock"
    ) -> None:
        fired: list[str] = []

        async def sleeper(label: str, delay: float) -> None:
            await clock.sleep(delay)
            fired.append(label)

        # "late" is registered FIRST but wakes LATER; "early" registered second
        # but wakes sooner. By wake time, "early" must fire first.
        late = asyncio.ensure_future(sleeper("late", 5.0))
        early = asyncio.ensure_future(sleeper("early", 1.0))
        await asyncio.sleep(0)
        await clock.advance(5.0)  # both due in one advance
        assert fired == ["early", "late"]
        await asyncio.gather(late, early)
