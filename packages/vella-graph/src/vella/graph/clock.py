"""Clock seam: the injectable time source the follower carries (M6).

The graph projection owns no clock of its own. :class:`Clock` is the abstract time
seam; :class:`ManualClock` is the deterministic, harness-controlled implementation
that makes the no-``pytest-asyncio`` test idiom possible — a test advances time by
hand and observes the resulting wakeups in a reproducible order. :class:`SystemClock`
is the production wiring (real monotonic time + ``asyncio.sleep``); it is a concrete
impl rather than a public export — the follower injects it as the default while
:class:`ManualClock` stays the supported testing seam.

This module is **re-declared verbatim from** ``vella.reconciler.clock`` rather than
imported: the dependency direction forbids ``vella.graph`` depending on
``vella.reconciler`` (they are sibling distributions; the graph depends only on
``vella-runtime`` and ``vella-core``). The two implementations are intentionally
identical in shape — a shared ``Clock`` conformance suite (a testing-utils package)
is a documented v0.1 follow-up; until then the duplication is the accepted cost of
the dependency rule.

The deterministic ``advance()`` ordering contract is the load-bearing part: any
sleepers park on :meth:`ManualClock.sleep`, so their interleaving is only
deterministic because ``advance()`` resolves waiters sorted by scheduled wake time,
ties broken by insertion order.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Injectable monotonic time source for the background follower.

    A consumer reads the current time via :meth:`now` and parks via :meth:`sleep`.
    Production wiring supplies a real clock; tests supply :class:`ManualClock` so
    every wake is deterministic. The follower carries a clock as its injectable
    time seam even though its v0.1 loop is purely event-driven on ``observe()``'s
    live tail — see :class:`~vella.graph.GraphFollower`.
    """

    def now(self) -> float:
        """Return the current time as a monotonic float (seconds).

        Returns:
            The clock's current time. Only differences are meaningful.
        """
        ...

    async def sleep(self, delay: float) -> None:
        """Suspend the caller for ``delay`` seconds of clock time.

        Args:
            delay: Seconds of clock time to wait before resuming.
        """
        ...


class SystemClock:
    """Real-time :class:`Clock` backed by ``time.monotonic`` + ``asyncio.sleep``.

    This is the production default the follower injects when no clock is supplied.
    It is intentionally NOT part of ``vella.graph.__all__``: the supported public
    testing seam is :class:`ManualClock`, and the surface tripwire baseline freezes
    only the public exports.
    """

    def now(self) -> float:
        """Return the current monotonic time in seconds.

        Returns:
            ``time.monotonic()`` — only differences are meaningful.
        """
        return time.monotonic()

    async def sleep(self, delay: float) -> None:
        """Suspend the caller for ``delay`` real seconds via ``asyncio.sleep``.

        Args:
            delay: Seconds of wall-clock time to wait before resuming.
        """
        await asyncio.sleep(delay)


class ManualClock:
    """Deterministic :class:`Clock` driven by the test/driver harness.

    Time only advances when :meth:`advance` is called, and sleepers wake in a
    fully specified order. This is a supported public testing seam — the
    no-``pytest-asyncio`` idiom relies on it — so it is part of the package's
    public surface.

    The ordering contract: a :meth:`sleep` call registers a waiter at
    ``now + delay`` with a monotonically increasing insertion sequence.
    :meth:`advance` moves ``now`` forward, then resolves every waiter whose wake
    time is ``<= now``, **sorted by wake time with ties broken by insertion
    order**, and finally yields (``await asyncio.sleep(0)``) so the woken
    coroutines actually run before ``advance()`` returns.

    Examples:
        >>> import asyncio
        >>> async def _demo() -> float:
        ...     clock = ManualClock()
        ...     await clock.advance(2.5)
        ...     return clock.now()
        >>> asyncio.run(_demo())
        2.5
    """

    def __init__(self) -> None:
        """Create a manual clock positioned at time zero with no waiters."""
        self._now: float = 0.0
        self._seq: int = 0
        # Each waiter: its scheduled wake time, an insertion sequence number (the
        # tiebreak), and the Future the sleeper awaits. Stored as a list and sorted
        # at resolution time — the waiter count per advance() is small and the
        # ordering (wake_at, seq) is what the contract pins.
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []

    def now(self) -> float:
        """Return the current manual time.

        Returns:
            The current time, advanced only by :meth:`advance`.
        """
        return self._now

    async def sleep(self, delay: float) -> None:
        """Park the caller until :meth:`advance` reaches ``now + delay``.

        Registers a waiter at ``now + delay`` tagged with the next insertion
        sequence number (the tiebreak for same-wake-time ordering), then awaits a
        per-waiter :class:`asyncio.Future` that :meth:`advance` resolves.

        Args:
            delay: Seconds of clock time to wait before resuming.
        """
        wake_at = self._now + delay
        seq = self._seq
        self._seq += 1
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        self._waiters.append((wake_at, seq, waiter))
        await waiter

    async def advance(self, dt: float) -> None:
        """Move time forward by ``dt`` and wake every due sleeper, in order.

        Steps:
          1. move ``now`` forward by ``dt``;
          2. resolve ALL waiters whose wake time ``<= now``, sorted by
             ``(wake_at, insertion_seq)`` so same-wake-time sleepers fire in
             insertion order and an earlier wake time always fires first;
          3. ``await asyncio.sleep(0)`` so the woken coroutines observably progress
             before ``advance()`` returns.

        Args:
            dt: Seconds of clock time to advance. Must be non-negative.
        """
        self._now += dt
        due = [w for w in self._waiters if w[0] <= self._now]
        due.sort(key=lambda w: (w[0], w[1]))
        remaining = [w for w in self._waiters if w[0] > self._now]
        self._waiters = remaining
        for _wake_at, _seq, waiter in due:
            if not waiter.done():
                waiter.set_result(None)
        # Yield control so each resolved coroutine resumes (and any side effects
        # fire) before advance() returns — the woken work is observable to callers.
        await asyncio.sleep(0)
