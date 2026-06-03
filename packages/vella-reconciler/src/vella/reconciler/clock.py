"""Clock seam: the injected time source the driver and handlers share.

The reconciler owns no clock of its own. :class:`Clock` is the abstract time
seam; :class:`ManualClock` is the deterministic, driver-controlled implementation
that makes the no-``pytest-asyncio`` test idiom possible — tests advance time by
hand and observe the resulting wakeups in a reproducible order.

The full deterministic ``advance()`` ordering contract is implemented in
milestone M2; this module declares the public seam so the surface tripwire can
baseline it from M1.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Injected monotonic time source for the reconciler.

    The driver reads the current time via :meth:`now` and parks (resync ticks,
    backoff waits) via :meth:`sleep`. Production wiring supplies a real clock;
    tests supply :class:`ManualClock` so every wake is deterministic.
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


class ManualClock:
    """Deterministic :class:`Clock` driven by the test/driver harness.

    Time only advances when :meth:`advance` is called, and sleepers wake in a
    fully specified order. This is a supported public testing seam — the
    no-``pytest-asyncio`` idiom relies on it — so it is part of the package's
    public surface. The deterministic ``advance()`` ordering and the ``sleep()``
    waiter machinery are implemented in milestone M2.
    """

    def __init__(self) -> None:
        """Create a manual clock positioned at time zero with no waiters."""
        self._now: float = 0.0

    def now(self) -> float:
        """Return the current manual time.

        Returns:
            The current time, advanced only by :meth:`advance`.
        """
        return self._now

    async def sleep(self, delay: float) -> None:
        """Park the caller until the clock advances past ``delay`` (M2).

        Args:
            delay: Seconds of clock time to wait before resuming.

        Raises:
            NotImplementedError: The waiter machinery lands in milestone M2.
        """
        raise NotImplementedError("ManualClock.sleep lands in M2")
