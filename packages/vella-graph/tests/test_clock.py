"""``ManualClock`` invariants + conformance binding.

Binds :class:`ManualClock` to the shared :class:`ClockConformance` suite (the
ordering contract any deterministic clock must satisfy). A structural
``_c: Clock = ManualClock()`` / ``_s: Clock = SystemClock()`` assignment makes
mypy/pyright prove Protocol conformance at definition time (mirrors the runtime /
reconciler idiom). All async is driven by ``asyncio.run`` + a bounded
``asyncio.wait_for`` backstop; no ``pytest-asyncio``.
"""

from __future__ import annotations

import asyncio

from vella.graph import Clock, ManualClock
from vella.graph.clock import SystemClock

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


def test_systemclock_conforms_to_protocol() -> None:
    """``SystemClock`` is a structural ``Clock`` (its ``now``/``sleep`` shape)."""
    assert isinstance(SystemClock(), Clock)


def test_manualclock_conforms_to_protocol() -> None:
    """``ManualClock`` is a structural ``Clock`` (and adds the ``advance`` seam)."""
    assert isinstance(ManualClock(), Clock)


def test_systemclock_sleep_is_real_time_but_zero_returns() -> None:
    """A zero-delay ``SystemClock.sleep`` returns promptly (real ``asyncio.sleep``)."""
    _drive(SystemClock().sleep(0.0))
