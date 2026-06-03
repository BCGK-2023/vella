"""``Context`` M4 invariants.

Pins:

1. ``ctx.runtime`` returns the injected runtime (identity check);
2. ``ctx.clock`` returns the injected clock (identity check);
3. the work-set internals are NOT accessible via ``Context`` — no ``queue``,
   ``workset``, ``_queue``, or ``_workset`` attribute is exposed;
4. ``Context`` is read-only — the ``runtime`` and ``clock`` properties have no
   setter and assignment raises ``AttributeError``.

The test uses a ``ManualClock`` for the clock and a minimal ``Runtime`` stand-in
(a ``unittest.mock.MagicMock`` typed as ``Runtime``) since constructing a real
``Runtime`` requires a ``Store``. Type checkers see only the structural ``Runtime``
Protocol shape; this is acceptable for a runtime-surface invariant test.

No pytest-asyncio. Entirely synchronous.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vella.reconciler import ManualClock
from vella.reconciler.context import Context
from vella.runtime import Runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> tuple[Context, Runtime, ManualClock]:
    """Return a ``(ctx, runtime, clock)`` triple with a mock Runtime."""
    mock_runtime: Runtime = MagicMock(spec=Runtime)
    clock = ManualClock()
    ctx = Context(runtime=mock_runtime, clock=clock)
    return ctx, mock_runtime, clock


# ---------------------------------------------------------------------------
# Exposes runtime and clock
# ---------------------------------------------------------------------------


def test_context_exposes_runtime() -> None:
    """ctx.runtime returns the injected runtime."""
    ctx, runtime, _ = _make_context()
    assert ctx.runtime is runtime


def test_context_exposes_clock() -> None:
    """ctx.clock returns the injected clock."""
    ctx, _, clock = _make_context()
    assert ctx.clock is clock


def test_context_runtime_and_clock_are_distinct_objects() -> None:
    """The runtime and clock are different Python objects."""
    ctx, runtime, clock = _make_context()
    # Verify via identity to their respective injected references, not to each other
    # (mypy correctly notes Runtime and Clock are non-overlapping types).
    assert ctx.runtime is runtime
    assert ctx.clock is clock
    assert id(ctx.runtime) != id(ctx.clock)


# ---------------------------------------------------------------------------
# Read-only: no setters on runtime or clock
# ---------------------------------------------------------------------------


def test_runtime_property_has_no_setter() -> None:
    """Assigning to ctx.runtime raises AttributeError (no setter)."""
    ctx, _, _ = _make_context()
    with pytest.raises(AttributeError):
        ctx.runtime = MagicMock(spec=Runtime)  # type: ignore[misc]


def test_clock_property_has_no_setter() -> None:
    """Assigning to ctx.clock raises AttributeError (no setter)."""
    ctx, _, _ = _make_context()
    with pytest.raises(AttributeError):
        ctx.clock = ManualClock()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Work-set internals are not exposed
# ---------------------------------------------------------------------------


def test_no_queue_attribute() -> None:
    """Context does not expose a public 'queue' attribute."""
    ctx, _, _ = _make_context()
    assert not hasattr(ctx, "queue")


def test_no_workset_attribute() -> None:
    """Context does not expose a public 'workset' attribute."""
    ctx, _, _ = _make_context()
    assert not hasattr(ctx, "workset")


def test_no_private_queue_attribute() -> None:
    """Context does not expose a '_queue' attribute (work-set internals)."""
    ctx, _, _ = _make_context()
    assert not hasattr(ctx, "_queue")


def test_no_private_workset_attribute() -> None:
    """Context does not expose a '_workset' attribute (work-set internals)."""
    ctx, _, _ = _make_context()
    assert not hasattr(ctx, "_workset")


# ---------------------------------------------------------------------------
# Multiple independent contexts share nothing
# ---------------------------------------------------------------------------


def test_two_contexts_are_independent() -> None:
    """Two Context instances with different runtimes/clocks do not share state."""
    ctx1, rt1, clk1 = _make_context()
    ctx2, rt2, clk2 = _make_context()
    assert ctx1.runtime is not ctx2.runtime
    assert ctx1.clock is not ctx2.clock
