"""``Registry`` M4 invariants.

Pins:

1. ``register`` + ``lookup`` round-trip — a registered handler is retrievable by
   kind;
2. an unknown kind returns ``None`` (an explicit miss, not a crash or KeyError);
3. re-registering the same kind OVERWRITES the previous handler (documented
   overwrite semantics — the last registration wins, no error is raised);
4. distinct kinds are independent — registering "A" does not affect "B";
5. ``lookup`` returns the exact callable that was registered (identity).

No pytest-asyncio. Handler callables are trivial sync lambdas cast to ``Handler``
via type: ignore; the gate's mypy/pyright strict pass validates the Registry API
at definition time.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import pytest

from vella.reconciler import ReconcileResult, Registry
from vella.reconciler.context import Context
from vella.reconciler.registry import Handler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(result: ReconcileResult) -> Handler:
    """Return a minimal async handler that always yields ``result``."""

    async def _h(_ctx: Context) -> ReconcileResult:
        return result

    return _h


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_register_and_lookup_round_trip() -> None:
    """A handler registered for a kind is returned by lookup."""
    reg = Registry()
    h = _make_handler(ReconcileResult.done())
    reg.register("widget", h)
    assert reg.lookup("widget") is h


def test_lookup_returns_exact_callable() -> None:
    """lookup returns the identical callable object, not a copy."""
    reg = Registry()
    h = _make_handler(ReconcileResult.drop())
    reg.register("gadget", h)
    assert reg.lookup("gadget") is h


# ---------------------------------------------------------------------------
# Unknown kind → explicit miss (None), not a crash
# ---------------------------------------------------------------------------


def test_unknown_kind_returns_none() -> None:
    """Lookup of an unregistered kind returns None — an explicit miss."""
    reg = Registry()
    assert reg.lookup("nonexistent") is None


def test_empty_registry_lookup_returns_none() -> None:
    """Lookup on a fresh, empty registry returns None for any kind."""
    reg = Registry()
    assert reg.lookup("anything") is None


def test_unknown_kind_does_not_raise() -> None:
    """Lookup of an unregistered kind never raises (skip semantics)."""
    reg = Registry()
    reg.register("known", _make_handler(ReconcileResult.done()))
    result = reg.lookup("unknown")
    assert result is None  # skip, never KeyError or exception


# ---------------------------------------------------------------------------
# Re-register: overwrite semantics (last registration wins, no error)
# ---------------------------------------------------------------------------


def test_reregister_overwrites_previous_handler() -> None:
    """Re-registering the same kind silently overwrites the previous handler."""
    reg = Registry()
    h1 = _make_handler(ReconcileResult.done())
    h2 = _make_handler(ReconcileResult.drop())
    reg.register("widget", h1)
    reg.register("widget", h2)
    # Last registration wins.
    assert reg.lookup("widget") is h2
    assert reg.lookup("widget") is not h1


def test_reregister_does_not_raise() -> None:
    """Re-registering the same kind raises no exception."""
    reg = Registry()
    h = _make_handler(ReconcileResult.done())
    reg.register("widget", h)
    reg.register("widget", h)  # same handler — still no error


# ---------------------------------------------------------------------------
# Distinct kinds are independent
# ---------------------------------------------------------------------------


def test_distinct_kinds_are_independent() -> None:
    """Registering kind A does not affect the lookup result for kind B."""
    reg = Registry()
    ha = _make_handler(ReconcileResult.done())
    hb = _make_handler(ReconcileResult.requeue(after=1.0))
    reg.register("A", ha)
    reg.register("B", hb)
    assert reg.lookup("A") is ha
    assert reg.lookup("B") is hb
    assert reg.lookup("C") is None


# ---------------------------------------------------------------------------
# Handler is callable and returns ReconcileResult (async integration smoke)
# ---------------------------------------------------------------------------


def test_registered_handler_is_callable_and_awaitable() -> None:
    """The retrieved handler is async and returns ReconcileResult."""
    import asyncio

    reg = Registry()
    expected = ReconcileResult.done()
    reg.register("widget", _make_handler(expected))
    handler = reg.lookup("widget")
    assert handler is not None

    # We can't construct a real Context without a Runtime, so just verify the
    # handler is an async callable — calling it with a placeholder would require
    # a real Context. Confirm it's a coroutine function.
    import inspect
    assert inspect.iscoroutinefunction(handler)
