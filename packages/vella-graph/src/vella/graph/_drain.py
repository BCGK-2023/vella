"""The generic bounded-drain race kernel (internal mechanism, M2).

``runtime.observe()`` is catch-up-then-live and BLOCKS at the live edge (the
in-memory store parks on ``await queue.get()``), so "caught up" can NEVER be "the
``async for`` ended" — it never ends (TRAP-2). Every consumer of ``observe()`` in
this package (the cold fold M2, pull ``refresh()`` M5, the background follower M6)
must therefore drain the backlog to the live edge WITHOUT blocking, then stop.

This module extracts ONLY the race kernel — the zero-delay ``anext`` probe — from
the reconciler's ``workset.fold_available`` (``workset.py:281-326``), generically:
it knows nothing of a work-set, a queue, or an :class:`asyncio.Event`. The caller
supplies two callbacks — ``sink`` (fold one entry) and ``on_caught_up`` (the live
edge was reached / the stream was exhausted) — and owns all coupling. The follower
(M6) reuses this verbatim under cancellation, so the ``fetch`` lifecycle is wrapped
in a cancellation-robust ``finally`` that drives the in-flight pull to ``done()``
for certain (a leaked ``fetch`` keeps the generator's ``__anext__`` running, which
both leaks the task and makes the only safe ``aclose()`` race a still-running
generator — the reconciler's "aclose(): already running" defect).
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

from vella.runtime import LogEntry


async def bounded_drain(
    stream: AsyncIterator[LogEntry],
    *,
    sink: Callable[[LogEntry], None],
    on_caught_up: Callable[[], None],
) -> None:
    """Drain every immediately-available entry into ``sink``, then signal caught-up.

    Pulls entries that are available WITHOUT blocking on the live edge — racing
    each ``stream.__anext__()`` against a single event-loop yield — passing each to
    ``sink``. The first time the yield wins (no entry was immediately available: the
    live edge is reached) it calls ``on_caught_up`` and returns, leaving the stream
    open for a later resume. A finite/exhausted stream (``StopAsyncIteration``) is
    likewise "fully drained": it calls ``on_caught_up`` and returns cleanly.

    The in-flight ``fetch`` task is driven to ``done()`` in a cancellation-robust
    ``finally`` on every path — the normal live-edge return AND a ``CancelledError``
    thrown into the bare yield — so no pull leaks past this frame (the follower in
    M6 relies on this for clean ``aclose()`` teardown).

    Args:
        stream: The ``runtime.observe(since)`` async iterator to drain.
        sink: Called once per drained entry, in stream order. Pure side effect; its
            return value is ignored.
        on_caught_up: Called exactly once when the live edge is reached or the
            stream is exhausted, immediately before returning.
    """
    while True:
        # ``fetch`` holds ``stream.__anext__()`` in flight. It MUST reach ``done()``
        # before this frame unwinds — including when THIS coroutine is cancelled at
        # the bare yield below, BEFORE the explicit cancel path. The ``finally``
        # guarantees that.
        fetch: asyncio.Task[LogEntry] = asyncio.ensure_future(_anext(stream))
        try:
            # A bare yield: if ``fetch`` already has an entry buffered it resolves on
            # this turn; otherwise ``fetch`` parks on the live edge and the yield wins.
            await asyncio.sleep(0)
            if fetch.done():
                try:
                    entry = fetch.result()
                except StopAsyncIteration:
                    # Stream exhausted (e.g. a finite test stream): the backlog is
                    # fully drained — signal caught-up and stop.
                    on_caught_up()
                    return
                sink(entry)
                continue
            # The live edge: no entry was immediately available. The parked pull is
            # cancelled (in the ``finally``); signal caught-up and hand the stream
            # back open.
            on_caught_up()
            return
        finally:
            # Drive ``fetch`` to ``done()`` unconditionally — on the normal live-edge
            # return AND on a ``CancelledError`` thrown into the ``await`` above. The
            # re-await is cancellation-robust: an outer cancellation re-thrown into us
            # must not abandon ``fetch`` still-pending. ``fetch`` is ``cancel()``-ed,
            # so this terminates promptly; the terminal
            # ``CancelledError``/``StopAsyncIteration`` is swallowed either way.
            fetch.cancel()
            while not fetch.done():
                try:
                    await fetch
                except (asyncio.CancelledError, StopAsyncIteration):
                    if fetch.done():
                        break


async def _anext(stream: AsyncIterator[LogEntry]) -> LogEntry:
    """Pull the next entry from ``stream`` (a typed ``anext`` wrapper for tasks).

    Args:
        stream: The async iterator to advance.

    Returns:
        The next ``LogEntry``.
    """
    return await stream.__anext__()
