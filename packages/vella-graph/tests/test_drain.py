"""Bounded-drain kernel (M2): live-edge stop + StopAsyncIteration handling.

The generic ``bounded_drain`` is the single seam every ``observe()`` consumer
shares (fold M2, refresh M5, follower M6). Two behaviours are load-bearing and
asserted here directly (the fold tests cover it indirectly through a real runtime):

* a FINITE / exhausted stream raises ``StopAsyncIteration`` from ``__anext__`` —
  ``bounded_drain`` must treat that as "fully drained": sink every entry, call
  ``on_caught_up`` exactly once, return cleanly (no leak, no raise);
* a stream that PARKS at the live edge (never raising) must also stop at the live
  edge after draining what is available, calling ``on_caught_up`` once.

Driven via ``asyncio.run`` + a bounded ``wait_for`` (no pytest-asyncio); a bug that
fails to stop would hang and trip the timeout.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from vella.graph._drain import bounded_drain

from _fixtures import drive


async def _finite(n: int) -> AsyncIterator[Any]:
    """A finite async iterator that yields ``0..n-1`` then exhausts."""
    for i in range(n):
        yield i


async def _empty() -> AsyncIterator[Any]:
    """An async iterator that yields nothing and exhausts immediately."""
    return
    yield  # pragma: no cover - unreachable, makes this an async generator


def test_finite_stream_drains_then_caught_up() -> None:
    """A finite stream is fully drained; on_caught_up fires exactly once."""
    drive(_finite_case())


async def _finite_case() -> None:
    seen: list[Any] = []
    caught: list[bool] = []
    await bounded_drain(
        _finite(3),
        sink=seen.append,
        on_caught_up=lambda: caught.append(True),
    )
    assert seen == [0, 1, 2]
    assert caught == [True]


def test_empty_stream_caught_up_immediately() -> None:
    """An empty stream calls on_caught_up once and drains nothing."""
    drive(_empty_case())


async def _empty_case() -> None:
    seen: list[Any] = []
    caught: list[bool] = []
    await bounded_drain(
        _empty(),
        sink=seen.append,
        on_caught_up=lambda: caught.append(True),
    )
    assert seen == []
    assert caught == [True]


def test_parked_live_edge_stops_after_available() -> None:
    """A stream that parks (never exhausts) still stops at the live edge."""
    drive(_parked_case())


async def _parked_case() -> None:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(2):
        queue.put_nowait(i)

    async def parked() -> AsyncIterator[Any]:
        # Yields the two buffered items, then blocks forever on an empty queue —
        # exactly the shape of runtime.observe() at the live edge.
        while True:
            yield await queue.get()

    seen: list[Any] = []
    caught: list[bool] = []
    await bounded_drain(
        parked(),
        sink=seen.append,
        on_caught_up=lambda: caught.append(True),
    )
    # Drained the two available items, then stopped at the live edge (did NOT block).
    assert seen == [0, 1]
    assert caught == [True]
