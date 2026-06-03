"""Adapter-agnostic conformance suite for the :class:`~vella.reconciler.CursorStore`.

This module IS the round-trip contract any ``CursorStore`` must satisfy: the
default load is ``None``, a saved cursor round-trips verbatim (the store never
inspects or compares the opaque token), and the latest save wins. Bind an
implementation by subclassing :class:`CursorStoreConformance` and supplying a
``store_factory``; the reference :class:`InMemoryCursorStore` is bound in
``tests/test_cursor_store.py``.

No async plugin is required: each case runs under ``asyncio.run`` with a bounded
``asyncio.wait_for`` backstop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from vella.runtime import Cursor

from vella.reconciler import CursorStore


class CursorStoreConformance:
    """Conformance cases; a subclass supplies ``store_factory`` to bind a store."""

    store_factory: Callable[[], CursorStore]

    def _run(self, case: Callable[[CursorStore], Awaitable[Any]]) -> None:
        store = type(self).store_factory()
        asyncio.run(asyncio.wait_for(case(store), timeout=1.0))

    def test_default_is_none(self) -> None:
        self._run(self._case_default_is_none)

    async def _case_default_is_none(self, store: CursorStore) -> None:
        assert await store.load() is None

    def test_round_trips_verbatim(self) -> None:
        self._run(self._case_round_trips_verbatim)

    async def _case_round_trips_verbatim(self, store: CursorStore) -> None:
        cursor = Cursor(token="42")
        await store.save(cursor)
        loaded = await store.load()
        assert loaded is not None
        assert loaded.token == "42"

    def test_latest_save_wins(self) -> None:
        self._run(self._case_latest_save_wins)

    async def _case_latest_save_wins(self, store: CursorStore) -> None:
        await store.save(Cursor(token="1"))
        await store.save(Cursor(token="2"))
        loaded = await store.load()
        assert loaded is not None
        assert loaded.token == "2"
