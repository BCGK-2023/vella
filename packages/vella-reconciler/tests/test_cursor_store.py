"""``InMemoryCursorStore`` M2 invariants + conformance binding.

Binds :class:`InMemoryCursorStore` to the shared :class:`CursorStoreConformance`
suite (default ``None`` + verbatim round-trip + latest-save-wins) and adds a focused
round-trip + default check. All async via ``asyncio.run``; no ``pytest-asyncio``.
A structural ``_s: CursorStore = InMemoryCursorStore()`` proves Protocol conformance
at type-check time.
"""

from __future__ import annotations

import asyncio

from vella.runtime import Cursor

from vella.reconciler import CursorStore, InMemoryCursorStore

from conformance.cursor_store_suite import CursorStoreConformance

# Structural Protocol-conformance proof at type-check time.
_s: CursorStore = InMemoryCursorStore()


class TestInMemoryCursorStoreConforms(CursorStoreConformance):
    """Run the full CursorStore conformance suite against ``InMemoryCursorStore``."""

    store_factory = staticmethod(InMemoryCursorStore)


def test_default_load_is_none_and_round_trip() -> None:
    asyncio.run(asyncio.wait_for(_case(), timeout=1.0))


async def _case() -> None:
    store = InMemoryCursorStore()
    assert await store.load() is None
    await store.save(Cursor(token="seven"))
    loaded = await store.load()
    assert loaded is not None
    assert loaded.token == "seven"
