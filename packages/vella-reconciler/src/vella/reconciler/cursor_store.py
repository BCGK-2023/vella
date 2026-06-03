"""Resume-cursor persistence seam.

The driver hands the runtime a ``Cursor`` via ``observe(since=...)`` to resume
from where it left off. :class:`CursorStore` is the abstract seam that persists
that cursor; :class:`InMemoryCursorStore` is the process-local v0.1 implementation.

The stored ``Cursor`` is opaque: it is round-tripped verbatim and handed back to
``observe``, never compared by value (``Cursor`` deliberately carries no ordering).
The seam is async to match the runtime's async-first contract — a future durable
store (SQL, file) does real I/O, and the driver awaits load/save inline.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from vella.runtime import Cursor


@runtime_checkable
class CursorStore(Protocol):
    """Persists the resume :class:`~vella.runtime.Cursor` for the watch task.

    The driver saves the latest observed cursor and reloads it on restart to
    resume the ``observe`` stream. Implementations treat the cursor as opaque.
    """

    async def load(self) -> Optional[Cursor]:
        """Return the persisted resume cursor, or ``None`` if none is stored.

        Returns:
            The last saved cursor, or ``None`` before anything is saved.
        """
        ...

    async def save(self, cursor: Cursor) -> None:
        """Persist ``cursor`` as the resume position, replacing any prior value.

        Args:
            cursor: The opaque cursor to store verbatim.
        """
        ...


class InMemoryCursorStore:
    """Process-local :class:`CursorStore` holding a single cursor in memory.

    The default (nothing saved yet) is ``None``. Persistence beyond process
    lifetime is an explicit post-v0.1 follow-up.
    """

    def __init__(self) -> None:
        """Create an empty store whose initial cursor is ``None``."""
        self._cursor: Optional[Cursor] = None

    async def load(self) -> Optional[Cursor]:
        """Return the in-memory cursor, or ``None`` if nothing was saved.

        Returns:
            The last saved cursor, or ``None``.
        """
        return self._cursor

    async def save(self, cursor: Cursor) -> None:
        """Replace the in-memory cursor with ``cursor``.

        Args:
            cursor: The opaque cursor to retain.
        """
        self._cursor = cursor
