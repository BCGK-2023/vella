"""The frozen, immutable graph view (M2).

A :class:`GraphView` is the queryable snapshot a fold produces. It is frozen: a
pull ``refresh()`` (M5) returns a NEW view sharing untouched index buckets by
identity, rather than mutating in place. M2 builds the view and exposes its
high-water :class:`~vella.runtime.Cursor`; the query surface (``neighbors`` / BFS /
shortest path) lands in M3 over the same always-built topology index.

The view holds:

* the always-built :class:`~vella.graph._index.GraphIndex` (topology — identical in
  both materialization modes),
* the :data:`~vella.graph.MaterializationMode` it was folded under,
* the opaque high-water ``Cursor`` (the resume token for ``refresh``), and
* (``full`` mode only) the fold-pinned resident node/edge bodies; an LRU for
  ``lean`` hydration lands in M3.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, PrivateAttr
from vella.runtime import Cursor

from ._index import GraphIndex
from .mode import MaterializationMode


class GraphView(BaseModel):
    """A frozen, per-tenant snapshot of the graph at one log position.

    Immutable by construction (pydantic ``frozen=True``); query methods (M3) read
    the index without mutating it, and ``refresh()`` (M5) returns a new view. The
    topology index is mode-independent — only body residency differs between
    ``full`` and ``lean``.

    Attributes:
        mode: The materialization mode this view was folded under.

    Examples:
        >>> from vella.graph._index import GraphIndex
        >>> view = GraphView(index=GraphIndex(), mode="lean", high_water=None)
        >>> view.mode
        'lean'
        >>> view.high_water is None
        True
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mode: MaterializationMode

    # Internals carried as private attributes so they stay off the frozen public
    # field schema (the surface tripwire snapshots only ``mode``); tests reach them
    # through the documented accessors below.
    _index: GraphIndex = PrivateAttr()
    _high_water: Optional[Cursor] = PrivateAttr(default=None)
    _resident: dict[UUID, Any] = PrivateAttr(default_factory=lambda: {})

    def __init__(
        self,
        *,
        index: GraphIndex,
        mode: MaterializationMode,
        high_water: Optional[Cursor] = None,
        resident: Optional[dict[UUID, Any]] = None,
    ) -> None:
        """Build a frozen view over a folded index.

        Args:
            index: The always-built adjacency index (topology).
            mode: The materialization mode (residency, not results).
            high_water: The opaque high-water cursor (resume token), or ``None``
                when the folded log was empty.
            resident: ``full``-mode fold-pinned bodies (id -> ``Node``/``Edge``);
                empty / ``None`` in ``lean`` mode.
        """
        super().__init__(mode=mode)
        self._index = index
        self._high_water = high_water
        self._resident = resident if resident is not None else {}

    @property
    def high_water(self) -> Optional[Cursor]:
        """The opaque high-water :class:`~vella.runtime.Cursor` (resume token).

        Stored verbatim and passed back to ``observe(since=)`` by ``refresh`` (M5);
        never compared by value (``Cursor`` has no ordering). ``None`` when the
        folded log carried no entries.

        Returns:
            The last folded entry's cursor, or ``None``.
        """
        return self._high_water

    def _internal_index(self) -> GraphIndex:
        """Return the underlying adjacency index (documented test-only accessor).

        Exposed so M2 topology-equivalence / dangling / tenancy tests can assert on
        the folded structure before the M3 query surface exists. Not part of the
        query contract; the public methods land in M3.

        Returns:
            The view's :class:`~vella.graph._index.GraphIndex`.
        """
        return self._index

    def _resident_count(self) -> int:
        """Return the number of resident bodies held (documented test-only accessor).

        ``full`` mode holds one body per live node+edge; ``lean`` holds none. The
        mode-equivalence test asserts topology is identical while this count differs.

        Returns:
            The count of fold-pinned resident bodies.
        """
        return len(self._resident)


__all__ = ["GraphView"]
