"""The frozen, immutable graph view + deterministic query surface (M2, M3).

A :class:`GraphView` is the queryable snapshot a fold produces. It is frozen: a
pull ``refresh()`` (M5) returns a NEW view sharing untouched index buckets by
identity, rather than mutating in place. M2 built the view and exposed its
high-water :class:`~vella.runtime.Cursor`; M3 adds the query surface
(``neighbors`` / ``bfs`` / ``dfs`` / ``reachable`` / ``shortest_path``) plus
explicit and inline hydration over the same always-built topology index.

The view holds:

* the always-built :class:`~vella.graph._index.GraphIndex` (topology — identical in
  both materialization modes),
* the :data:`~vella.graph.MaterializationMode` it was folded under,
* the opaque high-water ``Cursor`` (the resume token for ``refresh``),
* the body hydrator (``full`` = fold-pinned resident map; ``lean`` = bounded LRU +
  ``get()``-on-miss), and
* a reference to the :class:`~vella.runtime.Runtime` it was folded from, used by
  ``lean`` hydration to read live bodies at query time (acceptable for v0.1 — the
  spec calls lean hydration "live via the LRU", so the view needs the runtime it
  was folded from).

Determinism: every query returns ids in a canonical order (sorted-id results,
sorted-id traversal visitation, paths by lexicographically-smallest node-id
sequence). Queries read only topology, so their id results are byte-identical
across modes; ``hydrate=True`` then attaches bodies, and ``hydrate=True`` inline is
defined to return exactly the same ids as ``hydrate=False`` followed by an explicit
``hydrate()`` of those ids.
"""

from __future__ import annotations

from typing import Any, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, PrivateAttr
from vella.runtime import Cursor, Runtime

from ._hydrate import FullHydrator, LeanHydrator
from ._index import GraphIndex
from ._query import (
    QueryDirection,
    bfs as _bfs,
    dfs as _dfs,
    neighbor_records as _neighbor_records,
    reachable as _reachable,
    shortest_path as _shortest_path,
)
from .mode import MaterializationMode
from .results import Neighbor, Path

_Hydrator = Union[FullHydrator, LeanHydrator]


class GraphView(BaseModel):
    """A frozen, per-tenant snapshot of the graph at one log position.

    Immutable by construction (pydantic ``frozen=True``); query methods read the
    index without mutating it, and ``refresh()`` (M5) returns a new view. The
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
    _tenant_id: Optional[str] = PrivateAttr(default=None)
    _runtime: Optional[Runtime] = PrivateAttr(default=None)
    _hydrator: _Hydrator = PrivateAttr()

    def __init__(
        self,
        *,
        index: GraphIndex,
        mode: MaterializationMode,
        high_water: Optional[Cursor] = None,
        resident: Optional[dict[UUID, Any]] = None,
        tenant_id: Optional[str] = None,
        runtime: Optional[Runtime] = None,
        lru_capacity: int = 1024,
    ) -> None:
        """Build a frozen view over a folded index.

        Args:
            index: The always-built adjacency index (topology).
            mode: The materialization mode (residency, not results).
            high_water: The opaque high-water cursor (resume token), or ``None``
                when the folded log was empty.
            resident: ``full``-mode fold-pinned bodies (id -> ``Node``/``Edge``);
                empty / ``None`` in ``lean`` mode.
            tenant_id: The tenant this view projects; required for ``lean`` live
                hydration. ``None`` is permitted for body-free / doctest views.
            runtime: The runtime this view was folded from; ``lean`` hydration reads
                live bodies through it at query time. ``None`` for ``full`` views or
                body-free views.
            lru_capacity: The bound for the ``lean`` hydration LRU.
        """
        super().__init__(mode=mode)
        self._index = index
        self._high_water = high_water
        self._resident = resident if resident is not None else {}
        self._tenant_id = tenant_id
        self._runtime = runtime
        self._hydrator = (
            FullHydrator(self._resident)
            if mode == "full"
            else LeanHydrator(lru_capacity)
        )

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

    async def get(self, entity_id: UUID) -> Optional[Any]:
        """Hydrate one entity body by id (``full`` = fold-pinned, ``lean`` = live).

        Args:
            entity_id: The node or edge id to hydrate.

        Returns:
            The ``Node``/``Edge`` body, or ``None`` for an absent / dangling /
            deleted id. In ``full`` mode the body is the fold-pinned snapshot; in
            ``lean`` mode it is the live body via the LRU.
        """
        return await self._hydrator.get(self._runtime, self._tenant_id_or_empty(), entity_id)

    async def hydrate(self, ids: list[UUID]) -> list[Optional[Any]]:
        """Hydrate a list of ids, preserving order (``None`` per absent id).

        Args:
            ids: The ids to hydrate, in the order the result should keep.

        Returns:
            The hydrated bodies aligned to ``ids`` (an element is ``None`` for a
            dangling / deleted id). ``full`` returns fold-pinned bodies; ``lean``
            returns live bodies via the LRU.
        """
        return [await self.get(entity_id) for entity_id in ids]

    async def neighbors(
        self,
        node: UUID,
        *,
        edge_type: Optional[str] = None,
        direction: QueryDirection = "both",
        hydrate: bool = False,
    ) -> list[Neighbor]:
        """The canonically-ordered neighbours of ``node``.

        Returns one :class:`~vella.graph.Neighbor` per incident edge (so two edges of
        different types to the same node yield two neighbours), ordered by
        ``(str(node_id), edge_type, str(edge_id))``. When ``hydrate=True`` each
        neighbour carries its hydrated body; the ids are identical to ``hydrate=False``
        (inline hydration is exactly "ids then ``hydrate()`` of those ids").

        Args:
            node: The anchor node.
            edge_type: When given, restrict to this edge type only.
            direction: ``"out"`` / ``"in"`` / ``"both"``.
            hydrate: When ``True``, attach each neighbour's body.

        Returns:
            The neighbours in canonical order.
        """
        pairs = _neighbor_records(self._index, node, direction=direction, edge_type=edge_type)
        out: list[Neighbor] = []
        for endpoint_id, rec in pairs:
            body = await self.get(endpoint_id) if hydrate else None
            out.append(
                Neighbor(
                    node_id=endpoint_id,
                    edge_id=rec.edge_id,
                    edge_type=rec.edge_type,
                    body=body,
                )
            )
        return out

    async def bfs(
        self,
        start: UUID,
        *,
        depth: int,
        edge_type: Optional[str] = None,
        direction: QueryDirection = "both",
        hydrate: bool = False,
    ) -> list[Union[UUID, Neighbor]]:
        """Breadth-first reachable node ids within ``depth`` hops of ``start``.

        Returns the reachable node ids (excluding ``start``) sorted by ``str(id)``.
        With ``hydrate=False`` the elements are bare ``UUID``s; with ``hydrate=True``
        each is a :class:`~vella.graph.Neighbor` carrying the body (its ``edge_id`` /
        ``edge_type`` reflect one canonical incident edge — traversal reaches a node,
        not a single edge). The id sequence is identical in both cases.

        Args:
            start: The anchor node.
            depth: The maximum number of hops (``>= 0``).
            edge_type: Optional single-type restriction on every hop.
            direction: ``"out"`` / ``"in"`` / ``"both"``.
            hydrate: When ``True``, return hydrated :class:`~vella.graph.Neighbor`s.

        Returns:
            The reachable ids (or hydrated neighbours), sorted by ``str(id)``.
        """
        ids = _bfs(self._index, start, depth=depth, direction=direction, edge_type=edge_type)
        return await self._as_traversal_result(ids, direction, edge_type, hydrate)

    async def dfs(
        self,
        start: UUID,
        *,
        depth: int,
        edge_type: Optional[str] = None,
        direction: QueryDirection = "both",
        hydrate: bool = False,
    ) -> list[Union[UUID, Neighbor]]:
        """Depth-first reachable node ids within ``depth`` hops of ``start``.

        Visits neighbours in sorted-id order and returns the reachable set (excluding
        ``start``) sorted by ``str(id)`` — byte-identical to :meth:`bfs` for the same
        bound. ``hydrate`` behaves as in :meth:`bfs`.

        Args:
            start: The anchor node.
            depth: The maximum number of hops (``>= 0``).
            edge_type: Optional single-type restriction on every hop.
            direction: ``"out"`` / ``"in"`` / ``"both"``.
            hydrate: When ``True``, return hydrated :class:`~vella.graph.Neighbor`s.

        Returns:
            The reachable ids (or hydrated neighbours), sorted by ``str(id)``.
        """
        ids = _dfs(self._index, start, depth=depth, direction=direction, edge_type=edge_type)
        return await self._as_traversal_result(ids, direction, edge_type, hydrate)

    async def reachable(
        self,
        start: UUID,
        target: UUID,
        *,
        direction: QueryDirection = "both",
    ) -> bool:
        """Whether ``target`` is reachable from ``start`` (unbounded).

        Args:
            start: The source node.
            target: The node to reach.
            direction: ``"out"`` / ``"in"`` / ``"both"``.

        Returns:
            ``True`` iff ``target == start`` or a directed walk reaches ``target``.
        """
        return _reachable(self._index, start, target, direction=direction)

    async def shortest_path(
        self,
        start: UUID,
        target: UUID,
        *,
        direction: QueryDirection = "both",
        hydrate: bool = False,
    ) -> Optional[Path]:
        """The canonical unweighted shortest path ``start -> target``, or ``None``.

        Among all equal-length shortest paths the engine returns the
        lexicographically smallest by node-id sequence. With ``hydrate=True`` the
        result's ``bodies`` are the hydrated bodies aligned to ``nodes``; the node
        sequence is identical to ``hydrate=False``.

        Args:
            start: The path's first node.
            target: The path's last node.
            direction: ``"out"`` / ``"in"`` / ``"both"``.
            hydrate: When ``True``, attach the per-node bodies.

        Returns:
            The :class:`~vella.graph.Path`, or ``None`` when ``target`` is
            unreachable.
        """
        nodes = _shortest_path(self._index, start, target, direction=direction)
        if nodes is None:
            return None
        bodies: Optional[tuple[Optional[Any], ...]] = None
        if hydrate:
            bodies = tuple([await self.get(node_id) for node_id in nodes])
        return Path(nodes=nodes, bodies=bodies)

    async def _as_traversal_result(
        self,
        ids: list[UUID],
        direction: QueryDirection,
        edge_type: Optional[str],
        hydrate: bool,
    ) -> list[Union[UUID, Neighbor]]:
        """Render a traversal's sorted id list as bare ids or hydrated neighbours.

        The id sequence is identical regardless of ``hydrate``; with ``hydrate`` each
        id becomes a :class:`~vella.graph.Neighbor` carrying its body and one
        canonical incident edge (the lexicographically smallest, or zeroed when the
        node is dangling / has no recorded incident edge in ``direction``).
        """
        if not hydrate:
            return list(ids)
        out: list[Union[UUID, Neighbor]] = []
        for node_id in ids:
            body = await self.get(node_id)
            out.append(self._neighbor_for(node_id, direction, edge_type, body))
        return out

    def _neighbor_for(
        self,
        node_id: UUID,
        direction: QueryDirection,
        edge_type: Optional[str],
        body: Optional[Any],
    ) -> Neighbor:
        """Build a :class:`~vella.graph.Neighbor` for a traversal-reached ``node_id``.

        A traversal reaches a *node*, not a single edge, so the carried edge is the
        node's lexicographically smallest incident edge (by ``(edge_type, edge_id)``)
        in the same ``direction`` / ``edge_type`` filter — a deterministic, canonical
        choice. When the node has no recorded incident edge in that scope (a dangling
        target) the edge fields are zeroed so the model stays well-formed; the
        ``node_id`` is the load-bearing, mode-independent value.
        """
        concretes = ("out", "in") if direction == "both" else (direction,)
        candidates: list[tuple[str, str]] = []
        for concrete in concretes:
            for rec in self._index.neighbors(node_id, concrete):  # type: ignore[arg-type]
                if edge_type is None or rec.edge_type == edge_type:
                    candidates.append((rec.edge_type, str(rec.edge_id)))
        if candidates:
            best_type, best_edge_str = min(candidates)
            return Neighbor(
                node_id=node_id,
                edge_id=UUID(best_edge_str),
                edge_type=best_type,
                body=body,
            )
        return Neighbor(node_id=node_id, edge_id=UUID(int=0), edge_type="", body=body)

    def _tenant_id_or_empty(self) -> str:
        """Return the view's tenant id (``""`` for a body-free / doctest view)."""
        return self._tenant_id if self._tenant_id is not None else ""

    def _internal_index(self) -> GraphIndex:
        """Return the underlying adjacency index (documented test-only accessor).

        Exposed so topology-equivalence / dangling / tenancy tests can assert on the
        folded structure directly. Not part of the query contract.

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
