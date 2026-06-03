"""Frozen result models for the deterministic query surface (M3).

The query methods on :class:`~vella.graph.GraphView` return these models (or bare
sorted ids). They are the public, immutable shapes a caller receives:

* :class:`Neighbor` — one adjacent node id reached across one edge, with the
  optional fold-pinned / live body attached when the caller asked to hydrate.
* :class:`Path` — an ordered node-id sequence (the canonical, lexicographically
  smallest unweighted shortest path among equal-length candidates), with optional
  hydrated bodies aligned to the node sequence. Also returned by weighted shortest
  path (M4) over baked or per-query-override edge weights.
* :class:`Match` — one bounded-motif match: the anchor-to-final node-id tuple a
  :class:`~vella.graph.MotifPattern` matched, with optional hydrated bodies aligned
  to that tuple. Matches are returned in canonical (sorted node-id-tuple) order.

Both mirror the reconciler's model style: frozen pydantic, fully documented,
``model_dump(mode="json")``-friendly. The id-derived fields (``node_id``,
``nodes``, the edge id) come straight from the always-built topology index, so they
are byte-identical across materialization modes; only the attached ``body`` /
``bodies`` are mode-dependent (``full`` = fold-pinned, ``lean`` = live via the LRU)
and are explicitly NOT part of the topology-equivalence claim.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class Neighbor(BaseModel):
    """One adjacent node reached across one directed edge.

    Returned by :meth:`~vella.graph.GraphView.neighbors`. The id fields are
    topology (mode-independent); ``body`` is attached only when the caller asked to
    hydrate, and its *contents* are mode-dependent by design.

    Attributes:
        node_id: The adjacent node's id (the ``to`` endpoint for ``direction="out"``,
            the ``from`` endpoint for ``direction="in"``). May be a dangling id whose
            node body is absent in both modes.
        edge_id: The id of the edge traversed to reach ``node_id``.
        edge_type: The traversed edge's type.
        body: The hydrated ``Node`` body (``full`` = fold-pinned, ``lean`` = live via
            LRU), or ``None`` when hydration was not requested or the id is dangling /
            deleted.

    Examples:
        >>> from uuid import UUID
        >>> n = Neighbor(
        ...     node_id=UUID(int=2),
        ...     edge_id=UUID(int=9),
        ...     edge_type="knows",
        ... )
        >>> n.edge_type
        'knows'
        >>> n.body is None
        True
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    node_id: UUID
    edge_id: UUID
    edge_type: str
    body: Optional[Any] = None


class Path(BaseModel):
    """An ordered node-id sequence — a canonical unweighted shortest path.

    Returned by :meth:`~vella.graph.GraphView.shortest_path`. Among all equal-length
    shortest paths the engine returns the lexicographically smallest by node-id
    sequence (``tuple(str(node_id) ...)``), so the result is deterministic regardless
    of fold / hash order.

    Attributes:
        nodes: The path's node ids in order, ``nodes[0]`` the start and ``nodes[-1]``
            the target. A single-element tuple is the trivial ``start == target``
            path.
        bodies: When hydration was requested, the hydrated body per node aligned to
            ``nodes`` (same length; an element is ``None`` for a dangling / deleted
            id). ``None`` (the whole field) when hydration was not requested.

    Examples:
        >>> from uuid import UUID
        >>> p = Path(nodes=(UUID(int=1), UUID(int=2)))
        >>> len(p.nodes)
        2
        >>> p.bodies is None
        True
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    nodes: tuple[UUID, ...]
    bodies: Optional[tuple[Optional[Any], ...]] = None


class Match(BaseModel):
    """One bounded-motif match — the node-id tuple a pattern matched, in order.

    Returned by :meth:`~vella.graph.GraphView.match`. ``nodes[0]`` is always the
    anchor; ``nodes[i+1]`` is the node reached by hop ``i`` of the
    :class:`~vella.graph.MotifPattern`. Matches are returned in canonical order —
    sorted by their node-id tuple (``tuple(str(node_id) ...)``) — so the match list
    is deterministic regardless of fold / hash order; the node ids are topology, so
    they are byte-identical across materialization modes.

    Attributes:
        nodes: The matched node ids in hop order (length = ``len(pattern.hops) + 1``;
            a hop-free pattern yields the single-element anchor tuple). May contain a
            dangling endpoint id when a hop has no ``to_node_type`` filter.
        bodies: When hydration was requested, the hydrated body per node aligned to
            ``nodes`` (same length; an element is ``None`` for a dangling / deleted
            id). ``None`` (the whole field) when hydration was not requested.

    Examples:
        >>> from uuid import UUID
        >>> m = Match(nodes=(UUID(int=1), UUID(int=2)))
        >>> len(m.nodes)
        2
        >>> m.bodies is None
        True
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    nodes: tuple[UUID, ...]
    bodies: Optional[tuple[Optional[Any], ...]] = None


__all__ = ["Match", "Neighbor", "Path"]
