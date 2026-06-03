"""The deterministic traversal engine over the M2 index (internal, M3).

Every query reads the always-built :class:`~vella.graph._index.GraphIndex` and
returns ids in a canonical order. Determinism is structural, not hoped-for:

* **Results are sorted-id.** ``neighbors`` returns edge records already sorted by the
  index; BFS/DFS return the visited node ids ``sorted()``.
* **Visitation order is sorted-id.** When a traversal expands a frontier it visits a
  node's neighbours in sorted endpoint-id order, so the set of nodes reached at each
  depth (and hence the depth-bound) is independent of fold / hash order.
* **Paths are canonical.** Among all equal-length shortest paths the engine keeps the
  lexicographically smallest by node-id sequence (``tuple(str(id), ...)``), so
  ``shortest_path`` is a single well-defined answer.

``direction`` is one of ``"out"`` / ``"in"`` / ``"both"``; an ``edge_type`` filter
restricts every hop to a single bucket. The engine never reads bodies — it is pure
topology, so its results are byte-identical across materialization modes.
"""

from __future__ import annotations

from collections import deque
from typing import Literal, Optional
from uuid import UUID

from ._index import Direction, EdgeRecord, GraphIndex

QueryDirection = Literal["out", "in", "both"]
"""A query traversal direction: ``"out"``, ``"in"``, or ``"both"`` (the union)."""

# The concrete index directions a query direction expands over.
_DIRECTIONS: dict[QueryDirection, tuple[Direction, ...]] = {
    "out": ("out",),
    "in": ("in",),
    "both": ("out", "in"),
}


def _endpoint(rec: EdgeRecord, concrete: Direction) -> UUID:
    """The far endpoint of ``rec`` when traversed in ``concrete`` direction."""
    return rec.to_id if concrete == "out" else rec.from_id


def _adjacent_ids(
    index: GraphIndex,
    node_id: UUID,
    direction: QueryDirection,
    edge_type: Optional[str],
) -> list[UUID]:
    """The sorted, de-duplicated adjacent node ids from ``node_id``.

    Used as the per-hop expansion for BFS/DFS/reachability/shortest-path. Sorting
    here is what makes traversal *visitation* order deterministic.

    Args:
        index: The adjacency index to read.
        node_id: The anchor node.
        direction: ``"out"`` / ``"in"`` / ``"both"``.
        edge_type: Optional single-type restriction.

    Returns:
        The adjacent node ids, sorted by ``str(id)`` and de-duplicated.
    """
    seen: set[UUID] = set()
    for concrete in _DIRECTIONS[direction]:
        for rec in index.neighbors(node_id, concrete):
            if edge_type is None or rec.edge_type == edge_type:
                seen.add(_endpoint(rec, concrete))
    return sorted(seen, key=str)


def neighbor_records(
    index: GraphIndex,
    node_id: UUID,
    *,
    direction: QueryDirection,
    edge_type: Optional[str],
) -> list[tuple[UUID, EdgeRecord]]:
    """The ``(endpoint_id, edge_record)`` pairs for ``node_id``, canonically ordered.

    The pairs are sorted by ``(str(endpoint_id), edge_type, str(edge_id))`` so the
    neighbour list is fully deterministic even when two edges of different types
    reach the same endpoint.

    Args:
        index: The adjacency index to read.
        node_id: The anchor node.
        direction: ``"out"`` / ``"in"`` / ``"both"``.
        edge_type: Optional single-type restriction.

    Returns:
        Sorted ``(endpoint_id, record)`` pairs (one per incident edge).
    """
    pairs: list[tuple[UUID, EdgeRecord]] = []
    for concrete in _DIRECTIONS[direction]:
        for rec in index.neighbors(node_id, concrete):
            if edge_type is None or rec.edge_type == edge_type:
                pairs.append((_endpoint(rec, concrete), rec))
    pairs.sort(key=lambda p: (str(p[0]), p[1].edge_type, str(p[1].edge_id)))
    return pairs


def bfs(
    index: GraphIndex,
    start: UUID,
    *,
    depth: int,
    direction: QueryDirection,
    edge_type: Optional[str],
) -> list[UUID]:
    """Breadth-first reachable node ids within ``depth`` hops of ``start``, sorted.

    Expands each frontier in sorted endpoint-id order. ``start`` itself is excluded
    from the result (it is the anchor, at depth 0); a node reachable within ``depth``
    hops appears exactly once. ``depth=0`` reaches nothing.

    Args:
        index: The adjacency index to read.
        start: The anchor node.
        depth: The maximum number of hops (``>= 0``).
        direction: ``"out"`` / ``"in"`` / ``"both"``.
        edge_type: Optional single-type restriction.

    Returns:
        The reachable node ids (excluding ``start``), sorted by ``str(id)``.
    """
    visited: set[UUID] = {start}
    frontier: deque[tuple[UUID, int]] = deque([(start, 0)])
    reached: set[UUID] = set()
    while frontier:
        node_id, dist = frontier.popleft()
        if dist >= depth:
            continue
        for adj in _adjacent_ids(index, node_id, direction, edge_type):
            if adj not in visited:
                visited.add(adj)
                reached.add(adj)
                frontier.append((adj, dist + 1))
    return sorted(reached, key=str)


def dfs(
    index: GraphIndex,
    start: UUID,
    *,
    depth: int,
    direction: QueryDirection,
    edge_type: Optional[str],
) -> list[UUID]:
    """Depth-first reachable node ids within ``depth`` hops of ``start``, sorted.

    Visits a node's neighbours in sorted endpoint-id order (canonical visitation),
    bounded by ``depth``. The set of reachable nodes equals BFS's for the same bound;
    the result is returned ``sorted()`` so DFS and BFS agree byte-for-byte. ``start``
    is excluded.

    Args:
        index: The adjacency index to read.
        start: The anchor node.
        depth: The maximum number of hops (``>= 0``).
        direction: ``"out"`` / ``"in"`` / ``"both"``.
        edge_type: Optional single-type restriction.

    Returns:
        The reachable node ids (excluding ``start``), sorted by ``str(id)``.
    """
    visited: set[UUID] = {start}
    reached: set[UUID] = set()
    # Stack of (node, remaining_depth); push neighbours in reverse-sorted order so
    # they pop in sorted order (canonical visitation).
    stack: list[tuple[UUID, int]] = [(start, depth)]
    while stack:
        node_id, remaining = stack.pop()
        if remaining <= 0:
            continue
        for adj in reversed(_adjacent_ids(index, node_id, direction, edge_type)):
            if adj not in visited:
                visited.add(adj)
                reached.add(adj)
                stack.append((adj, remaining - 1))
    return sorted(reached, key=str)


def reachable(
    index: GraphIndex,
    start: UUID,
    target: UUID,
    *,
    direction: QueryDirection,
) -> bool:
    """Whether ``target`` is reachable from ``start`` (unbounded), or ``start == target``.

    A plain reachability BFS over the whole component; not depth-bounded. Visitation
    is sorted-id but the boolean answer is order-independent.

    Args:
        index: The adjacency index to read.
        start: The source node.
        target: The node to reach.
        direction: ``"out"`` / ``"in"`` / ``"both"``.

    Returns:
        ``True`` iff ``target == start`` or a directed walk reaches ``target``.
    """
    if start == target:
        return True
    visited: set[UUID] = {start}
    frontier: deque[UUID] = deque([start])
    while frontier:
        node_id = frontier.popleft()
        for adj in _adjacent_ids(index, node_id, direction, None):
            if adj == target:
                return True
            if adj not in visited:
                visited.add(adj)
                frontier.append(adj)
    return False


def shortest_path(
    index: GraphIndex,
    start: UUID,
    target: UUID,
    *,
    direction: QueryDirection,
) -> Optional[tuple[UUID, ...]]:
    """The canonical unweighted shortest path ``start -> target``, or ``None``.

    A breadth-first search expanding each frontier in sorted endpoint-id order, so
    the FIRST time ``target`` is dequeued it carries the lexicographically smallest
    node-id sequence among all shortest (equal-length) paths — the canonical answer.

    Args:
        index: The adjacency index to read.
        start: The path's first node.
        target: The path's last node.
        direction: ``"out"`` / ``"in"`` / ``"both"``.

    Returns:
        The node-id tuple from ``start`` to ``target`` (length 1 when
        ``start == target``), or ``None`` when ``target`` is unreachable.
    """
    if start == target:
        return (start,)
    # BFS carrying the path; expanding neighbours in sorted order means the first
    # path that reaches a node is its canonical (lex-smallest) shortest path, so we
    # never need to revisit. (Equal-length alternatives arrive later and are dropped.)
    visited: set[UUID] = {start}
    frontier: deque[tuple[UUID, tuple[UUID, ...]]] = deque([(start, (start,))])
    while frontier:
        node_id, path = frontier.popleft()
        for adj in _adjacent_ids(index, node_id, direction, None):
            if adj in visited:
                continue
            new_path = path + (adj,)
            if adj == target:
                return new_path
            visited.add(adj)
            frontier.append((adj, new_path))
    return None
