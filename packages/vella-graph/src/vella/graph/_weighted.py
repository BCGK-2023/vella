"""Weighted shortest path over the index — Dijkstra/heapq (internal, M4).

Reads the always-built :class:`~vella.graph._index.GraphIndex` and returns the
minimum-weight directed path ``start -> target`` (a node-id tuple) or ``None``.

Two weight sources share this one engine:

* **Baked** — the float baked onto each :class:`~vella.graph._index.EdgeRecord` at
  fold time (``fold(weight=...)``). Pure in-memory, so baked weighted SP is
  mode-equivalent: identical result in ``full`` and ``lean`` (decision D1).
* **Per-query override** — a caller-supplied weight recomputed from live edge
  *bodies* at query time; full-mode only (the view raises
  :class:`~vella.graph.WeightOverrideRequiresFullMode` in lean before calling here).

The weight source is injected as ``edge_weight: EdgeRecord -> float`` so this engine
never depends on bodies or mode.

**Determinism (the load-bearing invariant).** Dijkstra's frontier is a binary heap;
when two frontier nodes carry equal accumulated distance, the order they pop is
otherwise an artifact of insertion / object identity. We make the tie-break
canonical by pushing ``(distance, str(node_id), node_id, path)`` so equal-distance
nodes pop in sorted node-id order — never relying on ``heapq``'s incidental order or
on object ids. The ``str(node_id)`` key is what ``heapq`` compares on ties (it never
reaches the raw ``UUID``/path elements, which are not order-comparable), so the
settled path to every node — and hence the returned path among equal-weight
alternatives — is the lexicographically smallest by node-id sequence.
"""

from __future__ import annotations

import heapq
from typing import Callable, Optional
from uuid import UUID

from ._index import Direction, EdgeRecord, GraphIndex
from ._query import QueryDirection

# The concrete index directions a query direction expands over (mirrors _query).
_DIRECTIONS: dict[QueryDirection, tuple[Direction, ...]] = {
    "out": ("out",),
    "in": ("in",),
    "both": ("out", "in"),
}


def dijkstra(
    index: GraphIndex,
    start: UUID,
    target: UUID,
    *,
    direction: QueryDirection,
    edge_weight: Callable[[EdgeRecord], float],
) -> Optional[tuple[UUID, ...]]:
    """The minimum-weight directed path ``start -> target``, or ``None``.

    A canonical-tie-break Dijkstra over ``edge_weight(rec)`` per incident edge.
    Among all minimum-weight paths the engine settles each node on its
    lexicographically-smallest-by-node-id path, so the returned path is a single
    well-defined answer. Edge weights are assumed non-negative (baked weights and
    overrides are both costs); ``start == target`` returns the trivial one-node path.

    Args:
        index: The adjacency index to read.
        start: The path's first node.
        target: The path's last node.
        direction: ``"out"`` / ``"in"`` / ``"both"``.
        edge_weight: The per-edge cost (``rec.weight`` for baked; an override-derived
            cost for a per-query override).

    Returns:
        The node-id tuple from ``start`` to ``target`` (length 1 when
        ``start == target``), or ``None`` when ``target`` is unreachable.
    """
    if start == target:
        return (start,)
    # Heap entries are (dist, str(node_id), node_id, path). The str(node_id) key is
    # the canonical tie-break heapq compares on equal distance — it never falls
    # through to the raw UUID / path tuple (which are not order-comparable), and it
    # makes equal-distance nodes pop in sorted node-id order (determinism gate).
    heap: list[tuple[float, str, UUID, tuple[UUID, ...]]] = [
        (0.0, str(start), start, (start,))
    ]
    best: dict[UUID, float] = {start: 0.0}
    settled: set[UUID] = set()
    while heap:
        dist, _key, node_id, path = heapq.heappop(heap)
        if node_id in settled:
            continue
        if node_id == target:
            return path
        settled.add(node_id)
        for endpoint, weight in _incident_weights(index, node_id, direction, edge_weight):
            if endpoint in settled:
                continue
            new_dist = dist + weight
            prior = best.get(endpoint)
            # Push when strictly better OR equal-but-not-yet-seen: the canonical
            # tie-break in the heap key resolves equal-distance arrivals to the
            # lexicographically-smallest path without an explicit path compare here.
            if prior is None or new_dist < prior:
                best[endpoint] = new_dist
                heapq.heappush(heap, (new_dist, str(endpoint), endpoint, path + (endpoint,)))
    return None


def _incident_weights(
    index: GraphIndex,
    node_id: UUID,
    direction: QueryDirection,
    edge_weight: Callable[[EdgeRecord], float],
) -> list[tuple[UUID, float]]:
    """The ``(endpoint_id, weight)`` pairs incident to ``node_id``, min-weight per endpoint.

    When two edges reach the same endpoint, the cheapest is kept (a min-cost graph
    has no use for a parallel heavier edge); the per-endpoint reduction is
    order-independent so it does not perturb determinism.
    """
    cheapest: dict[UUID, float] = {}
    for concrete in _DIRECTIONS[direction]:
        concrete_dir: Direction = concrete
        for rec in index.neighbors(node_id, concrete_dir):
            endpoint = rec.to_id if concrete_dir == "out" else rec.from_id
            w = edge_weight(rec)
            prior = cheapest.get(endpoint)
            if prior is None or w < prior:
                cheapest[endpoint] = w
    return list(cheapest.items())
