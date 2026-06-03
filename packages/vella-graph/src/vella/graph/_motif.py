"""The bounded motif matcher — anchored, type-pruned guided join (internal, M4).

Evaluates a :class:`~vella.graph.MotifPattern` (an ordered, fixed-shape tuple of
hops) anchored at a single caller-supplied start node. Each hop is a guided join
step: from every match-prefix frontier node it reads exactly the index's
``edge_type`` partition for that hop's type in that hop's direction (type-pruned —
it never scans all of a node's edges), follows the edge to its endpoint, and keeps
the endpoint only when it satisfies the hop's optional ``to_node_type`` (read from
the index ``node_types`` map; a dangling endpoint with no folded type is pruned by
any ``to_node_type`` filter).

This is NOT a general query language: the hop count and shape are fixed by the
pattern, there is no branching/optional/variable-length step, and there is no
opaque predicate — only declarative edge-type / direction / node-type pruning. That
boundedness is exactly what keeps the matcher deterministic and prunable.

Determinism: matches are returned as node-id tuples sorted by ``tuple(str(id) ...)``
— a single canonical order independent of fold / hash order. Within the join, a
node already on the current prefix is not revisited (simple paths), so a cycle
cannot make the matcher loop.
"""

from __future__ import annotations

from uuid import UUID

from ._index import Direction, GraphIndex
from .motif import MotifPattern


def match(
    index: GraphIndex,
    anchor: UUID,
    pattern: MotifPattern,
) -> list[tuple[UUID, ...]]:
    """Every match of ``pattern`` anchored at ``anchor``, as sorted node-id tuples.

    Args:
        index: The adjacency index to read (topology + ``node_types``).
        anchor: The fixed start node every match begins at (``nodes[0]``).
        pattern: The fixed-shape hop sequence to match.

    Returns:
        The matched node-id tuples (each of length ``len(pattern.hops) + 1``,
        beginning with ``anchor``), sorted by ``tuple(str(id) ...)``. A hop-free
        pattern yields the single anchor tuple ``[(anchor,)]``.
    """
    prefixes: list[tuple[UUID, ...]] = [(anchor,)]
    for hop in pattern.hops:
        concrete: Direction = hop.direction
        extended: list[tuple[UUID, ...]] = []
        for prefix in prefixes:
            current = prefix[-1]
            # Type-pruned read: exactly the (direction, current, edge_type) bucket,
            # never a full scan of the node's edges.
            bucket = index.adj[concrete].get(current, {}).get(hop.edge_type, ())
            for rec in bucket:
                endpoint = rec.to_id if concrete == "out" else rec.from_id
                # Node-type pruning: honour the hop's optional to_node_type against
                # the folded node_types map (a dangling endpoint has no folded type
                # and is pruned by any filter).
                if hop.to_node_type is not None:
                    if index.node_types.get(endpoint) != hop.to_node_type:
                        continue
                # Simple-path discipline: never revisit a node already on the prefix
                # (keeps a cyclic graph from looping the bounded matcher).
                if endpoint in prefix:
                    continue
                extended.append(prefix + (endpoint,))
        prefixes = extended
    prefixes.sort(key=lambda nodes: tuple(str(n) for n in nodes))
    return prefixes
