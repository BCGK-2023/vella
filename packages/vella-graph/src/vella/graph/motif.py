"""The public motif pattern surface — a bounded, fixed-shape hop sequence (M4).

A :class:`MotifPattern` is an ordered sequence of :class:`MotifHop` steps evaluated
as an anchored, type-pruned guided join over the index (``GraphView.match``). It is
deliberately NOT a general query language (no Cypher-shaped DSL, no callable
predicate): the shape is fixed — a start anchor followed by a fixed number of typed,
directed hops — so the matcher can prune by ``edge_type`` partition and node type
and return canonically-ordered matches. Expressivity beyond this fixed shape is an
explicit v0.1 non-goal (rejected options C2/C3 in the consensus plan).

Each hop names the ``edge_type`` partition to traverse, the ``direction``
(``"out"``/``"in"``), and an optional ``to_node_type`` that prunes the reached node
by its folded type — a hop whose endpoint is the wrong node type (or a dangling
endpoint with no folded type) is discarded.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

MotifDirection = Literal["out", "in"]
"""A single motif hop's traversal direction: ``"out"`` (leaving) or ``"in"``."""


class MotifHop(BaseModel):
    """One typed, directed step of a :class:`MotifPattern`.

    Frozen and fully declarative so the matcher can prune by edge-type partition
    and node type rather than evaluating an opaque predicate.

    Attributes:
        edge_type: The edge-type partition this hop traverses (matched exactly
            against the folded edge type); only edges in this partition are
            followed.
        direction: ``"out"`` to follow edges leaving the current node, ``"in"`` to
            follow arriving edges.
        to_node_type: When given, the reached node must have this folded node type;
            an endpoint of the wrong type — or a dangling endpoint with no folded
            type — is pruned. ``None`` means "any node type".

    Examples:
        >>> hop = MotifHop(edge_type="knows", direction="out")
        >>> hop.direction
        'out'
        >>> hop.to_node_type is None
        True
    """

    model_config = ConfigDict(frozen=True)

    edge_type: str
    direction: MotifDirection
    to_node_type: Optional[str] = None


class MotifPattern(BaseModel):
    """An ordered, fixed-shape sequence of :class:`MotifHop` steps.

    Anchored at a caller-supplied start node by :meth:`~vella.graph.GraphView.match`
    and evaluated as a type-pruned guided join: hop ``i`` expands every match-prefix
    frontier node along its ``edge_type`` partition in ``direction``, keeping only
    endpoints whose folded node type matches the hop's ``to_node_type`` (when set).
    Not a general query language — the hop count and shape are fixed at construction.

    Attributes:
        hops: The hops in traversal order; an empty tuple matches only the anchor
            itself (the trivial one-node match).

    Examples:
        >>> p = MotifPattern(hops=(MotifHop(edge_type="knows", direction="out"),))
        >>> len(p.hops)
        1
    """

    model_config = ConfigDict(frozen=True)

    hops: tuple[MotifHop, ...]


__all__ = ["MotifHop", "MotifPattern"]
