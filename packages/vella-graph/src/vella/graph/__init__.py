"""Vella graph projection (a read-only traversal view over the runtime log).

Where ``vella.runtime`` is *physics* — the append-only log, the
optimistic-concurrency store, and the write verbs that move world state forward —
``vella.graph`` is a *read-only projection* on top of it. It folds ``observe()``
into an always-built, type-partitioned, bidirectional adjacency index and answers
graph/traversal queries from memory. It owns no storage and performs no writes;
the runtime remains the sole authority.

Design principles
-----------------
* **The index is forced; read-through is impossible.** The runtime exposes only
  ``get``/``history``/``observe`` — no list/scan, no "edges touching X". Every
  query is answered from an in-memory adjacency index folded from ``observe()``.
* **Maximal expression, fast via structure.** Working-set memory buys latency: a
  type-partitioned bidirectional index makes expressive queries fast through
  anchoring, type-pruning, and baked weights, without amputating capability.
* **Determinism is a property, not a hope.** Every query returns ``sorted()`` ids;
  the gated determinism artifact is topology-derived and byte-identical across
  hash seeds. Any set-derived serialized value is ``sorted()``.
* **Depend downward only.** The graph imports only the published ``vella.runtime``
  and ``vella.core`` surfaces; both layers are unaware of it.

The public surface grows milestone by milestone; everything in ``__all__`` is
importable, documented, and snapshotted by the surface tripwire from M1 onward.
The concrete projection, view, queries, and follower land in later milestones; the
surface is baselined now (empty) so the tripwire guards it from the start.
"""

from __future__ import annotations

from .errors import WeightOverrideRequiresFullMode
from .mode import MaterializationMode
from .motif import MotifHop, MotifPattern
from .projection import GraphProjection
from .results import Match, Neighbor, Path
from .view import GraphView

__all__ = [
    "GraphProjection",
    "GraphView",
    "Match",
    "MaterializationMode",
    "MotifHop",
    "MotifPattern",
    "Neighbor",
    "Path",
    "WeightOverrideRequiresFullMode",
]

