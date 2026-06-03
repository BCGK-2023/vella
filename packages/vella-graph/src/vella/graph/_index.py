"""The A1 copy-on-write adjacency index (internal mechanism, M2).

The substrate every query reads. The runtime exposes no list/scan verb — no way to
ask "which edges touch node X" — so the only way to answer a graph query is to fold
``observe()`` into an in-memory adjacency index once and read it many times.

Structure (the A1 option from the consensus plan)::

    adj[direction][node_id][edge_type] -> tuple[EdgeRecord, ...]   (sorted)

a three-level nested map keyed by direction (``"out"`` / ``"in"``), then the
anchor node id, then the edge type — so a type-pruned motif hop (M4) reads exactly
the bucket it needs and a neighbour read is ``O(distinct edge_types at the node)``,
bounded by schema not data. Records within a bucket are sorted by a canonical key
(``(edge_type, str(edge_id))``) so neighbour reads are deterministic regardless of
fold/hash order.

The index is **always built** in both materialization modes — it is topology
(ids), not bodies. ``MaterializationMode`` (M2's ``mode.py``) controls only whether
``Node``/``Edge`` bodies are held resident; it never changes the topology this
index records. Dangling edges (an endpoint node absent / deleted / not-yet-seen)
are kept: ids are truth (spec decision #4).

Copy-on-write (``apply_delta``, used by M5's ``refresh``): rebuilding the index for
a delta touches ONLY the ``(direction, node_id)`` buckets whose incident edge set
changed; every untouched bucket is shared BY IDENTITY with the prior index
(``is``-equal), so an incremental refresh is ``O(Δ)`` in memory and the M5 gate can
assert structural sharing directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping
from uuid import UUID

Direction = Literal["out", "in"]
"""An adjacency direction: ``"out"`` (edges leaving a node) or ``"in"`` (arriving)."""

# A sorted bucket of edge records sharing one (direction, node_id, edge_type).
_Bucket = tuple["EdgeRecord", ...]
# node_id -> edge_type -> sorted bucket.
_NodeMap = dict[UUID, dict[str, _Bucket]]


@dataclass(frozen=True)
class EdgeRecord:
    """One directed edge baked into the index: ids, type, and a baked weight.

    Frozen and id-only: the authoritative endpoints and type are read once via
    ``runtime.get(tenant_id, edge_id)`` at fold time (TRAP-1 — never from
    ``LogEntry.payload``); the optional fold-time weight function bakes a float so
    weighted shortest-path (M4) is pure in-memory in both modes.
    """

    edge_id: UUID
    from_id: UUID
    to_id: UUID
    edge_type: str
    weight: float = 0.0

    def sort_key(self) -> tuple[str, str]:
        """Canonical, total ordering key for deterministic bucket ordering.

        Returns:
            ``(edge_type, str(edge_id))`` — stable across fold order and hash seed.
        """
        return (self.edge_type, str(self.edge_id))


def _empty_adj() -> dict[Direction, _NodeMap]:
    """A fresh, empty bidirectional adjacency map (both directions present)."""
    return {"out": {}, "in": {}}


@dataclass(frozen=True)
class GraphIndex:
    """The immutable type-partitioned bidirectional adjacency index.

    Built once by the fold (``_fold.py``) and shared by every query on a
    ``GraphView``. Frozen: a ``refresh`` (M5) produces a NEW ``GraphIndex`` via
    :meth:`apply_delta`, sharing untouched buckets by identity rather than mutating
    in place.

    Attributes:
        adj: ``direction -> node_id -> edge_type -> sorted tuple of records``. The
            same edge appears under ``adj["out"][from_id]`` and
            ``adj["in"][to_id]``.
        node_types: live node id -> its ``type`` string (for M4 motif node-type
            pruning). Dangling-edge endpoints that were never seen as live nodes are
            absent here (ids are still truth in ``adj``).
        live_edges: the set of live (non-tombstoned) edge ids the fold recorded.
    """

    adj: dict[Direction, _NodeMap] = field(default_factory=_empty_adj)
    node_types: dict[UUID, str] = field(default_factory=lambda: {})
    live_edges: frozenset[UUID] = field(default_factory=lambda: frozenset())

    def neighbors(self, node_id: UUID, direction: Direction) -> tuple[EdgeRecord, ...]:
        """All edge records incident to ``node_id`` in ``direction``, sorted.

        Concatenates the node's per-edge-type buckets in sorted ``edge_type`` order
        (each bucket is itself sorted by :meth:`EdgeRecord.sort_key`), so the whole
        result is deterministic.

        Args:
            node_id: The anchor node.
            direction: ``"out"`` or ``"in"``.

        Returns:
            The incident edge records (empty if the node has no edges in that
            direction). May reference dangling endpoint ids.
        """
        by_type = self.adj[direction].get(node_id, {})
        out: list[EdgeRecord] = []
        for edge_type in sorted(by_type):
            out.extend(by_type[edge_type])
        return tuple(out)

    @classmethod
    def build(
        cls,
        records: list[EdgeRecord],
        node_types: Mapping[UUID, str],
    ) -> "GraphIndex":
        """Build a fresh index from the full live edge-record set.

        Used by the cold fold (``_fold.py``). Every record is placed into both its
        ``out`` bucket (keyed by ``from_id``) and its ``in`` bucket (keyed by
        ``to_id``); each bucket is sorted by :meth:`EdgeRecord.sort_key`.

        Args:
            records: Every live edge as an ``EdgeRecord`` (baked weight included).
            node_types: live node id -> type (dangling endpoints may be absent).

        Returns:
            The immutable index.
        """
        # Group into per-anchor, per-type lists, then sort each bucket once
        # (cheaper than re-sorting on every insert).
        scratch_out: dict[UUID, dict[str, list[EdgeRecord]]] = {}
        scratch_in: dict[UUID, dict[str, list[EdgeRecord]]] = {}
        for rec in records:
            scratch_out.setdefault(rec.from_id, {}).setdefault(rec.edge_type, []).append(rec)
            scratch_in.setdefault(rec.to_id, {}).setdefault(rec.edge_type, []).append(rec)
        adj = _empty_adj()
        for node_id, by_type in scratch_out.items():
            adj["out"][node_id] = {
                et: tuple(sorted(bucket, key=EdgeRecord.sort_key))
                for et, bucket in by_type.items()
            }
        for node_id, by_type in scratch_in.items():
            adj["in"][node_id] = {
                et: tuple(sorted(bucket, key=EdgeRecord.sort_key))
                for et, bucket in by_type.items()
            }
        return cls(
            adj=adj,
            node_types=dict(node_types),
            live_edges=frozenset(rec.edge_id for rec in records),
        )

    def apply_delta(
        self,
        records: list[EdgeRecord],
        node_types: Mapping[UUID, str],
    ) -> "GraphIndex":
        """Rebuild a NEW index for the full live set, sharing untouched buckets.

        Copy-on-write: a ``(direction, node_id)`` whose incident edge set is
        BYTE-IDENTICAL to this index's bucket map is shared BY IDENTITY (the same
        ``dict`` object) in the result; only touched anchors are rebuilt. This makes
        M5's ``refresh`` ``O(Δ)`` and lets the gate assert structural sharing with
        ``is``.

        Args:
            records: The FULL live edge-record set after applying the delta (the
                fold recomputes the live set; this method diffs it against the
                current index to decide which buckets to rebuild).
            node_types: live node id -> type after the delta.

        Returns:
            A new ``GraphIndex``; untouched ``adj[dir][node]`` maps are the same
            objects as in ``self``.
        """
        fresh = GraphIndex.build(records, node_types)
        shared = _empty_adj()
        for direction in ("out", "in"):
            old_dir = self.adj[direction]
            new_dir = fresh.adj[direction]
            merged: _NodeMap = {}
            for node_id, new_bucket in new_dir.items():
                old_bucket = old_dir.get(node_id)
                # Share by identity when the rebuilt bucket map is value-equal to the
                # prior one (untouched anchor); otherwise keep the freshly-built one.
                if old_bucket is not None and old_bucket == new_bucket:
                    merged[node_id] = old_bucket
                else:
                    merged[node_id] = new_bucket
            shared[direction] = merged
        return GraphIndex(
            adj=shared,
            node_types=fresh.node_types,
            live_edges=fresh.live_edges,
        )
