"""The cold fold: ``observe()`` -> live id-set -> authoritative index (M2).

The fold is the projection's bootstrap. It bounded-drains ``observe(since=None)``
to the live edge (TRAP-2), computing a per-tenant live id-set + tombstones from the
TYPED top-level ``LogEntry`` fields ONLY (``entity_kind`` / ``entity_id`` /
``transition`` / ``tenant_id``) — NEVER ``.payload`` (TRAP-1). Once caught up, it
reads each live entity's authority via ``runtime.get()`` exactly once (edges always;
nodes only in ``full`` mode) and builds the :class:`~vella.graph._index.GraphIndex`.

Two disciplines ported verbatim from the reconciler:

* **Skip ``observe_only``** (telemetry): such an entry advances an internal
  monotonic high-water counter (it WAS drained) but never enters the live-set /
  tombstone tracking and never triggers a ``get()`` (mirrors
  ``workset._NON_STATE_CHANGING``).
* **Opaque cursor.** The fold stores the LAST entry's ``Cursor`` verbatim as the
  resume token (passed back to ``observe(since=)`` in M5); it NEVER compares
  cursors (``Cursor`` has no ``__lt__``). "How far" is the internal int only.

Dangling edges (an endpoint node absent / deleted / not-yet-seen) are KEPT — ids
are truth (spec decision #4). ``delete`` / ``unlink`` remove the entity from the
live set; a deleted node's incident edges remain (dangling).
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from uuid import UUID

from vella.core import Edge
from vella.runtime import Cursor, LogEntry, Runtime

from ._drain import bounded_drain
from ._index import EdgeRecord, GraphIndex
from .mode import MaterializationMode

# Transitions that do NOT change live state: drained + high-water-counted, but never
# tracked. Only ``observe_only`` (telemetry) — mirrors ``workset._NON_STATE_CHANGING``.
_NON_STATE_CHANGING: frozenset[str] = frozenset({"observe_only"})
# Transitions that remove an entity from the live set.
_REMOVING: frozenset[str] = frozenset({"delete", "unlink"})


class _FoldState:
    """Mutable accumulator the bounded-drain sink writes into for one tenant.

    Tracks the live id-set per kind from typed fields only, the internal monotonic
    high-water (int — never a cursor compare), and the last cursor verbatim.
    """

    def __init__(self, tenant_id: str) -> None:
        """Start an empty fold for ``tenant_id`` positioned before the first entry."""
        self._tenant_id = tenant_id
        # Live id-sets per kind (insertion order irrelevant — sorted at build).
        self.live_nodes: set[UUID] = set()
        self.live_edges: set[UUID] = set()
        # Monotonic count of drained entries (observe_only included). Never a cursor.
        self.high_water_count: int = 0
        # The last entry's cursor, stored verbatim as the opaque resume token.
        self.high_water: Optional[Cursor] = None

    def apply(self, entry: LogEntry) -> None:
        """Fold one ``LogEntry`` from typed top-level fields only (TRAP-1).

        Advances the internal high-water for EVERY drained entry (observe_only
        included) and records the cursor verbatim. Skips other tenants entirely and
        skips ``observe_only`` from live-set tracking. ``delete``/``unlink`` remove
        the entity; every other state-changing transition adds it.

        Args:
            entry: The log entry to fold. Only ``cursor`` / ``tenant_id`` /
                ``entity_kind`` / ``entity_id`` / ``transition`` are read.
        """
        # Every drained entry advances the high-water and records the resume cursor.
        self.high_water_count += 1
        self.high_water = entry.cursor

        # Foreign tenants are not part of THIS view (filtered, not tracked).
        if entry.tenant_id != self._tenant_id:
            return
        # Telemetry advances the high-water but never touches the live set.
        if entry.transition in _NON_STATE_CHANGING:
            return

        target = self.live_nodes if entry.entity_kind == "node" else self.live_edges
        if entry.transition in _REMOVING:
            target.discard(entry.entity_id)
        else:
            target.add(entry.entity_id)


async def fold(
    runtime: Runtime,
    tenant_id: str,
    *,
    mode: MaterializationMode,
    weight: Optional[Callable[[Edge[Any, Any]], float]],
) -> "_FoldResult":
    """Bounded-drain ``observe()`` and build the authoritative index for a tenant.

    Args:
        runtime: The runtime whose log is folded (read-only — ``observe``/``get``).
        tenant_id: The tenant to project; foreign-tenant entries are ignored.
        mode: ``"full"`` holds node+edge bodies resident; ``"lean"`` holds none.
        weight: Optional pure ``Edge -> float`` baked onto each edge record (default
            ``0.0`` when ``None``).

    Returns:
        A :class:`_FoldResult` carrying the built index, the high-water cursor, and
        (in ``full`` mode) the resident bodies.
    """
    state = _FoldState(tenant_id)
    stream = runtime.observe(since=None)
    try:
        await bounded_drain(
            stream,
            sink=state.apply,
            on_caught_up=lambda: None,
        )
    finally:
        await stream.aclose()

    # Authority pass: get() each live edge once for endpoints/type (+ baked weight);
    # in full mode also get() each live node once and hold the body resident.
    records: list[EdgeRecord] = []
    node_types: dict[UUID, str] = {}
    resident: dict[UUID, Any] = {}

    for edge_id in state.live_edges:
        got = await runtime.get(tenant_id, edge_id)
        if got is None or not isinstance(got, Edge):
            # Tombstoned between fold pass and authority read, or kind mismatch:
            # drop from topology (the live set was id-derived; get() is authority).
            continue
        baked = weight(got) if weight is not None else 0.0
        records.append(
            EdgeRecord(
                edge_id=got.id,
                from_id=got.from_node_id,
                to_id=got.to_node_id,
                edge_type=got.type,
                weight=baked,
            )
        )
        if mode == "full":
            resident[got.id] = got

    for node_id in state.live_nodes:
        got = await runtime.get(tenant_id, node_id)
        if got is None or isinstance(got, Edge):
            continue
        node_types[got.id] = got.type
        if mode == "full":
            resident[got.id] = got

    index = GraphIndex.build(records, node_types)
    return _FoldResult(
        index=index,
        high_water=state.high_water,
        resident=resident if mode == "full" else {},
    )


class _FoldResult:
    """The output of one fold pass: index, opaque high-water cursor, resident bodies."""

    def __init__(
        self,
        *,
        index: GraphIndex,
        high_water: Optional[Cursor],
        resident: dict[UUID, Any],
    ) -> None:
        """Bundle a fold's outputs (see attributes)."""
        self.index = index
        self.high_water = high_water
        # full mode only: id -> Node|Edge body, fold-pinned. Empty in lean.
        self.resident = resident
