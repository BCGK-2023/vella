"""The pull refresh: a delta fold + copy-on-write index update (M5).

``refresh`` is the incremental counterpart to the cold fold (``_fold.py``). Where
the fold drains ``observe(since=None)`` from the start, ``refresh`` drains
``observe(since=high_water)`` from the view's stored opaque resume token — so it
sees ONLY the entries appended after the folded position (the delta). It seeds the
live id-set from the index it is refreshing, so a ``delete`` / ``unlink`` in the
delta removes a prior-live entity and a ``create`` / ``link`` adds a new one, then
runs the SAME authority pass as the cold fold (``_fold.authority_pass``) and folds
the result into a NEW index via copy-on-write (``GraphIndex.apply_delta``).

Two load-bearing invariants (the M5 gate):

* **The cursor is opaque.** ``high_water`` is passed STRAIGHT to ``observe(since=)``
  as an opaque token; it is NEVER compared (``Cursor`` has no ``__lt__``). "How far"
  is the internal monotonic int on ``_FoldState`` only. The new high-water is the
  LAST delta entry's cursor, or the prior token unchanged when the delta is empty
  (the seed on ``_FoldState`` guarantees this without a comparison).
* **Copy-on-write.** ``apply_delta`` rebuilds ONLY the ``(direction, node_id)``
  buckets whose incident edge set changed; every untouched ``adj[dir][node]``
  sub-dict is shared BY IDENTITY (``is``) with the prior index. The refresh is
  ``O(Δ)`` in memory and the gate asserts the structural sharing directly. The prior
  index is never mutated, so the pre-refresh view is unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from uuid import UUID

from vella.core import Edge
from vella.runtime import Cursor, Runtime

from ._drain import bounded_drain
from ._fold import authority_pass, seed_state
from ._index import GraphIndex
from .mode import MaterializationMode


class _RefreshResult:
    """The output of one delta refresh: new index, new high-water, resident bodies."""

    def __init__(
        self,
        *,
        index: GraphIndex,
        high_water: Optional[Cursor],
        resident: dict[UUID, Any],
    ) -> None:
        """Bundle a refresh's outputs (see attributes)."""
        self.index = index
        self.high_water = high_water
        # full mode only: id -> Node|Edge body for the refreshed live set. Empty lean.
        self.resident = resident


async def refresh_index(
    runtime: Runtime,
    tenant_id: str,
    current: GraphIndex,
    high_water: Optional[Cursor],
    *,
    mode: MaterializationMode,
    weight: Optional[Callable[[Edge[Any, Any]], float]],
) -> "_RefreshResult":
    """Drain the delta after ``high_water`` and copy-on-write a new index.

    Seeds a ``_FoldState`` from ``current`` (its live edge ids + live node ids), then
    bounded-drains ``observe(since=high_water)`` — the opaque token passed straight
    through, never compared — so the seeded live set evolves by exactly the delta's
    transitions. Runs the shared authority pass (``get()`` per live entity) and folds
    the full recomputed live set into a new index via ``current.apply_delta``, which
    shares untouched buckets by identity. ``current`` is never mutated.

    Args:
        runtime: The runtime to drain/read (read-only ``observe``/``get``).
        tenant_id: The tenant this view projects; foreign-tenant entries skipped.
        current: The index being refreshed (read for the seed live set; not mutated).
        high_water: The opaque resume token passed to ``observe(since=)``; ``None``
            re-drains from the start (an empty original fold). Never compared.
        mode: ``"full"`` holds bodies resident; ``"lean"`` holds none.
        weight: Optional pure ``Edge -> float`` baked onto each refreshed edge record.

    Returns:
        A :class:`_RefreshResult` with the copy-on-write index, the new high-water
        cursor (the last delta entry's, or ``high_water`` unchanged on an empty
        delta), and (``full`` only) the resident bodies.
    """
    # Seed the live set from the index being refreshed so delta deletes/unlinks
    # remove prior-live entities and creates/links add new ones. Copy the sets so
    # the fold never mutates the prior index's tracking.
    state = seed_state(
        tenant_id,
        live_nodes=set(current.node_types),
        live_edges=set(current.live_edges),
        high_water=high_water,
    )
    # The opaque resume token goes STRAIGHT to observe — never compared.
    stream = runtime.observe(since=high_water)
    try:
        await bounded_drain(
            stream,
            sink=state.apply,
            on_caught_up=lambda: None,
        )
    finally:
        await stream.aclose()

    records, node_types, resident = await authority_pass(
        runtime, tenant_id, state, mode=mode, weight=weight
    )
    index = current.apply_delta(records, node_types)
    return _RefreshResult(
        index=index,
        high_water=state.high_water,
        resident=resident,
    )
