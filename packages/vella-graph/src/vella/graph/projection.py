"""The fold builder — ``GraphProjection`` (M2).

``GraphProjection`` is the entry point: it folds a runtime's ``observe()`` stream
into a frozen :class:`~vella.graph.GraphView` for one tenant. It is stateless (it
holds no view); each ``fold`` is an independent bounded-drain + authority pass.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from vella.core import Edge
from vella.runtime import Runtime

from ._fold import fold as _fold
from .mode import MaterializationMode
from .view import GraphView


class GraphProjection:
    """Builds a per-tenant :class:`~vella.graph.GraphView` from a runtime log.

    Stateless and reusable: call :meth:`fold` once per snapshot. The projection
    reads the runtime through ``observe`` and ``get`` only — it never writes.

    Examples:
        >>> from vella.graph import GraphProjection
        >>> projection = GraphProjection()
        >>> type(projection).__name__
        'GraphProjection'
    """

    async def fold(
        self,
        runtime: Runtime,
        tenant_id: str,
        *,
        mode: MaterializationMode = "full",
        weight: Optional[Callable[[Edge[Any, Any]], float]] = None,
        lru_capacity: int = 1024,
    ) -> GraphView:
        """Fold the runtime's log into a frozen view for ``tenant_id``.

        Bounded-drains ``observe(since=None)`` to the live edge (never blocking),
        computing the live id-set from typed ``LogEntry`` fields only (TRAP-1), then
        reads each live entity's authority via ``get()`` once (edges always; nodes
        only in ``full`` mode) and builds the always-built adjacency index.

        Args:
            runtime: The runtime whose log is projected (read-only).
            tenant_id: The tenant to project; other tenants' entries are ignored.
            mode: ``"full"`` holds node+edge bodies resident (fold-pinned, zero
                round-trips on hydrate); ``"lean"`` holds none (LRU + ``get()``
                on demand, landing in M3). Topology is identical in both.
            weight: Optional pure ``Edge -> float`` baked onto each edge record at
                fold time for weighted shortest path (M4); ``None`` bakes ``0.0``.
            lru_capacity: The bound for the ``lean`` hydration LRU (M3); accepted
                now so the signature is frozen by the surface tripwire from M2.

        Returns:
            A frozen :class:`~vella.graph.GraphView` at the current high-water
            cursor.
        """
        result = await _fold(runtime, tenant_id, mode=mode, weight=weight)
        return GraphView(
            index=result.index,
            mode=mode,
            high_water=result.high_water,
            resident=result.resident,
            tenant_id=tenant_id,
            runtime=runtime,
            weight=weight,
            lru_capacity=lru_capacity,
        )


__all__ = ["GraphProjection"]
