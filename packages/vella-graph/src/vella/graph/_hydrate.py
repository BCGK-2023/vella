"""Body hydration — the mode-dependent half of a query (internal, M3).

Topology (ids) is mode-independent; *bodies* are where ``MaterializationMode``
matters, and this module is the only place the difference lives:

* **full** — bodies were read once at fold time and held resident on the view. A
  hydrate is a pure in-memory lookup, pinned to the fold-time snapshot: an entity
  edited / deleted AFTER the fold still hydrates to its fold-time body.
* **lean** — no bodies are held. A hydrate goes through a bounded
  :class:`LeanHydrator` LRU (an ``OrderedDict`` of capacity ``lru_capacity``,
  move-to-end on access, evict-oldest on insert) with a ``runtime.get()`` on miss,
  so it reflects LIVE state and costs a store round-trip on a cold key.

Both return ``None`` for an absent / dangling / deleted id (``full``: the id was
never resident; ``lean``: ``get()`` returns ``None``). This is the documented
per-mode body difference — explicitly excluded from the topology-equivalence claim.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional
from uuid import UUID

from vella.runtime import Runtime


class FullHydrator:
    """Fold-pinned hydration over the ``full``-mode resident body map.

    Holds the bodies the fold read once and pinned; a lookup is a pure dict ``get``
    (zero store round-trips), pinned to the fold-time snapshot.
    """

    def __init__(self, resident: dict[UUID, Any]) -> None:
        """Wrap the fold-pinned resident map (id -> ``Node``/``Edge`` body)."""
        self._resident = resident

    async def get(
        self, _runtime: Optional[Runtime], _tenant_id: str, entity_id: UUID
    ) -> Optional[Any]:
        """Return the fold-pinned body for ``entity_id``, or ``None`` if not resident.

        Args:
            _runtime: Unused in ``full`` mode (bodies are already resident); accepted
                so the hydrator interface matches ``lean``.
            _tenant_id: Unused in ``full`` mode; accepted for interface symmetry.
            entity_id: The id to hydrate.

        Returns:
            The fold-pinned ``Node``/``Edge`` body, or ``None`` for an id that was
            never resident (dangling / deleted at fold time).
        """
        return self._resident.get(entity_id)


class LeanHydrator:
    """Live ``lean``-mode hydration through a bounded LRU + ``get()``-on-miss.

    The LRU is an ``OrderedDict`` of capacity ``lru_capacity``: a hit moves the key
    to the most-recently-used end; a miss reads ``runtime.get()`` (LIVE state) and
    inserts, evicting the least-recently-used key when over capacity. ``None``
    results (absent / deleted ids) are NOT cached — a later create would be missed.
    """

    def __init__(self, capacity: int) -> None:
        """Start an empty LRU bounded at ``capacity`` entries."""
        self._capacity = capacity
        self._cache: "OrderedDict[UUID, Any]" = OrderedDict()

    async def get(
        self, runtime: Optional[Runtime], tenant_id: str, entity_id: UUID
    ) -> Optional[Any]:
        """Return the live body for ``entity_id`` via the LRU, reading on a miss.

        Args:
            runtime: The runtime to read authority from on a cache miss; ``None``
                only for a body-free view, which can hydrate to ``None`` only.
            tenant_id: The tenant to read under (never crosses tenants).
            entity_id: The id to hydrate.

        Returns:
            The live ``Node``/``Edge`` body, or ``None`` if the runtime no longer
            holds it (deleted / never created — a dangling id) or no runtime is held.
        """
        if runtime is None:
            return None
        if entity_id in self._cache:
            self._cache.move_to_end(entity_id)
            return self._cache[entity_id]
        body = await runtime.get(tenant_id, entity_id)
        if body is None:
            return None
        self._cache[entity_id] = body
        self._cache.move_to_end(entity_id)
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return body
