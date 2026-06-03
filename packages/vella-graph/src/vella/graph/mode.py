"""The materialization mode — residency, not results.

A :data:`MaterializationMode` is chosen once, at ``GraphProjection.fold(...)``, and
controls ONLY where entity bodies come from. It never changes the topology the
index records, so every topology query (neighbour ids, traversal, shortest path)
is byte-identical across modes — a load-bearing verifier invariant.
"""

from __future__ import annotations

from typing import Literal

MaterializationMode = Literal["full", "lean"]
"""Body residency for a folded :class:`~vella.graph.GraphView`.

* ``"full"`` — every live ``Node`` and ``Edge`` body is read once at fold time and
  held resident; later hydration is a pure in-memory lookup (zero store
  round-trips), pinned to the fold-time snapshot.
* ``"lean"`` — bodies are NOT held; only edge endpoints/type/weight are baked into
  the index. Hydration goes through a bounded LRU via ``get()`` on demand (M3), so
  it reflects live state and costs a store round-trip on a miss.

The index itself is always built identically in both modes — mode is *residency,
not results*. Hydrated body *contents* are mode-dependent by design (``full`` =
fold-pinned, ``lean`` = live) and are explicitly NOT part of the topology
equivalence claim.

Example:
    >>> from vella.graph import MaterializationMode
    >>> import typing
    >>> sorted(typing.get_args(MaterializationMode))
    ['full', 'lean']
"""

__all__ = ["MaterializationMode"]
