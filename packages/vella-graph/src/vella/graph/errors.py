"""Public, typed errors raised by the graph projection (M4).

The graph derives its errors from :class:`vella.core.VellaError` — the published
core base — so a caller can catch every Vella-layer error with one ``except``
clause regardless of which layer raised it (the dependency direction is downward:
the graph depends on core, never the reverse).

The single error here guards the one place a query's contract is mode-dependent: a
per-query weight override re-reads edge *bodies* to compute weights at query time,
which only a ``full``-mode view holds resident. A ``lean`` view would have to round-
trip every frontier edge through the store mid-traversal (and a deleted-after-fold
edge would silently drop), so the override fails closed with
:class:`WeightOverrideRequiresFullMode` rather than degrading the result quietly.
"""

from __future__ import annotations

from vella.core import VellaError


class WeightOverrideRequiresFullMode(VellaError):
    """A per-query weight override was supplied to a ``lean``-mode view.

    Weighted shortest path over the *baked* edge weights (the floats baked into the
    index at fold time) is mode-equivalent: it reads only the in-memory index and
    works identically in ``full`` and ``lean``. A per-query ``weight`` override,
    however, recomputes weights from live edge *bodies*, which only ``full`` mode
    holds resident — so it is full-mode only and a ``lean`` view raises this rather
    than silently falling back to the baked weights or round-tripping the store.

    This is the documented exclusion from the topology/weight mode-equivalence
    claim: baked weighted SP is mode-equivalent; an override is not (it raises in
    lean).
    """
