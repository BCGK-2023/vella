"""Vella SDK runtime substrate.

The physics/state layer the rest of Vella runs on: the append-only log, the
optimistic-concurrency store, and the transition verbs that move graph state
forward. Where ``vella.core`` is pure, frozen data with no behavior, ``vella.runtime``
is the substrate that records, serializes, and concurrency-controls changes to
that data.

Design principles
-----------------
* **Deterministic serialization.** Any set-derived value that gets serialized is
  ``sorted()``; reproducible artifacts never depend on hash-seed iteration order.
* **Optimistic concurrency.** Read-modify-write verbs check ``expected_version``
  inside a transactional scope and raise ``ConcurrencyConflict`` on mismatch;
  callers retry.
* **Same front door for everyone.** Internal and external consumers use this exact
  surface — no privileged internal API.
* **Depends downward only.** Runtime depends on ``vella.core``; core never depends
  on runtime.

The public surface grows milestone by milestone; everything in ``__all__`` is
importable, documented, and snapshotted by the surface tripwire.
"""

from __future__ import annotations

from .errors import ConcurrencyConflict

__all__ = [
    "ConcurrencyConflict",
]
