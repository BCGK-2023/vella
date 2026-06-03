"""Vella reconciliation loop (controller-runtime over the runtime contract).

Where ``vella.runtime`` is *physics* — the append-only log, the
optimistic-concurrency store, and the write verbs that move world state forward —
``vella.reconciler`` is a *control loop* on top of it. It observes the log,
computes drift (desired vs. current), and drives convergent actions back through
the runtime's write contract. It owns no storage and no clock of its own; both are
injected.

Design principles
-----------------
* **The runtime is physics; the reconciler is a control loop.** The reconciler
  reads the log and reconciles toward desired state through the runtime's verbs.
  It never persists world state and never imports a private runtime symbol.
* **Determinism is a property, not a hope.** Every ordering — resync ticks,
  backoff wakes, dead-letter serialization — is explicit and reproducible under an
  injected :class:`ManualClock`. Any set-derived serialized value is ``sorted()``.
* **Depend downward only.** The reconciler imports only the published
  ``vella.runtime`` and ``vella.core`` surfaces; both layers are unaware of it.
* **Idle is observable.** Convergence ("the loop goes quiet") is a predicate the
  driver evaluates, never an emergent side effect of a loop that happened not to
  fire.

The public surface grows milestone by milestone; everything in ``__all__`` is
importable, documented, and snapshotted by the surface tripwire from M1 onward.
The concrete implementations of the controller land in later milestones; the
surface is baselined now so the tripwire guards it from the start.
"""

from __future__ import annotations

from .clock import Clock, ManualClock
from .context import Context
from .cursor_store import CursorStore, InMemoryCursorStore
from .deadletter import (
    DeadLetterRecord,
    DeadLetterStore,
    InMemoryDeadLetterStore,
)
from .reconciler import Reconciler
from .registry import Registry
from .result import ReconcileResult

__all__ = [
    "Clock",
    "Context",
    "CursorStore",
    "DeadLetterRecord",
    "DeadLetterStore",
    "InMemoryCursorStore",
    "InMemoryDeadLetterStore",
    "ManualClock",
    "ReconcileResult",
    "Reconciler",
    "Registry",
]
