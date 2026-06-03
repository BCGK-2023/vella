"""The controller: the watch / worker / resync driver.

:class:`Reconciler` is the three-task control loop (decision A1) — a watch/fold
task draining ``runtime.observe(since)`` into the work-set and dedup queue, a
single-flight worker task draining the queue, and a resync ticker on the injected
:class:`~vella.reconciler.clock.Clock`. Idle is an explicit, race-free predicate
(decision B1), ``step()`` is non-blocking, and ``run(max_steps=...)`` bounds worker
iterations.

The full driver and reconcile policy are implemented in milestone M5; this module
declares the public controller so the surface tripwire can baseline it from M1.
"""

from __future__ import annotations

from typing import Optional

from vella.runtime import Runtime

from .clock import Clock
from .cursor_store import CursorStore
from .deadletter import DeadLetterStore
from .registry import Registry


class Reconciler:
    """Watch/worker/resync controller over the runtime contract.

    The controller observes the log, computes drift against fresh ``get`` reads,
    and drives convergent actions through the runtime's write verbs. It owns no
    storage and no clock; both are injected. The driver body lands in M5.
    """

    def __init__(
        self,
        runtime: Runtime,
        registry: Registry,
        clock: Clock,
        cursor_store: CursorStore,
        deadletter_store: DeadLetterStore,
        *,
        resync_interval: float,
        backoff: float,
    ) -> None:
        """Wire the controller to its injected runtime, registry, and seams.

        Args:
            runtime: The runtime contract the loop reads and writes through.
            registry: The entity-kind -> handler map the worker dispatches with.
            clock: The injected time source for resync ticks and backoff.
            cursor_store: Persists the resume cursor handed to ``observe``.
            deadletter_store: Records keys the loop has given up on.
            resync_interval: Seconds of clock time between resync ticks.
            backoff: The backoff budget governing requeue/give-up.
        """
        self._runtime = runtime
        self._registry = registry
        self._clock = clock
        self._cursor_store = cursor_store
        self._deadletter_store = deadletter_store
        self._resync_interval = resync_interval
        self._backoff = backoff

    async def step(self) -> None:
        """Process at most one queued work item without blocking (M5).

        Raises:
            NotImplementedError: The non-blocking worker step lands in M5.
        """
        raise NotImplementedError("Reconciler.step lands in M5")

    async def run(self, max_steps: Optional[int] = None) -> None:
        """Drive watch + worker + resync to convergence (M5).

        Args:
            max_steps: Optional bound on worker iterations; ``None`` runs until
                idle.

        Raises:
            NotImplementedError: The driver lands in M5.
        """
        raise NotImplementedError("Reconciler.run lands in M5")

    async def drain(self) -> None:
        """Re-enqueue dead-lettered keys for an explicit retry pass (M5).

        Raises:
            NotImplementedError: The drain re-entry path lands in M5.
        """
        raise NotImplementedError("Reconciler.drain lands in M5")
