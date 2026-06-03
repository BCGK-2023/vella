"""The read-only handle a reconcile handler receives.

:class:`Context` exposes exactly the two injected seams a handler legitimately
needs — the :class:`~vella.runtime.Runtime` write/read contract and the
:class:`~vella.reconciler.clock.Clock` — and nothing about the driver's internal
work-set or queue. Handlers compute drift from a FRESH ``runtime.get(...)``, never
from a folded ``LogEntry``.
"""

from __future__ import annotations

from vella.runtime import Runtime

from .clock import Clock


class Context:
    """Read-only handle passed to a reconcile handler.

    Attributes are the injected runtime and clock; the driver's work-set internals
    are deliberately not exposed.
    """

    def __init__(self, runtime: Runtime, clock: Clock) -> None:
        """Bind the injected runtime and clock for one handler invocation.

        Args:
            runtime: The runtime contract the handler reads from and writes to.
            clock: The injected time source shared with the driver.
        """
        self._runtime = runtime
        self._clock = clock

    @property
    def runtime(self) -> Runtime:
        """The runtime contract the handler reconciles against.

        Returns:
            The injected :class:`~vella.runtime.Runtime`.
        """
        return self._runtime

    @property
    def clock(self) -> Clock:
        """The injected time source shared with the driver.

        Returns:
            The injected :class:`~vella.reconciler.clock.Clock`.
        """
        return self._clock
