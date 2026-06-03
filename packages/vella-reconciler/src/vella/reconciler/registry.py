"""Entity-kind -> reconcile-handler mapping.

:class:`Registry` is where callers register one async handler per entity kind. The
worker looks a handler up by the entity's kind at dispatch time; an unregistered
kind is an explicit miss (the worker skips it, it never crashes).
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

from .context import Context
from .result import ReconcileResult

Handler = Callable[[Context], Awaitable[ReconcileResult]]
"""An async reconcile handler: given a :class:`Context`, returns a verdict."""


class Registry:
    """Maps an entity kind to the async handler that reconciles it."""

    def __init__(self) -> None:
        """Create an empty registry with no handlers registered."""
        self._handlers: dict[str, Handler] = {}

    def register(self, kind: str, handler: Handler) -> None:
        """Register ``handler`` as the reconcile handler for ``kind``.

        Args:
            kind: The entity kind the handler reconciles.
            handler: The async handler invoked for entities of that kind.
        """
        self._handlers[kind] = handler

    def lookup(self, kind: str) -> Optional[Handler]:
        """Return the handler registered for ``kind``, or ``None`` on a miss.

        Args:
            kind: The entity kind to resolve.

        Returns:
            The registered handler, or ``None`` if the kind is unregistered (an
            explicit miss the worker skips rather than treating as an error).
        """
        return self._handlers.get(kind)
