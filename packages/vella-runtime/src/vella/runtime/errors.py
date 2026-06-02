"""Runtime exception hierarchy.

Runtime errors subclass core's ``VellaError`` root, so the whole stack shares
one catchable tree — a consumer can ``except VellaError`` to catch everything
the SDK raises, core and runtime alike.

Errors carry structured attributes (not just a message string) so callers and
self-healing flows can branch programmatically instead of parsing English.
"""

from __future__ import annotations

from vella.core import VellaError


class ConcurrencyConflict(VellaError):
    """Raised when ``edit``/``set_desired`` finds ``version != expected_version``.

    Callers (and the future reconciler) implement retry-on-conflict.
    Part of the runtime's public surface — snapshotted by the tripwire.
    """


__all__ = [
    "ConcurrencyConflict",
]
