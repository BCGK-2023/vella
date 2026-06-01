"""
State — Overlay (plain mutable) and Actuator (current/desired pair),
discriminated by ``kind``.

  * Overlay[T]   — the 80% case: any property that changes more often than the
    core data and has no actuator semantics (email is_read, task status,
    sensor readings, edge measurements).
  * Actuator[T]  — a current/desired pair, for things that can be *commanded*.
    ``desired`` is declarative full state, not a patch or a command. A
    reconciliation loop in the runtime converges ``current`` toward ``desired``
    (level-triggered, so it is robust to missed events and self-heals on
    restart). Use ``Node.update_desired`` for idempotent partial target updates.
"""

from __future__ import annotations

from typing import Generic, Literal, Optional

from pydantic import SerializeAsAny

from .base import UTCDatetime, VellaModel
from ._typevars import TState

# Polymorphic slots use SerializeAsAny so the *actual* state object is
# serialized, not the (erased) declared TypeVar — otherwise a bare/loosely
# parametrized container would dump its value as an empty object.


class Overlay(VellaModel, Generic[TState]):
    """Plain mutable state. Read with ``node.state.value.<field>``."""

    kind: Literal["overlay"] = "overlay"
    value: SerializeAsAny[TState]
    last_updated_at: Optional[UTCDatetime] = None


class Actuator(VellaModel, Generic[TState]):
    """
    Current/desired pair. ``current`` is ground truth from the world; ``desired``
    is the full target state Vella wants ``current`` to become. Read with
    ``node.state.current.<field>`` / ``node.state.desired.<field>``.
    """

    kind: Literal["actuator"] = "actuator"
    current: SerializeAsAny[TState]
    desired: Optional[SerializeAsAny[TState]] = None
    last_updated_at: Optional[UTCDatetime] = None
    last_desired_at: Optional[UTCDatetime] = None


__all__ = ["Overlay", "Actuator"]
