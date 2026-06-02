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

from typing import TYPE_CHECKING, Any, Generic, Literal, Optional, Union

from pydantic import SerializeAsAny

from .base import UTCDatetime, VellaModel, utcnow
from .errors import VellaError
from ._typevars import TState

if TYPE_CHECKING:
    from typing_extensions import Self

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


class StatefulEnvelope(Generic[TState]):
    """
    Copy-on-write state helpers shared by Node and Edge.

    A behavior-only mixin (NOT a model): it adds no fields, so it cannot affect
    either model's serialized schema. It operates on the discriminated ``state``
    field and the re-validating ``evolve`` that both Node and Edge already
    declare — making "edges are full peers of nodes" structural rather than a
    copy-pasted promise. Generic over the (BaseModel-bound) state type so the
    helpers stay precisely typed (``type(st.value).model_validate`` is known,
    not erased to ``Any``).
    """

    if TYPE_CHECKING:
        # Provided by the concrete model (Node/Edge); declared here only so the
        # helpers type-check. Absent at runtime, so pydantic never collects these
        # as fields.
        state: Optional[Union[Overlay[TState], Actuator[TState]]]

        def evolve(self, **updates: Any) -> Self: ...

    def update_state(self, **partial: object) -> Self:
        """Structural-merge ``partial`` into an Overlay's value; returns a new instance."""
        st = self.state
        if not isinstance(st, Overlay):
            raise VellaError("update_state requires Overlay state; use update_desired for Actuator.")
        new_value = type(st.value).model_validate({**dict(st.value), **partial})
        return self.evolve(state=Overlay(value=new_value, last_updated_at=utcnow()))

    def update_desired(self, **partial: object) -> Self:
        """
        Idempotent, level-triggered: structural-merge ``partial`` into the
        Actuator's desired *state* (declarative target, not a command). Returns a
        new instance; the reconciliation loop converges ``current`` toward it.
        """
        st = self.state
        if not isinstance(st, Actuator):
            raise VellaError("update_desired requires Actuator state; use update_state for Overlay.")
        base = st.desired if st.desired is not None else st.current
        new_desired = type(base).model_validate({**dict(base), **partial})
        return self.evolve(
            state=Actuator(
                current=st.current,
                desired=new_desired,
                last_updated_at=st.last_updated_at,
                last_desired_at=utcnow(),
            )
        )


__all__ = ["Overlay", "Actuator", "StatefulEnvelope"]
