"""Shared primitives for the core data model.

The frozen base model, tenancy default, the tz-aware datetime type, the
permissive base for agent-managed types, and node flags.

Enforcement posture (locked during design):
  * Every core model is **frozen** — changes go through copy-on-write
    (``evolve`` / ``update_state`` / ``update_desired``), which re-validate.
  * Strict models **forbid extra fields** so typos and injected data fail loud.
  * ``model_construct`` (Pydantic's validation-skipping fast path) is **locked**;
    the one blessed trusted door is ``hydrate`` (used by the storage layer for
    already-valid rows). Everything else goes through validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Mapping, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, PrivateAttr
from typing_extensions import Self

from .errors import VellaError

# --- Multi-tenancy -----------------------------------------------------------
# Partition isolation is structural, not optional: every Node/Edge belongs to
# exactly one tenant. Single-tenant / personal deployments use this default and
# never think about it; commercial deployments pass real tenant ids. There is
# never a null-tenant population to migrate. (Access-control beyond partitioning
# — scopes, redaction — is deferred; see DESIGN.md.)
DEFAULT_TENANT = "__local__"
"""Tenant id used when none is supplied — every Node/Edge belongs to exactly one tenant.

Single-tenant and personal deployments use this default and never think about it;
multi-tenant deployments pass real tenant ids. There is never a null-tenant
population to migrate.
"""


# --- Time --------------------------------------------------------------------
def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError(
            "datetime must be timezone-aware (UTC). Naive datetimes are rejected "
            "to prevent silent timezone bugs; use datetime.now(timezone.utc)."
        )
    return value.astimezone(timezone.utc)


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]
"""A ``datetime`` that must be timezone-aware; coerced to UTC at the boundary.

Naive datetimes are rejected at validation to prevent silent timezone bugs.
"""


def utcnow() -> datetime:
    """Current time as a tz-aware UTC datetime (the only clock core uses)."""
    return datetime.now(timezone.utc)


# --- Frozen base -------------------------------------------------------------
class VellaModel(BaseModel):
    """Frozen, strict base for all Vella core models."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The registry this instance was validated against (set by ``parse_*``), so
    # copy-on-write (``evolve``/``model_copy``) re-validates against the SAME
    # registry rather than silently falling back to the default. Not a field;
    # never serialized.
    _vella_registry: Any = PrivateAttr(default=None)

    def _validation_context(self) -> Optional[dict[str, Any]]:
        return {"registry": self._vella_registry} if self._vella_registry is not None else None

    def _carry_registry(self, other: Self) -> Self:
        if self._vella_registry is not None:
            other._vella_registry = self._vella_registry
        return other

    @classmethod
    def model_construct(cls, *args: Any, **kwargs: Any) -> Self:  # type: ignore[override]
        """Locked: raise rather than skip validation (use ``hydrate`` instead)."""
        raise VellaError(
            f"{cls.__name__}.model_construct is locked (it skips all validation). "
            f"Use {cls.__name__}.hydrate(...) for the trusted fast path from "
            f"storage, or normal construction / parse_* for validation."
        )

    @classmethod
    def hydrate(cls, **fields: Any) -> Self:
        """Build without validation from already-valid field objects.

        Trusted fast path (e.g. rows the storage layer just deserialized). This
        is the one explicit door past validation — do not use it for untrusted
        input; use ``parse_node`` / ``parse_edge`` for that.
        """
        return super().model_construct(**fields)

    def model_copy(self, *, update: Optional[Mapping[str, Any]] = None, deep: bool = False) -> Self:
        """Re-validating copy.

        Pydantic's stock ``model_copy`` skips all validation (a silent bypass of
        the frozen posture); we route the result back through validation so
        ``model_copy`` cannot inject an invalid state. Equivalent to a validated
        ``evolve``.
        """
        copied = super().model_copy(update=dict(update) if update else None, deep=deep)
        revalidated = type(self).model_validate(dict(copied), context=self._validation_context())
        return self._carry_registry(revalidated)

    def evolve(self, **updates: Any) -> Self:
        """Copy-on-write a new instance with ``updates`` applied, re-validating.

        Invariants and after-validators run against the same registry the
        instance was parsed with. Does not touch concurrency or timestamp fields —
        the runtime stamps ``version`` / ``updated_at`` on a successful write.
        """
        revalidated = type(self).model_validate(
            {**dict(self), **updates}, context=self._validation_context()
        )
        return self._carry_registry(revalidated)


# --- Permissive base for agent-managed types ---------------------------------
class FlexibleData(BaseModel):
    """Base for agent-managed node types — people, locations, concepts, memory.

    Allows arbitrary extra fields (``extra="allow"``); agents shape data as they
    judge best. Frozen like everything else, so it is changed via copy-on-write.
    System-managed types should define strict models (subclass ``VellaModel`` or
    set ``frozen=True``) instead. The off-ramp from Flexible to strict
    (crystallization) is deferred; see DESIGN.md.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    def evolve(self, **updates: Any) -> Self:
        """Copy-on-write a new instance with ``updates`` applied, re-validating."""
        return type(self).model_validate({**dict(self), **updates})


# --- Flags -------------------------------------------------------------------
class NodeFlags(VellaModel):
    """Behavioral flags.

    Minimal on purpose — permissions are enforced at the action-classifier
    layer (runtime), not via per-field flags here.
    """

    system_protected: bool = False
    user_private: bool = False


__all__ = [
    "DEFAULT_TENANT",
    "UTCDatetime",
    "utcnow",
    "VellaModel",
    "FlexibleData",
    "NodeFlags",
]
