"""References — pointers to other entities inside data.

References move through two type-distinct phases: ``unresolved`` (just an
identifier from the upstream system) and ``resolved`` (linked to a concrete
node). The discriminated union forces the resolution check at the type level
rather than scattering ``if ref.node_id is not None`` guards everywhere.

Provenance carries confidence/verification metadata on a resolved reference, so
"show me low-confidence resolutions for review" is a typed query. The same
machinery backs self-healing (a quarantined node carries a low-confidence
Provenance marker — see ``parse``).
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union
from uuid import UUID

from pydantic import Field

from .base import UTCDatetime, VellaModel

ResolutionMethod = Literal[
    "exact_match",
    "fuzzy_match",
    "agent_inference",
    "human_verified",
    "system",
]
"""How a reference was resolved to a concrete node — recorded on a ``ResolvedRef``.

Ranges from high-confidence (``exact_match``) to lower-confidence
(``fuzzy_match``, ``agent_inference``), with ``human_verified`` and ``system``
marking provenance for audit and re-resolution decisions.
"""


class Provenance(VellaModel):
    """Confidence and verification metadata for a resolved reference."""

    confidence: float = Field(ge=0.0, le=1.0)
    method: ResolutionMethod
    verified_by: Optional[Union[UUID, "UnresolvedRef"]] = None
    verified_at: Optional[UTCDatetime] = None


class UnresolvedRef(VellaModel):
    """A pointer to an entity not yet linked to a node.

    Integrations write these during ingestion using whatever the upstream
    system gave them; a mapper agent later transitions one into a ResolvedRef
    via ``resolve``.
    """

    resolution: Literal["unresolved"] = "unresolved"
    identifier: str
    label: Optional[str] = None
    kind: Optional[str] = None        # "person", "channel", "document", ...
    edge_type: Optional[str] = None   # hint for the edge to create on resolution

    def resolve(
        self, node_id: UUID, provenance: Optional["Provenance"] = None
    ) -> "ResolvedRef":
        """Transition this reference to a resolved one pointing at ``node_id``."""
        return ResolvedRef(
            identifier=self.identifier,
            node_id=node_id,
            label=self.label,
            kind=self.kind,
            edge_type=self.edge_type,
            provenance=provenance,
        )


class ResolvedRef(VellaModel):
    """A reference resolved to a concrete node.

    ``identifier`` is kept alongside ``node_id`` for traceability. The resolver
    is also expected to create the edge described by ``edge_type`` (if set)
    between the container and ``node_id``.
    """

    resolution: Literal["resolved"] = "resolved"
    identifier: str
    node_id: UUID
    label: Optional[str] = None
    kind: Optional[str] = None
    edge_type: Optional[str] = None
    provenance: Optional[Provenance] = None


Reference = Annotated[
    Union[UnresolvedRef, ResolvedRef],
    Field(discriminator="resolution"),
]
"""Reference field that may carry either phase — unresolved or resolved.

Use in data models for a field that may hold either an ``UnresolvedRef`` or a
``ResolvedRef``. Pydantic discriminates on the ``resolution`` literal; consumers
should pattern-match (``isinstance`` / ``match``) to reach ``node_id`` safely.
"""


# --- Convenience constructors (all produce UnresolvedRef) --------------------
def _ref(identifier: str, label: Optional[str], kind: str, edge: Optional[str]) -> UnresolvedRef:
    return UnresolvedRef(identifier=identifier, label=label, kind=kind, edge_type=edge)


def person_ref(identifier: str, label: Optional[str] = None, edge: Optional[str] = None) -> UnresolvedRef:
    """Reference to a person. ``identifier`` is typically an email or username."""
    return _ref(identifier, label, "person", edge)


def channel_ref(identifier: str, label: Optional[str] = None, edge: Optional[str] = None) -> UnresolvedRef:
    """Reference to a channel (Slack, Discord, etc.)."""
    return _ref(identifier, label, "channel", edge)


def document_ref(identifier: str, label: Optional[str] = None, edge: Optional[str] = None) -> UnresolvedRef:
    """Reference to a document, file, or attachment."""
    return _ref(identifier, label, "document", edge)


def email_ref(identifier: str, label: Optional[str] = None, edge: Optional[str] = None) -> UnresolvedRef:
    """Reference to an email or email-like message."""
    return _ref(identifier, label, "email", edge)


def location_ref(identifier: str, label: Optional[str] = None, edge: Optional[str] = None) -> UnresolvedRef:
    """Reference to a location, place, or room."""
    return _ref(identifier, label, "location", edge)


__all__ = [
    "ResolutionMethod",
    "Provenance",
    "UnresolvedRef",
    "ResolvedRef",
    "Reference",
    "person_ref",
    "channel_ref",
    "document_ref",
    "email_ref",
    "location_ref",
]
