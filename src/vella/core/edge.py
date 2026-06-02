"""
Edge — a typed, directed relationship between two nodes.

Edges are full peers of nodes for integration purposes (they carry
``integrations``, ``tool_overrides``, ``extra_tools``) and are polymorphic over
both ``data`` and ``state``. They share the same copy-on-write state helpers as
nodes (``update_state`` / ``update_desired``, via ``StatefulEnvelope``). Use
``EdgeTypes`` constants where they fit; custom strings are allowed but trigger a
did-you-mean warning to catch typos.

Edge dedup/cardinality ("an Invoice has exactly one OWNED_BY edge") is deferred
to the graph-invariants work — see DESIGN.md.
"""

from __future__ import annotations

import difflib
import warnings
from typing import Annotated, Generic, Optional, Union
from uuid import UUID

from pydantic import Field, SerializeAsAny, ValidationInfo, model_validator
from typing_extensions import Self

from ._typevars import TEdgeData, TEdgeState
from .base import DEFAULT_TENANT, NodeFlags, UTCDatetime, VellaModel, utcnow
from ._uuid7 import uuid7
from .errors import UnknownEdgeTypeWarning
from .integration import IntegrationBinding
from .references import UnresolvedRef
from .state import Actuator, Overlay, StatefulEnvelope
from .tooling import (
    ToolDeclaration,
    ToolOverride,
    registry_from_context,
    validate_tool_overrides,
)


class EdgeTypes:
    """Canonical edge type constants — conventions, not enforced."""

    # Structural / containment
    PART_OF = "part_of"
    CONTAINS = "contains"
    PART_OF_DEVICE = "part_of_device"
    # Spatial
    LOCATED_IN = "located_in"
    # Provenance / source
    PRODUCED_BY = "produced_by"
    MENTIONED_IN = "mentioned_in"
    REFERENCES = "references"
    # Social / participation
    KNOWS = "knows"
    HAS_ATTENDEE = "has_attendee"
    OWNED_BY = "owned_by"
    # Communication
    SENT_TO = "sent_to"
    CC_TO = "cc_to"
    BCC_TO = "bcc_to"
    REPLY_TO = "reply_to"
    # Attached content
    HAS_DOCUMENT = "has_document"
    HAS_CHILD_ENTITY = "has_child_entity"
    HAS_ATTACHMENT = "has_attachment"


def known_edge_types() -> set[str]:
    """All canonical edge type strings declared on ``EdgeTypes``."""
    return {
        v for k, v in EdgeTypes.__dict__.items()
        if not k.startswith("_") and isinstance(v, str)
    }


class Edge(VellaModel, StatefulEnvelope[TEdgeState], Generic[TEdgeData, TEdgeState]):
    id: UUID = Field(default_factory=uuid7)
    type: str

    from_node_id: UUID
    to_node_id: UUID

    name: Optional[str] = None
    description: Optional[str] = None

    # --- Concurrency / schema / tenancy (mirror Node) ---
    version: int = 1
    etag: Optional[str] = None
    schema_version: int = 1
    tenant_id: str = DEFAULT_TENANT

    # --- Provenance ---
    created_at: UTCDatetime = Field(default_factory=utcnow)
    created_by: Union[UUID, UnresolvedRef]
    updated_at: UTCDatetime = Field(default_factory=utcnow)

    # --- Polymorphic body and state ---
    data: Optional[SerializeAsAny[TEdgeData]] = None
    state: Optional[
        Annotated[Union[Overlay[TEdgeState], Actuator[TEdgeState]], Field(discriminator="kind")]
    ] = None

    # --- Tool overrides / extras / surfaces (mirror Node) ---
    tool_overrides: list[ToolOverride] = Field(default_factory=list[ToolOverride])
    extra_tools: list[ToolDeclaration] = Field(default_factory=list[ToolDeclaration])
    integrations: list[IntegrationBinding] = Field(default_factory=list[IntegrationBinding])
    flags: NodeFlags = Field(default_factory=NodeFlags)

    @model_validator(mode="after")
    def _check_tool_overrides(self, info: ValidationInfo) -> Self:
        validate_tool_overrides(
            self.type, self.tool_overrides, registry=registry_from_context(info.context)
        )
        return self

    @model_validator(mode="after")
    def _warn_on_unknown_edge_type(self) -> Self:
        known = known_edge_types()
        if self.type in known:
            return self
        suggestions = difflib.get_close_matches(self.type, known, n=3, cutoff=0.6)
        if suggestions:
            hint = ", ".join(repr(s) for s in suggestions)
            warnings.warn(
                f"Edge type {self.type!r} is not a canonical EdgeTypes constant. "
                f"Did you mean: {hint}? Custom types are allowed; this is a heads-up.",
                UnknownEdgeTypeWarning,
                stacklevel=2,
            )
        return self


__all__ = ["EdgeTypes", "known_edge_types", "Edge"]
