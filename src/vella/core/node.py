"""
Node — the single, unified envelope for everything Vella knows about.

Pure data: identity, concurrency (version/etag), schema versioning, tenancy,
provenance, a polymorphic ``data`` body, an optional ``state`` overlay/actuator,
tool overrides/extras, integrations, embedding, flags. Behavior lives in the
integration API; the only logic here is construction-time validation and
copy-on-write helpers.

Obtain a node through exactly one of four doors:
    Node(type=..., data=..., ...)   strict, validated            (new nodes)
    node.evolve(state=...)          validated copy-on-write       (changes)
    parse_node(raw)                 tolerant registry-driven       (untyped wire)
    Node.hydrate(**fields)          trusted fast path              (our own store)
"""

from __future__ import annotations

from typing import Annotated, Any, Generic, Optional, Union
from uuid import UUID

from pydantic import Field, SerializeAsAny, ValidationInfo, model_validator
from typing_extensions import Self

from ._typevars import TData, TState
from .base import (
    DEFAULT_TENANT,
    NodeFlags,
    UTCDatetime,
    VellaModel,
    utcnow,
)
from ._uuid7 import uuid7
from .embedding import Embedding
from .errors import VellaError
from .integration import IntegrationBinding
from .references import UnresolvedRef
from .state import Actuator, Overlay, StatefulEnvelope
from .tooling import (
    ToolDeclaration,
    ToolOverride,
    registry_from_context,
    validate_tool_overrides,
)


class Node(VellaModel, StatefulEnvelope[TState], Generic[TData, TState]):
    # --- Identity ---
    id: UUID = Field(default_factory=uuid7)
    type: str
    name: str
    description: Optional[str] = None

    # --- Concurrency (stamped by the runtime on write) ---
    version: int = 1
    etag: Optional[str] = None

    # --- Schema versioning (migration routing) ---
    schema_version: int = 1

    # --- Multi-tenancy: structural partition isolation, never null ---
    tenant_id: str = DEFAULT_TENANT

    # --- Provenance ---
    created_at: UTCDatetime = Field(default_factory=utcnow)
    created_by: Union[UUID, UnresolvedRef]
    updated_at: UTCDatetime = Field(default_factory=utcnow)

    # --- Polymorphic body (SerializeAsAny: dump the actual object, not the erased TypeVar) ---
    data: SerializeAsAny[TData]

    # --- Optional mutable overlay (discriminated on ``kind``) ---
    state: Optional[
        Annotated[Union[Overlay[TState], Actuator[TState]], Field(discriminator="kind")]
    ] = None

    # --- Tool overrides / extras ---
    tool_overrides: list[ToolOverride] = Field(default_factory=list[ToolOverride])
    extra_tools: list[ToolDeclaration] = Field(default_factory=list[ToolDeclaration])

    # --- Optional surfaces ---
    integrations: list[IntegrationBinding] = Field(default_factory=list[IntegrationBinding])
    embedding: Optional[Embedding] = None
    flags: NodeFlags = Field(default_factory=NodeFlags)

    @model_validator(mode="after")
    def _check_tool_overrides(self, info: ValidationInfo) -> Self:
        # Use the registry passed via parse context if present, else the default.
        validate_tool_overrides(
            self.type, self.tool_overrides, registry=registry_from_context(info.context)
        )
        return self

    # --- Construction --------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        data: VellaModel,
        *,
        name: str,
        created_by: Union[UUID, UnresolvedRef],
        description: Optional[str] = None,
        state: Optional[Union[Overlay[Any], Actuator[Any]]] = None,
        integrations: Optional[list[IntegrationBinding]] = None,
        embedding: Optional[Embedding] = None,
        tool_overrides: Optional[list[ToolOverride]] = None,
        extra_tools: Optional[list[ToolDeclaration]] = None,
        flags: Optional[NodeFlags] = None,
        tenant_id: str = DEFAULT_TENANT,
        schema_version: int = 1,
    ) -> "Node[Any, Any]":
        """Construct from a data instance whose class was registered via ``@node_type``."""
        type_name = getattr(type(data), "__vella_type__", None)
        if not type_name:
            raise VellaError(
                f"{type(data).__name__} is not a registered node type. Decorate it "
                f"with @node_type('your_type', ...) or construct Node with type=...."
            )
        return Node(
            type=type_name,
            name=name,
            description=description,
            created_by=created_by,
            data=data,
            state=state,
            integrations=integrations or [],
            embedding=embedding,
            tool_overrides=tool_overrides or [],
            extra_tools=extra_tools or [],
            flags=flags or NodeFlags(),
            tenant_id=tenant_id,
            schema_version=schema_version,
        )

    # Copy-on-write state helpers (update_state / update_desired) come from
    # StatefulEnvelope, shared verbatim with Edge.


__all__ = ["Node"]
