"""Vella SDK core data model.

The pure-data foundation everything in Vella is built on: ``Node``, ``Edge``,
references, state, tools, and the type registry. Zero first-party dependencies
(pydantic v2 + typing_extensions only), publishable standalone so anyone can
build integrations without the rest of the stack.

Design principles
-----------------
* Nodes and edges are **pure, frozen data**. Behavior lives in the integration
  API; the only logic here is construction-time validation and copy-on-write.
* **One envelope** across every type; polymorphic via generics. System-managed
  types declare strict models; agent-managed types use ``FlexibleData``.
* **Same front door for everyone** — internal and external integrations use this
  exact surface. No privileged internal API.
* Credentials never live on a node; an ``IntegrationBinding`` holds an opaque
  pointer into a separate secrets store.

Obtaining a node (four doors)
-----------------------------
* ``Node(type=..., data=..., ...)`` — strict, validated (new nodes).
* ``node.evolve(...)`` / ``update_state`` / ``update_desired`` — validated
  copy-on-write (changes); never touches ``version``/``updated_at``.
* ``parse_node(raw)`` — tolerant, registry-driven hydration from untyped data.
* ``Node.hydrate(**fields)`` — trusted fast path for already-valid storage rows.

Locked enforcement: models are frozen, strict models forbid extra fields, and
``model_construct`` is disabled in favor of ``hydrate``.

Deferred work (graph invariants, multi-tenant access control beyond partition
isolation, FlexibleData crystallization) is tracked in DESIGN.md.
"""

from __future__ import annotations

from .base import (
    DEFAULT_TENANT,
    FlexibleData,
    NodeFlags,
    UTCDatetime,
    VellaModel,
    utcnow,
)
from .edge import Edge, EdgeTypes, known_edge_types
from .embedding import Embedding
from .errors import (
    SchemaMigrationError,
    ToolOverrideError,
    UnknownEdgeTypeWarning,
    UnregisteredTypeError,
    VellaError,
    VellaWarning,
)
from .history import HistoryEntry
from .integration import IntegrationBinding, Surface
from .node import Node
from .parse import FlexibleEdge, FlexibleNode, parse_edge, parse_node
from .references import (
    Provenance,
    Reference,
    ResolutionMethod,
    ResolvedRef,
    UnresolvedRef,
    channel_ref,
    document_ref,
    email_ref,
    location_ref,
    person_ref,
)
from .state import Actuator, Overlay
from .tooling import (
    CompatPolicy,
    Migration,
    Registry,
    ToolDeclaration,
    ToolOverride,
    TypeSpec,
    default_registry,
    node_type,
    register_tools,
    tools_for,
)
from ._uuid7 import uuid7

__all__ = [
    # Envelopes
    "Node",
    "Edge",
    "EdgeTypes",
    "known_edge_types",
    "FlexibleNode",
    "FlexibleEdge",
    # Data bases
    "VellaModel",
    "FlexibleData",
    "NodeFlags",
    # References
    "Provenance",
    "ResolutionMethod",
    "UnresolvedRef",
    "ResolvedRef",
    "Reference",
    "person_ref",
    "channel_ref",
    "document_ref",
    "email_ref",
    "location_ref",
    # State
    "Overlay",
    "Actuator",
    # Surfaces
    "Embedding",
    "IntegrationBinding",
    "Surface",
    "HistoryEntry",
    # Tools & registry
    "ToolDeclaration",
    "ToolOverride",
    "TypeSpec",
    "Registry",
    "default_registry",
    "node_type",
    "register_tools",
    "tools_for",
    "CompatPolicy",
    "Migration",
    # Parsing
    "parse_node",
    "parse_edge",
    # Errors & warnings
    "VellaError",
    "UnregisteredTypeError",
    "ToolOverrideError",
    "SchemaMigrationError",
    "VellaWarning",
    "UnknownEdgeTypeWarning",
    # Utilities
    "DEFAULT_TENANT",
    "UTCDatetime",
    "utcnow",
    "uuid7",
]
