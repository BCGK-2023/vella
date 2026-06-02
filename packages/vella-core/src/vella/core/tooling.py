"""Tool declarations, the type registry, and the ``@node_type`` decorator.

The registry is the central contract object of the SDK: it maps a type name to
its data class, state class, current schema version, compatibility policy,
canonical tools, and schema migrations. It is a real ``Registry`` *class* with a
module-level ``default_registry`` instance (not a bare global) so that:

  * tests get isolation by constructing a fresh ``Registry()``;
  * runtime type discovery (e.g. Home Assistant) is lock-guarded and safe;
  * the ``@node_type`` decorator stays ergonomic by targeting the default.

Lookups (``tools_for``, ``parse_node``, ...) accept an optional ``registry=``
that defaults to ``default_registry``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional

from pydantic import Field

from .base import VellaModel
from .errors import ToolOverrideError, VellaError

CompatPolicy = Literal[
    "FULL", "BACKWARD", "FORWARD", "NONE",
    "FULL_TRANSITIVE", "BACKWARD_TRANSITIVE", "FORWARD_TRANSITIVE",
]
"""Per-type schema-evolution compatibility policy, enforced by the CI tripwire.

Confluent semantics: ``FULL`` = both directions; ``BACKWARD`` = a new reader
reads old data; ``FORWARD`` = an old reader reads new data; ``NONE`` =
schema-on-read. The ``*_TRANSITIVE`` variants extend the check across the whole
version history rather than just the adjacent version.
"""

Migration = Callable[[dict[str, Any]], dict[str, Any]]
"""Upgrades a raw ``data`` dict from one schema version to the next."""


class ToolDeclaration(VellaModel):
    """A capability exposed by a node. ``parameters``/``returns`` are JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    returns: Optional[dict[str, Any]] = None


class ToolOverride(VellaModel):
    """A thin patch applied to a registry-defined tool for a specific node/edge."""

    tool_name: str
    description_override: Optional[str] = None
    parameter_overrides: dict[str, Any] = Field(default_factory=dict)
    returns_override: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class TypeSpec:
    """Everything the registry knows about one node/edge type."""

    name: str
    data_cls: type
    state_cls: Optional[type] = None
    version: int = 1                         # current ("reader") schema version
    compat: CompatPolicy = "BACKWARD"
    tools: tuple[ToolDeclaration, ...] = ()
    migrations: Mapping[int, Migration] = field(default_factory=dict[int, Migration])


class Registry:
    """A mutable, thread-safe registry of node/edge type specifications."""

    def __init__(self) -> None:
        """Construct an empty, isolated registry with its own lock."""
        self._specs: dict[str, TypeSpec] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        *,
        data_cls: type,
        state_cls: Optional[type] = None,
        version: int = 1,
        compat: CompatPolicy = "BACKWARD",
        tools: tuple[ToolDeclaration, ...] = (),
        migrations: Optional[Mapping[int, Migration]] = None,
    ) -> None:
        """Bind a data class under ``name`` with its full type specification.

        Records the state class, current schema version, compatibility policy,
        canonical tools, and migrations as a ``TypeSpec``, replacing any prior
        registration for the same name.

        Args:
            name: The type name to register under.
            data_cls: The data class for this type.
            state_cls: Optional state class.
            version: Current ("reader") schema version.
            compat: Per-type compatibility policy.
            tools: Canonical tools for the type.
            migrations: Version-keyed migration callables.
        """
        with self._lock:
            self._specs[name] = TypeSpec(
                name=name,
                data_cls=data_cls,
                state_cls=state_cls,
                version=version,
                compat=compat,
                tools=tuple(tools),
                migrations=dict(migrations or {}),
            )

    def register_tools(self, name: str, tools: list[ToolDeclaration]) -> None:
        """Imperatively (re)set tools for a type, preserving its other spec fields."""
        with self._lock:
            existing = self._specs.get(name)
            if existing is None:
                self._specs[name] = TypeSpec(name=name, data_cls=object, tools=tuple(tools))
            else:
                self._specs[name] = TypeSpec(
                    name=name,
                    data_cls=existing.data_cls,
                    state_cls=existing.state_cls,
                    version=existing.version,
                    compat=existing.compat,
                    tools=tuple(tools),
                    migrations=existing.migrations,
                )

    def get(self, name: str) -> Optional[TypeSpec]:
        """Return the ``TypeSpec`` registered under ``name``, or ``None``."""
        return self._specs.get(name)

    def tools_for(self, name: str) -> list[ToolDeclaration]:
        """Return the canonical tools registered for ``name`` (empty if none)."""
        spec = self._specs.get(name)
        return list(spec.tools) if spec else []

    def names(self) -> list[str]:
        """Return the registered type names, sorted."""
        return sorted(self._specs)

    def clear(self) -> None:
        """Empty the registry (chiefly for test isolation)."""
        with self._lock:
            self._specs.clear()


default_registry = Registry()
"""The process-wide registry used by ``@node_type`` and by lookups when none is passed.

Tests and embedders that need isolation construct their own ``Registry()`` and
pass it explicitly instead of touching this shared default.
"""


def node_type(
    name: str,
    *,
    state: Optional[type] = None,
    version: int = 1,
    compat: CompatPolicy = "BACKWARD",
    tools: Optional[list[ToolDeclaration]] = None,
    migrations: Optional[Mapping[int, Migration]] = None,
    registry: Optional[Registry] = None,
) -> Callable[[type], type]:
    """Register a data class as a node/edge type via decorator.

    Binds (next to the data shape) its state class, current schema version,
    compatibility policy, canonical tools, and migrations. Enforces that the
    data class is frozen.
    """

    def decorator(cls: type) -> type:
        config = getattr(cls, "model_config", {})
        if not config.get("frozen"):
            raise VellaError(
                f"@node_type({name!r}) requires a frozen data class "
                f"({cls.__name__} is not). Subclass FlexibleData / VellaModel, "
                f"or set model_config = ConfigDict(frozen=True)."
            )
        (registry or default_registry).register(
            name,
            data_cls=cls,
            state_cls=state,
            version=version,
            compat=compat,
            tools=tuple(tools or ()),
            migrations=migrations,
        )
        cls.__vella_type__ = name  # type: ignore[attr-defined]
        return cls

    return decorator


def register_tools(
    name: str, tools: list[ToolDeclaration], *, registry: Optional[Registry] = None
) -> None:
    """Imperatively register tools for a type (for runtime-computed variants)."""
    (registry or default_registry).register_tools(name, tools)


def tools_for(name: str, *, registry: Optional[Registry] = None) -> list[ToolDeclaration]:
    """Canonical tools registered for a type; empty list if none."""
    return (registry or default_registry).tools_for(name)


def registry_from_context(context: Any) -> Optional[Registry]:
    """Extract a ``Registry`` passed via ``model_validate(context={'registry': ...})``."""
    get = getattr(context, "get", None)
    if get is None:
        return None
    reg = get("registry", None)
    return reg if isinstance(reg, Registry) else None


def validate_tool_overrides(
    type_name: str, overrides: list[ToolOverride], *, registry: Optional[Registry] = None
) -> None:
    """Reject tool overrides that cannot compose against the registry.

    Rejects an unknown tool name, or a parameter patch that isn't in the base
    tool. Used by Node and Edge model validators at construction time.
    """
    if not overrides:
        return
    base_tools = {t.name: t for t in tools_for(type_name, registry=registry)}
    for override in overrides:
        base = base_tools.get(override.tool_name)
        if base is None:
            raise ToolOverrideError(
                f"Tool override targets {override.tool_name!r} but no such tool is "
                f"registered for type {type_name!r}. Available: {sorted(base_tools)}",
                tool_name=override.tool_name,
                type_name=type_name,
            )
        base_props = base.parameters.get("properties", {})
        for param_name in override.parameter_overrides:
            if param_name not in base_props:
                raise ToolOverrideError(
                    f"Override for tool {override.tool_name!r} patches parameter "
                    f"{param_name!r}, not in the base tool. Available: {sorted(base_props)}",
                    tool_name=override.tool_name,
                    type_name=type_name,
                )


__all__ = [
    "CompatPolicy",
    "Migration",
    "ToolDeclaration",
    "ToolOverride",
    "TypeSpec",
    "Registry",
    "default_registry",
    "node_type",
    "register_tools",
    "tools_for",
    "validate_tool_overrides",
    "registry_from_context",
]
