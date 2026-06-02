"""Exception hierarchy.

A single root, ``VellaError``, so any consumer can ``except VellaError`` to
catch everything the SDK raises. Core defines only the errors core itself
raises; higher layers (runtime, integrations) define their own — e.g.
``VersionConflictError``, ``TenantViolationError`` — by subclassing this same
root, so the whole stack shares one catchable tree.

Field-level shape validation is *not* in this tree: let Pydantic's own
``ValidationError`` propagate. This tree is for semantic/domain failures.

Errors carry structured attributes (not just a message string) so callers and
self-healing flows can branch programmatically instead of parsing English.
"""

from __future__ import annotations


class VellaError(Exception):
    """Root of every error raised by the Vella SDK."""


class VellaWarning(UserWarning):
    """Root of every warning emitted by the Vella SDK (filterable as a group)."""


class UnknownEdgeTypeWarning(VellaWarning):
    """Emitted when an Edge ``type`` is not a canonical ``EdgeTypes`` constant."""


class UnregisteredTypeError(VellaError):
    """A node/edge type was used but is not registered in the active registry."""

    def __init__(self, type_name: str | None, available: list[str]) -> None:
        """Record the offending type name and the sorted available types."""
        self.type_name = type_name
        self.available = sorted(available)
        super().__init__(
            f"{type_name!r} is not a registered type. "
            f"Decorate its data class with @node_type('{type_name}', ...) or "
            f"construct with type=... explicitly. Available: {self.available}"
        )


class ToolOverrideError(VellaError):
    """A ToolOverride cannot be resolved against the registry for its type."""

    def __init__(self, message: str, *, tool_name: str, type_name: str) -> None:
        """Record the offending tool name and type name alongside ``message``."""
        self.tool_name = tool_name
        self.type_name = type_name
        super().__init__(message)


class SchemaMigrationError(VellaError):
    """No migration path exists between two schema versions, or one failed."""

    def __init__(
        self, message: str, *, type_name: str, from_version: int, to_version: int
    ) -> None:
        """Record the type name and the from/to schema versions for the failure."""
        self.type_name = type_name
        self.from_version = from_version
        self.to_version = to_version
        super().__init__(message)


__all__ = [
    "VellaError",
    "UnregisteredTypeError",
    "ToolOverrideError",
    "SchemaMigrationError",
    "VellaWarning",
    "UnknownEdgeTypeWarning",
]
