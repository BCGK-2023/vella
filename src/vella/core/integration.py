"""IntegrationBinding — links a node or edge to an upstream system.

A node/edge may carry *multiple* bindings (polysource): a Page synthesized from
WordPress + GA4 + Search Console + Ads + HubSpot has five, each declaring its
``role`` and which surfaces it ``contributes_to``.

Idempotency: ``(tenant_id, plugin, external_id)`` is the canonical upsert key.
The runtime enforces uniqueness on it and does lookup-or-create on ingestion, so
re-ingesting the same upstream resource updates the existing node rather than
minting a duplicate. (Because a node can have many bindings, idempotency is
binding-level — node ids are random UUIDv7, never derived from a natural key.)

Credentials NEVER live on the node; ``config_ref`` is an opaque pointer into a
separate secrets store.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from .base import VellaModel

Surface = Literal["data", "state", "embedding"]


def _default_surfaces() -> list[Surface]:
    return ["data", "state"]


class IntegrationBinding(VellaModel):
    """Links a node or edge to an upstream system (credentials live elsewhere)."""

    plugin: str                      # "philips_hue", "google_calendar", "wordpress", ...
    external_id: str
    config_ref: Optional[str] = None
    role: Literal["primary", "secondary", "observer"] = "primary"
    contributes_to: list[Surface] = Field(default_factory=_default_surfaces)


__all__ = ["Surface", "IntegrationBinding"]
