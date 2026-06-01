"""
HistoryEntry — one entry in a node's or edge's append-only history log.

History is NOT a field on Node/Edge: it is written via ``append_history`` and
read via ``get_history`` at the integration-API layer, keeping node reads cheap
regardless of how many writes have accumulated. In stream-table terms the
history log is the source-of-truth *log* and a node's current state is the
materialized *table*. This type just gives entries a shape.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .base import UTCDatetime, VellaModel, utcnow


class HistoryEntry(VellaModel):
    timestamp: UTCDatetime = Field(default_factory=utcnow)
    source: str                      # who/what caused the change
    change: dict[str, Any] = Field(default_factory=dict)
    note: Optional[str] = None


__all__ = ["HistoryEntry"]
