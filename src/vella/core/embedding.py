"""
Embedding — a vector with enough metadata to survive re-embedding cycles.

A bare ``list[float]`` loses track of which model produced a vector, its
dimensionality, and what content was embedded — so the first time you swap
embedding models you cannot tell new vectors from old. (Owned by core today as
part of the V6 model; the natural future home is the vector-store facet.)
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import UTCDatetime, VellaModel, utcnow


class Embedding(VellaModel):
    vector: list[float]
    model: str                       # "text-embedding-3-large", "voyage-3", ...
    dimensions: int
    normalized: bool = True
    generated_at: UTCDatetime = Field(default_factory=utcnow)
    generated_from: Literal["data", "data_and_state", "custom"] = "data"


__all__ = ["Embedding"]
