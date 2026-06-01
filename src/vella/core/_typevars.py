"""
Type variables for the polymorphic Node/Edge envelopes.

Imported from ``typing_extensions`` so PEP 696 ``default=`` works uniformly
across the supported Python range (it is only in stdlib ``typing`` from 3.13).
"""

from __future__ import annotations

from pydantic import BaseModel
from typing_extensions import TypeVar

TData = TypeVar("TData", bound=BaseModel)

# TState defaults to BaseModel so ``Node[EmailData]`` works without naming the
# state slot. The trap: if you rely on the default while *using* state,
# ``state.value`` types as ``Optional[BaseModel]`` (type erasure). Always
# specify TState explicitly when a node has state: ``Node[LightData, LightState]``.
TState = TypeVar("TState", bound=BaseModel, default=BaseModel)

TEdgeData = TypeVar("TEdgeData", bound=BaseModel)
TEdgeState = TypeVar("TEdgeState", bound=BaseModel, default=BaseModel)

__all__ = ["TData", "TState", "TEdgeData", "TEdgeState"]
