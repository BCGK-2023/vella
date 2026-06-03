"""Shared test fixtures: an isolated core registry + node/edge builders.

Mirrors the reconciler tests' idiom (``test_tenancy.py``): build a fresh
``Runtime()`` (default in-memory store via the public API — never importing
``vella.runtime._inmemory``), register node kinds into an ISOLATED ``CoreRegistry``
(never the global ``default_registry``), and drive async coroutines via
``asyncio.run`` + a bounded ``asyncio.wait_for`` (no pytest-asyncio).

All edges use canonical ``EdgeTypes`` constants — a non-canonical type trips
``UnknownEdgeTypeWarning`` (a ``UserWarning``), which the package gate's
``filterwarnings = ["error::UserWarning"]`` would turn into a hard error.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

from vella.core import Node, Registry as CoreRegistry, VellaModel, node_type
from vella.runtime import Runtime


def thing_registry() -> type:
    """Isolated core registry with one ``thing`` node kind (never the global)."""
    reg = CoreRegistry()

    @node_type("thing", registry=reg)
    class ThingData(VellaModel):
        label: str = "x"

    return ThingData


def make_node(
    thing_data: type,
    *,
    tenant_id: str,
    node_id: UUID,
    label: str = "x",
) -> "Node[Any, Any]":
    """A minimal ``thing`` node with the given id/tenant/label."""
    return Node[thing_data, Any](  # type: ignore[valid-type]
        id=node_id,
        type="thing",
        name="n",
        created_by=uuid4(),
        data=thing_data(label=label),
        tenant_id=tenant_id,
    )


def drive(coro: Any, *, timeout: float = 5.0) -> Any:
    """Run ``coro`` under a bounded backstop so a coordination bug fails fast."""
    return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
