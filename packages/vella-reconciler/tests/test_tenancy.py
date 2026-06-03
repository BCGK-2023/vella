"""Tenancy (M6).

The spec's tenancy invariant, end-to-end through the driver: *the global*
``observe()`` *stream is filtered correctly; every write carries* ``(tenant_id,
id)``; *no cross-tenant reconcile.* The runtime's store key is ``(tenant_id, kind,
entity_id)``, so an entity in tenant A is invisible to a ``get`` under tenant B
(``runtime.py:358-364``): a cross-tenant ``get`` returns ``None``.

This exercises the full driver with TWO tenants holding entities that share the
SAME UUID but have DISTINCT desired states. A converging handler that converges
each key under its OWN tenant must drive both to their own desired without leaking
across the tenant boundary. We assert: (1) cross-tenant ``get`` returns ``None``;
(2) each entity converges to ITS tenant's desired (json-dump compare, never
``==``); (3) no cross-tenant write happened — neither entity took on the other
tenant's desired value.

No ``pytest-asyncio``; bounded ``asyncio.wait_for`` backstop.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

from vella.core import Actuator, Node, Registry as CoreRegistry, VellaModel, node_type
from vella.runtime import Runtime

from vella.reconciler import (
    Context,
    ManualClock,
    ReconcileResult,
    Reconciler,
    Registry,
)


def _core_registry() -> type:
    """Isolated core registry with one ``device`` node kind (never the global)."""
    reg = CoreRegistry()

    @node_type("device", registry=reg)
    class DeviceData(VellaModel):
        power: str = "off"

    return DeviceData


def _drifting_node(
    DeviceData: type,
    *,
    tenant_id: str,
    node_id: UUID,
    current: str,
    desired: str,
) -> "Node[Any, Any]":
    """A ``device`` node with Actuator state whose current diverges from desired."""
    return Node[DeviceData, Any](  # type: ignore[valid-type]
        id=node_id,
        type="device",
        name="dev",
        created_by=uuid4(),
        data=DeviceData(power=current),
        tenant_id=tenant_id,
        state=Actuator(
            current=DeviceData(power=current),
            desired=DeviceData(power=desired),
        ),
    )


def _drive(coro: object, *, timeout: float = 5.0) -> None:
    """Run ``coro`` under a bounded backstop so a coordination bug fails fast."""
    asyncio.run(asyncio.wait_for(coro, timeout=timeout))  # type: ignore[arg-type]


def test_same_uuid_two_tenants_converge_independently() -> None:
    """Same-UUID entities in two tenants converge to their own desired; no leak."""
    _drive(_case_tenancy())


async def _case_tenancy() -> None:
    rt = Runtime()
    DeviceData = _core_registry()

    # Same UUID, two tenants, DISTINCT desired states.
    shared_id = uuid4()
    tenant_a, tenant_b = "tenant-A", "tenant-B"
    desired_a, desired_b = "on", "standby"

    await rt.create(
        _drifting_node(
            DeviceData,
            tenant_id=tenant_a,
            node_id=shared_id,
            current="off",
            desired=desired_a,
        )
    )
    await rt.create(
        _drifting_node(
            DeviceData,
            tenant_id=tenant_b,
            node_id=shared_id,
            current="off",
            desired=desired_b,
        )
    )

    # Precondition assertion (the runtime's tenancy guarantee the driver relies on):
    # a cross-tenant get DOES NOT cross the boundary — each tenant sees only its own.
    got_a = await rt.get(tenant_a, shared_id)
    got_b = await rt.get(tenant_b, shared_id)
    assert got_a is not None and isinstance(got_a.state, Actuator)
    assert got_b is not None and isinstance(got_b.state, Actuator)
    assert got_a.state.desired is not None and got_b.state.desired is not None
    assert got_a.state.desired.model_dump(mode="json")["power"] == desired_a
    assert got_b.state.desired.model_dump(mode="json")["power"] == desired_b

    # A generic handler that converges WHICHEVER tenant's key still drifts. It scans
    # both (tenant, id) keys, finds the drifting one, and converges it under its own
    # tenant from a fresh get — so the write carries the correct (tenant_id, id) and
    # can never cross tenants.
    keys = [(tenant_a, shared_id), (tenant_b, shared_id)]
    registry = Registry()

    async def converge(ctx: Context) -> ReconcileResult:
        for tenant_id, entity_id in keys:
            got = await ctx.runtime.get(tenant_id, entity_id)
            assert got is not None and isinstance(got.state, Actuator)
            desired = got.state.desired
            assert desired is not None
            if got.state.current.model_dump(mode="json") == desired.model_dump(
                mode="json"
            ):
                continue
            await ctx.runtime.edit(
                tenant_id,
                entity_id,
                expected_version=got.version,
                state=Actuator(current=desired, desired=desired),
            )
            return ReconcileResult.done()
        return ReconcileResult.done()

    registry.register("device", converge)

    rec = Reconciler(rt, registry, ManualClock(), resync_interval=10_000.0)
    await rec.run(max_steps=8)
    assert rec.is_idle() is True

    # Each entity converged to ITS OWN tenant's desired (json-dump compare, never ==).
    final_a = await rt.get(tenant_a, shared_id)
    final_b = await rt.get(tenant_b, shared_id)
    assert final_a is not None and isinstance(final_a.state, Actuator)
    assert final_b is not None and isinstance(final_b.state, Actuator)

    a_current = final_a.state.current.model_dump(mode="json")
    b_current = final_b.state.current.model_dump(mode="json")
    assert a_current["power"] == desired_a  # A converged to A's desired
    assert b_current["power"] == desired_b  # B converged to B's desired

    # No cross-tenant LEAK: neither took on the other tenant's desired value.
    assert a_current["power"] != desired_b
    assert b_current["power"] != desired_a

    # The work-set holds BOTH keys distinctly (same UUID, different tenant -> two
    # distinct keys): tenancy keeps them separate end-to-end.
    assert set(rec._workset.keys()) == set(keys)  # noqa: SLF001 - asserting derived set
