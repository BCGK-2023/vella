"""Convergence Definition-of-Done (M6).

The spec's first behavioral invariant: *given a drifting actuator and a handler
that sets* ``current := desired``, *the loop reaches* ``current == desired``,
*goes idle (no busy-spin), deterministically under* ``ManualClock`` *+ in-memory*
``Runtime``. This file proves it at the DoD level — from an ARBITRARY backlog of
many drifting actuators (across kinds and tenants), a single ``run()`` converges
EVERY entity, the loop's race-free idle predicate holds, and ``run()`` returns in
a bounded number of worker steps (the early-return-at-idle contract, must-fix 6).

All async is driven by ``asyncio.run`` + a bounded ``asyncio.wait_for`` backstop
(no ``pytest-asyncio``): a coordination bug surfaces as a TIMEOUT, never a hang.
Entity state is compared via ``model_dump(mode="json")``, NEVER ``==`` (core's
``_vella_registry`` ``PrivateAttr`` breaks structural ``==``). Fixtures use the
REAL in-memory :class:`~vella.runtime.Runtime` and real core
:class:`~vella.core.Actuator` / :class:`~vella.core.Node` types, so convergence is
exercised against real edge semantics.
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


# --- fixtures: real Runtime + real Actuator nodes ---------------------------
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


def _converged(state: Actuator[Any]) -> bool:
    """Return whether an actuator's current equals its desired (json-dump compare)."""
    assert state.desired is not None
    return bool(
        state.current.model_dump(mode="json")
        == state.desired.model_dump(mode="json")
    )


# --- convergence DoD: arbitrary backlog -> all converge, idle, bounded ------
def test_arbitrary_backlog_converges_goes_idle_bounded() -> None:
    """An arbitrary backlog of drifting actuators all converge; loop goes idle."""
    _drive(_case_backlog_converges())


async def _case_backlog_converges() -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    registry = Registry()

    # The convergence handler is generic over the key. The Context exposes the
    # runtime but not the in-flight key (v0.1 contract), so the worker's drift gate
    # has already confirmed exactly one drifting device is being dispatched. We
    # converge by scanning the known seeded keys for the one that still drifts and
    # writing current := desired on it. This keeps a SINGLE registered handler
    # driving the whole heterogeneous backlog to convergence.
    seeded: list[tuple[str, UUID]] = []

    async def converge(ctx: Context) -> ReconcileResult:
        for tenant_id, entity_id in seeded:
            got = await ctx.runtime.get(tenant_id, entity_id)
            if got is None or not isinstance(got.state, Actuator):
                continue
            desired = got.state.desired
            if desired is None:
                continue
            if got.state.current.model_dump(mode="json") == desired.model_dump(
                mode="json"
            ):
                continue
            # This key still drifts: converge it (fresh version from got).
            await ctx.runtime.edit(
                tenant_id,
                entity_id,
                expected_version=got.version,
                state=Actuator(current=desired, desired=desired),
            )
            return ReconcileResult.done()
        # No drifting key found (already converged): nothing to do.
        return ReconcileResult.done()

    registry.register("device", converge)

    # An arbitrary backlog: many drifting devices across two tenants, distinct ids,
    # distinct desired values. Each created entity appends a log entry the fold
    # picks up — so this is genuinely a backlog the watch task drains before the
    # live edge.
    specs = [
        ("t-alpha", "off", "on"),
        ("t-alpha", "low", "high"),
        ("t-alpha", "0", "100"),
        ("t-beta", "off", "on"),
        ("t-beta", "idle", "active"),
        ("t-beta", "cold", "hot"),
        ("t-alpha", "red", "green"),
    ]
    for tenant_id, current, desired in specs:
        node_id = uuid4()
        seeded.append((tenant_id, node_id))
        await rt.create(
            _drifting_node(
                DeviceData,
                tenant_id=tenant_id,
                node_id=node_id,
                current=current,
                desired=desired,
            )
        )

    # A long resync interval that the clock NEVER advances: if run() did not
    # early-return at idle, the wait_for backstop would trip. So a successful
    # return PROVES the early-return-at-idle contract (no busy-spin, no waiting for
    # a resync tick).
    rec = Reconciler(rt, registry, ManualClock(), resync_interval=10_000.0)

    # Bounded worker steps: there are len(specs) drifting keys; each needs exactly
    # one successful reconcile. A generous bound proves termination without relying
    # on the exact count — but the loop must STILL early-return at idle well within
    # it (no busy-spin would blow the wait_for first).
    await rec.run(max_steps=len(specs) * 4)

    # 1) Idle: the race-free predicate holds (queue empty, nothing in-flight, caught
    #    up to the live edge). NOT inferred from loop termination.
    assert rec.is_idle() is True

    # 2) Convergence: EVERY seeded entity reached current == desired, per tenant.
    for tenant_id, entity_id in seeded:
        got = await rt.get(tenant_id, entity_id)
        assert got is not None and isinstance(got.state, Actuator)
        assert _converged(got.state), (tenant_id, entity_id)


# --- convergence DoD: run() with no max_steps returns at idle ----------------
def test_unbounded_run_returns_at_idle() -> None:
    """``run()`` with no ``max_steps`` returns the moment the loop is idle."""
    _drive(_case_unbounded_run())


async def _case_unbounded_run() -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    registry = Registry()

    node = _drifting_node(
        DeviceData, tenant_id="t1", node_id=uuid4(), current="off", desired="on"
    )

    async def converge(ctx: Context) -> ReconcileResult:
        got = await ctx.runtime.get(node.tenant_id, node.id)
        assert got is not None and isinstance(got.state, Actuator)
        await ctx.runtime.edit(
            node.tenant_id,
            node.id,
            expected_version=got.version,
            state=Actuator(current=got.state.desired, desired=got.state.desired),
        )
        return ReconcileResult.done()

    registry.register("device", converge)
    await rt.create(node)

    # No max_steps: run() must early-return at idle (the clock is never advanced, so
    # waiting for a resync tick would hang the wait_for backstop).
    rec = Reconciler(rt, registry, ManualClock(), resync_interval=10_000.0)
    await rec.run()

    assert rec.is_idle() is True
    got = await rt.get(node.tenant_id, node.id)
    assert got is not None and isinstance(got.state, Actuator)
    assert _converged(got.state)
