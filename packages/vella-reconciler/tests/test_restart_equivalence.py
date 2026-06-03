"""Restart-equivalence (M6, must-fix 5 / M-1).

The spec's restart-equivalence invariant, work-set scope: *a fresh* ``Reconciler``
*replaying* ``observe(None)`` *rebuilds the identical latest-per-entity work-set and
converges the same set of entities as a continuous run, given a converging handler.*
It is about the derived work-set + eventual convergence; it EXPLICITLY does NOT
preserve retry/attempt counters or dead-letter membership (those reset on restart;
re-attempting is safe/idempotent — persistence is the v0.2 seam).

Preconditions (stated, not assumed):
  (a) a deterministic CONVERGING handler (``current := desired`` via a fresh-get
      ``edit``);
  (b) BOTH the continuous run and the replay-from-cursor run are driven by the
      SAME ``ManualClock.advance()`` sequence (here: NONE — both early-return at
      idle, so the advance sequence is the empty sequence, identical by
      construction; a long resync interval guarantees no tick fires);
  (c) equality is asserted on the work-set KEY SET ``{(tenant_id, entity_id)}`` AND,
      per key, that final ``current == desired`` compared via
      ``model_dump(mode="json")`` — NEVER ``==`` (core's ``_vella_registry``
      ``PrivateAttr`` — ``base.py:72`` — breaks structural ``==``).

Structure: run a continuous Reconciler to convergence over a runtime; capture its
work-set key set + final per-key state. Build a FRESH Reconciler over the SAME
runtime log (replay ``observe(None)``), drive it with the same (empty) advance
sequence, run to convergence; assert identical key set + identical per-key final
state via ``model_dump(mode="json")``.

Finding (honest deviation from the plan's mutation rationale)
-------------------------------------------------------------
The plan/spec assert that core's ``_vella_registry`` ``PrivateAttr`` "breaks
structural ``==``", so swapping ``model_dump(mode="json")`` for ``==`` should make
this test fail. Verified against the committed core, the precise mechanism is
narrower: pydantic v2 DOES compare ``__pydantic_private__`` (the ``PrivateAttr``
values) inside ``==``. But for SAME-ROW reads the private attr MATCHES — both
entities hydrate against the same ``_vella_registry`` — so that term of the
comparison is equal and ``==`` holds. The only thing that makes freshly-CONSTRUCTED
entities ``==``-unequal is differing per-run wall-clock
``created_at``/``updated_at``. Since both runs here read the SAME converged rows
(run 2 is a safe no-op), raw ``==`` would also hold — so the plan's specific "``==``
fails on the PrivateAttr" mutation does not bite on this codebase. The test still
uses ``model_dump(mode="json")`` because it is the project's mandated
entity-comparison discipline and is the correct tool (serializable + scrubable); the
restart-equivalence invariant's real non-vacuity rests on the WORK-SET KEY SET
equality and per-key convergence, both asserted below.

No ``pytest-asyncio``; bounded ``asyncio.wait_for`` backstops.
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


# Per-run wall-clock keys core mints at construction/write time — not part of the
# converged WORK-SET state. Scrubbed (mirroring runtime's _determinism_helper) so
# the comparison measures convergence, not timestamp noise.
_VOLATILE_KEYS = frozenset(
    {"created_at", "updated_at", "last_updated_at", "last_desired_at"}
)


def _scrub(obj: Any) -> Any:
    """Recursively drop per-run wall-clock keys from a JSON-mode structure."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _make_registry(seeded: list[tuple[str, UUID]]) -> Registry:
    """A registry with one generic ``device`` converging handler over ``seeded``.

    The handler scans the seeded keys for the one still drifting and writes
    ``current := desired`` on it from a FRESH ``get`` (the correct optimistic-
    version pattern). A single registered handler drives the whole backlog.
    """
    registry = Registry()

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
            await ctx.runtime.edit(
                tenant_id,
                entity_id,
                expected_version=got.version,
                state=Actuator(current=desired, desired=desired),
            )
            return ReconcileResult.done()
        return ReconcileResult.done()

    registry.register("device", converge)
    return registry


async def _final_state(rt: Runtime, keys: list[tuple[str, UUID]]) -> dict[
    tuple[str, UUID], dict[str, Any]
]:
    """Return each key's final FULL entity as a scrubbed json-mode dump.

    The WHOLE :class:`~vella.core.Node` is dumped (volatile wall-clock timestamps
    scrubbed) rather than just the actuator submodel, so the assertion states the
    STRONGEST equivalence: the entire reconstructed entity is identical across the
    two runs, not merely its converged ``current`` field. ``model_dump(mode="json")``
    is the project's mandated comparison (core forbids structural ``==`` on entities)
    and is the right tool here regardless: it is serializable and scrubable, so two
    constructions that differ only in per-run wall-clock timestamps still compare
    equal once scrubbed. (See the module-level finding note: for entities READ BACK
    from the same runtime row, pydantic ``==`` happens to also hold — pydantic v2
    DOES compare the ``_vella_registry`` ``PrivateAttr``, but both reads hydrate to
    the same registry so that term matches — so the json-dump's load-bearing value
    here is timestamp-scrub + serializability, not a PrivateAttr ``==`` break.)
    """
    out: dict[tuple[str, UUID], dict[str, Any]] = {}
    for tenant_id, entity_id in keys:
        got = await rt.get(tenant_id, entity_id)
        assert got is not None and isinstance(got.state, Actuator)
        out[(tenant_id, entity_id)] = _scrub(got.model_dump(mode="json"))
    return out


def test_replay_rebuilds_workset_and_converges_identically() -> None:
    """A fresh Reconciler over the same log rebuilds the work-set + converges same."""
    _drive(_case_restart_equivalence())


async def _case_restart_equivalence() -> None:
    rt = Runtime()
    DeviceData = _core_registry()

    # An arbitrary backlog: drifting devices across two tenants, distinct ids.
    specs = [
        ("t-alpha", "off", "on"),
        ("t-alpha", "0", "100"),
        ("t-beta", "cold", "hot"),
        ("t-beta", "idle", "active"),
        ("t-alpha", "red", "green"),
    ]
    seeded: list[tuple[str, UUID]] = []
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
    expected_key_set = set(seeded)

    # --- Run 1: the CONTINUOUS run. Long resync interval; clock never advanced
    # (advance sequence = empty). Early-returns at idle. ---
    registry1 = _make_registry(seeded)
    rec1 = Reconciler(rt, registry1, ManualClock(), resync_interval=10_000.0)
    await rec1.run(max_steps=len(specs) * 4)
    assert rec1.is_idle() is True

    # Work-set key set the continuous run derived (the fold's recorded membership).
    keyset1 = {k for k in rec1._workset.keys()}  # noqa: SLF001 - asserting the derived set
    assert keyset1 == expected_key_set
    final1 = await _final_state(rt, seeded)

    # Convergence holds after run 1 (current == desired, actuator-submodel json
    # compare — never == on the entity).
    for tenant_id, entity_id in seeded:
        got = await rt.get(tenant_id, entity_id)
        assert got is not None and isinstance(got.state, Actuator)
        assert got.state.desired is not None
        assert got.state.current.model_dump(mode="json") == got.state.desired.model_dump(
            mode="json"
        )

    # --- Run 2: a FRESH Reconciler replaying observe(None) over the SAME runtime
    # log. No cursor store -> observes from the start (replay-from-zero). Same
    # (empty) advance sequence. Re-attempting an already-converged key is a safe
    # no-op (the worker's fresh-get drift recheck clears it). ---
    registry2 = _make_registry(seeded)
    rec2 = Reconciler(rt, registry2, ManualClock(), resync_interval=10_000.0)
    await rec2.run(max_steps=len(specs) * 4)
    assert rec2.is_idle() is True

    keyset2 = {k for k in rec2._workset.keys()}  # noqa: SLF001 - asserting the derived set
    final2 = await _final_state(rt, seeded)

    # (c) IDENTICAL work-set KEY SET across the two runs.
    assert keyset2 == keyset1 == expected_key_set

    # (c) IDENTICAL per-key final state across the two runs, compared via
    # model_dump(mode="json") on the FULL entity (volatile timestamps scrubbed) —
    # the project's mandated entity comparison (core forbids structural == on
    # entities). final2 / final1 are dicts of scrubbed json dumps, so this is a safe
    # structural comparison. (Finding: see the module note — for store-reconstructed
    # converged rows raw == would ALSO hold here, so the plan's "== fails on the
    # _vella_registry PrivateAttr" mutation rationale does not bite on this codebase;
    # the json dump's real value is timestamp-scrub + serializability.)
    assert final2 == final1
    for tenant_id, entity_id in seeded:
        got = await rt.get(tenant_id, entity_id)
        assert got is not None and isinstance(got.state, Actuator)
        assert got.state.desired is not None
        # Final current equals desired for the replayed run too (json-dump compare).
        assert got.state.current.model_dump(mode="json") == got.state.desired.model_dump(
            mode="json"
        )
