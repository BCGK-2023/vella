"""End-to-end failure/retry-half coverage through the REAL ``run()`` loop (M6).

The driver's *convergence* half is exercised by ``tests/test_reconciler.py`` (a
converging ``done()`` handler runs to idle). Its *failure/retry* half — capped
backoff requeue, attempt tracking, and retry-through-to-give-up — was previously
provable only by calling ``_backoff_or_giveup``/``step`` directly with a separately
advanced clock, because the OLD implementation ``await``ed ``clock.sleep`` on the
single worker's own path: under :class:`~vella.reconciler.ManualClock` that
deadlocks ``run()`` (nothing can ``advance()`` the clock while the sole worker
coroutine is parked in the backoff). After the M6 fix the backoff waits OFF the
worker on a clock-driven delayed re-enqueue, so the retry half is now drivable end
to end. These tests drive it:

* ``test_requeue_backoff_through_run`` — a handler that requeues-after once then
  converges: ``run()`` must NOT re-dispatch before the delay elapses, and MUST
  re-dispatch + converge once the clock advances past it.
* ``test_error_giveup_through_run`` — an always-raising handler with
  ``max_attempts=3`` and a backoff schedule: advancing the clock through each
  backoff deadline drives exactly three attempts, then dead-letters the key and
  emits exactly one ``observe_only`` ``reconcile_giveup`` — all THROUGH ``run()``.
* ``test_requeue_backoff_ordering_through_run`` — two keys with different ``after``
  re-dispatch in clock order under ``advance()`` (the end-to-end version of the
  direct ordering test).

ACCEPTANCE (the defect this catches): against the OLD blocking implementation
these HANG — ``run()`` parks the single worker in ``clock.sleep`` and the test's
``advance()`` can never run, so the ``asyncio.wait_for`` backstop trips. Against
the off-worker fix they pass.

Each test drives ``run()`` IN A TASK on a PRE-EXISTING event loop (never
``asyncio.run`` per cycle — its loop-close machinery would mask leaks) and bounds
everything with ``asyncio.wait_for``. No ``pytest-asyncio``; warning-clean under
the gate's ``filterwarnings = error::UserWarning``.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any
from uuid import UUID, uuid4

from vella.core import Actuator, Node, Registry as CoreRegistry, VellaModel, node_type
from vella.runtime import Runtime

from vella.reconciler import (
    Context,
    InMemoryDeadLetterStore,
    ManualClock,
    ReconcileResult,
    Reconciler,
    Registry,
)


# --- fixtures (mirror tests/test_reconciler.py) -----------------------------
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
    tenant_id: str = "t1",
    node_id: UUID | None = None,
    current: str = "off",
    desired: str = "on",
) -> "Node[Any, Any]":
    """A ``device`` node with Actuator state whose current diverges from desired."""
    return Node[DeviceData, Any](  # type: ignore[valid-type]
        id=node_id or uuid4(),
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


def _converged(state: Actuator[Any]) -> bool:
    """Return whether an actuator's current equals its desired (json-dump compare)."""
    assert state.desired is not None
    return bool(
        state.current.model_dump(mode="json")
        == state.desired.model_dump(mode="json")
    )


async def _await_backoff_parked(
    rec: Reconciler, clock: ManualClock, keys: set[Any]
) -> None:
    """Yield the loop until every key in ``keys`` is parked on its backoff timer.

    A backoff timer is its own task: it only registers its ``clock.sleep`` waiter
    once the loop SCHEDULES that task (one event-loop turn after the worker calls
    ``_schedule_requeue``). ``ManualClock.advance()`` resolves only ALREADY-parked
    waiters, so a test must wait until the timer is parked before advancing — else
    it advances past a not-yet-registered deadline and the timer never fires.

    "Parked" requires BOTH: the key is in ``rec._delayed`` (the timer task exists),
    AND the clock has at least ``len(keys)`` backoff waiters BEYOND the always-on
    resync ticker's single waiter (so each timer has reached its ``clock.sleep``).
    Internal access is the established test idiom in this package (cf.
    ``rec._workset._versions``, ``rt._store._index``).
    """
    while not (
        keys <= set(rec._delayed)
        # resync parks one waiter at run() start; each backoff timer adds one more.
        and len(clock._waiters) >= len(keys) + 1  # noqa: SLF001 - test seam
    ):
        await asyncio.sleep(0)


def _on_loop(coro_factory: Any, *, timeout: float = 5.0) -> None:
    """Run an async case on a fresh PRE-EXISTING loop under a warning-error filter.

    Mirrors ``test_run_teardown_closes_generator_and_cancels_tasks``: drives the
    case on a loop we own (NOT ``asyncio.run``, whose loop-close
    ``shutdown_asyncgens`` masks leaks) so the off-worker backoff timers and the
    watch/resync tasks are torn down by ``run()``'s own ``finally``, not by loop
    shutdown. Any warning is a hard failure (mirrors the gate).
    """
    loop = asyncio.new_event_loop()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loop.run_until_complete(
                asyncio.wait_for(coro_factory(loop), timeout=timeout)
            )
    finally:
        loop.close()


# --- 1: requeue-backoff through run() ---------------------------------------
def test_requeue_backoff_through_run() -> None:
    """A requeue-after handler converges only after the clock passes the delay.

    Drives ``run()`` in a task. The handler returns ``requeue(after=5.0)`` on its
    first dispatch (NOT converging) then converges on the next. Asserts: the entity
    does NOT converge while the clock is below the delay (advancing less than the
    delay must not re-dispatch); once the clock advances past the delay the key is
    re-dispatched, converges, and ``run()`` goes idle and returns.

    Against the OLD blocking backoff this HANGS: the worker parks in
    ``clock.sleep(5.0)`` so the test's ``advance()`` never runs.
    """
    _on_loop(_case_requeue_backoff_through_run)


async def _case_requeue_backoff_through_run(
    loop: asyncio.AbstractEventLoop,
) -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    registry = Registry()
    clock = ManualClock()

    node = _drifting_node(DeviceData)
    calls = {"n": 0}

    async def requeue_then_converge(ctx: Context) -> ReconcileResult:
        calls["n"] += 1
        if calls["n"] == 1:
            # First pass: ask to be re-examined after a delay (do NOT converge).
            return ReconcileResult.requeue(after=5.0)
        # Subsequent pass: converge current -> desired.
        got = await ctx.runtime.get(node.tenant_id, node.id)
        assert got is not None and isinstance(got.state, Actuator)
        await ctx.runtime.edit(
            node.tenant_id,
            node.id,
            expected_version=got.version,
            state=Actuator(current=got.state.desired, desired=got.state.desired),
        )
        return ReconcileResult.done()

    registry.register("device", requeue_then_converge)
    await rt.create(node)

    rec = Reconciler(
        rt, registry, clock, resync_interval=10_000.0, backoff=1.0, max_attempts=5
    )
    run_task: asyncio.Task[None] = asyncio.ensure_future(rec.run())

    # Let the worker fold + dispatch once: the handler requeues-after, scheduling an
    # off-worker delayed re-enqueue, and wait until that timer is PARKED on the clock
    # (else advance() would race a not-yet-registered deadline). run() must NOT return
    # (the key is mid-backoff).
    key = (node.tenant_id, node.id)
    await asyncio.wait_for(_await_backoff_parked(rec, clock, {key}), timeout=2.0)
    assert calls["n"] == 1  # dispatched exactly once so far
    assert key in rec._delayed  # parked on the off-worker backoff timer
    assert rec.is_idle() is False  # mid-backoff is NOT idle
    assert not run_task.done()  # run() did not early-return as converged

    # Advance LESS than the delay: the timer must NOT fire, no re-dispatch, still
    # exactly one handler call, still not converged, run() still parked.
    await clock.advance(4.0)
    assert calls["n"] == 1
    assert key in rec._delayed
    got = await rt.get(node.tenant_id, node.id)
    assert got is not None and isinstance(got.state, Actuator)
    assert not _converged(got.state)
    assert not run_task.done()

    # Advance PAST the delay (reaches 5.0): the timer fires, re-enqueues the key, the
    # worker re-dispatches, the handler converges, run() goes idle and returns.
    await clock.advance(1.0)
    await asyncio.wait_for(run_task, timeout=2.0)

    assert calls["n"] == 2  # re-dispatched exactly once after the backoff
    assert rec.is_idle() is True
    got = await rt.get(node.tenant_id, node.id)
    assert got is not None and isinstance(got.state, Actuator)
    assert _converged(got.state)


# --- 2: error -> give-up through run() with max_attempts > 1 ----------------
def test_error_giveup_through_run() -> None:
    """An always-raising handler retries through the backoff then gives up via run().

    ``max_attempts=3`` with ``backoff=1.0`` (schedule: 1.0, 2.0). Drives ``run()`` in
    a task; advancing the clock through each backoff deadline drives exactly three
    attempts, after which the key is dead-lettered AND exactly one ``observe_only``
    ``reconcile_giveup`` telemetry entry is emitted — all THROUGH ``run()``, never a
    direct ``_backoff_or_giveup`` call. (The give-up entry's ``.version`` is never
    asserted.)

    Against the OLD blocking backoff this HANGS at the first retry: the worker parks
    in ``clock.sleep`` and the test's ``advance()`` never runs.
    """
    _on_loop(_case_error_giveup_through_run)


async def _case_error_giveup_through_run(
    loop: asyncio.AbstractEventLoop,
) -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    registry = Registry()
    clock = ManualClock()
    deadletter = InMemoryDeadLetterStore()

    node = _drifting_node(DeviceData)
    calls = {"n": 0}

    async def always_fail(_ctx: Context) -> ReconcileResult:
        calls["n"] += 1
        raise RuntimeError("permanent failure")

    registry.register("device", always_fail)
    await rt.create(node)
    key = (node.tenant_id, node.id)

    rec = Reconciler(
        rt,
        registry,
        clock,
        deadletter_store=deadletter,
        resync_interval=10_000.0,
        backoff=1.0,
        backoff_cap=60.0,
        max_attempts=3,
    )
    run_task: asyncio.Task[None] = asyncio.ensure_future(rec.run())

    # Attempt 1 (the initial dispatch): the handler raises (attempts -> 1), schedules
    # a backoff timer at delay 1.0 (base * 2**0). Wait until that timer is parked on
    # the clock. run() is mid-backoff -> not idle, not returned, not yet given up.
    await asyncio.wait_for(_await_backoff_parked(rec, clock, {key}), timeout=2.0)
    assert calls["n"] == 1
    assert key in rec._delayed
    assert not run_task.done()
    assert await deadletter.get(*key) is None  # not yet given up

    # Advance to the first deadline (1.0): the timer re-enqueues, the worker
    # re-dispatches (attempt 2 -> attempts 2), the handler raises again, scheduling
    # the next backoff at delay 2.0 (base * 2**1). Wait until THAT timer is parked.
    await clock.advance(1.0)
    await asyncio.wait_for(_await_backoff_parked(rec, clock, {key}), timeout=2.0)
    assert calls["n"] == 2
    assert key in rec._delayed
    assert not run_task.done()
    assert await deadletter.get(*key) is None

    # Advance to the second deadline (2.0): re-dispatch (attempt 3 -> attempts 3)
    # reaches max_attempts -> give-up. No further timer; key dead-lettered; run() idle.
    await clock.advance(2.0)
    await asyncio.wait_for(run_task, timeout=2.0)

    assert calls["n"] == 3  # exactly max_attempts dispatches
    assert rec.is_idle() is True
    assert key not in rec._delayed

    record = await deadletter.get(*key)
    assert record is not None
    assert record.attempts == 3
    assert "permanent failure" in record.reason

    # Exactly ONE observe_only reconcile_giveup telemetry (its .version not asserted).
    full = await rt.history(node.tenant_id, node.id)
    giveups = [
        e
        for e in full
        if e.transition == "observe_only"
        and e.payload.get("event") == "reconcile_giveup"
    ]
    assert len(giveups) == 1
    assert giveups[0].payload["attempts"] == 3


# --- 3: backoff ordering through run() --------------------------------------
def test_requeue_backoff_ordering_through_run() -> None:
    """Two keys with different ``after`` re-dispatch in clock order through run().

    The end-to-end version of the direct ordering test: two distinct entities whose
    handlers requeue-after with different delays (the longer one created first) then
    converge. Driving ``run()`` and advancing the clock must re-dispatch the SHORTER
    delay first (the deterministic ``advance()`` ordering), converging it before the
    longer one, and only go idle once both have converged.

    Against the OLD blocking backoff this HANGS: the first requeue parks the single
    worker, so the second key is never even dispatched and ``advance()`` never runs.
    """
    _on_loop(_case_requeue_backoff_ordering_through_run)


def _two_kind_registry() -> tuple[type, type]:
    """Isolated core registry with two device kinds, each its own handler target.

    The worker dispatches by entity KIND and a :class:`Context` does not name the
    entity, so a handler must close over one entity. Two distinct kinds give two
    distinct handlers — one per entity — so two keys can be driven independently.
    """
    reg = CoreRegistry()

    @node_type("device_slow", registry=reg)
    class SlowData(VellaModel):
        power: str = "off"

    @node_type("device_fast", registry=reg)
    class FastData(VellaModel):
        power: str = "off"

    return SlowData, FastData


def _kinded_node(
    Data: type, kind: str, *, tenant_id: str = "t1", node_id: UUID
) -> "Node[Any, Any]":
    """A drifting node of an explicit ``kind`` (current != desired)."""
    return Node[Data, Any](  # type: ignore[valid-type]
        id=node_id,
        type=kind,
        name="dev",
        created_by=uuid4(),
        data=Data(power="off"),
        tenant_id=tenant_id,
        state=Actuator(current=Data(power="off"), desired=Data(power="on")),
    )


async def _case_requeue_backoff_ordering_through_run(
    loop: asyncio.AbstractEventLoop,
) -> None:
    rt = Runtime()
    SlowData, FastData = _two_kind_registry()
    registry = Registry()
    clock = ManualClock()

    # The SLOW key is created first (folded/dispatched first) but must converge LAST
    # because its backoff delay is longer.
    slow_node = _kinded_node(SlowData, "device_slow", node_id=uuid4())
    fast_node = _kinded_node(FastData, "device_fast", node_id=uuid4())
    slow_key = (slow_node.tenant_id, slow_node.id)
    fast_key = (fast_node.tenant_id, fast_node.id)
    requeued: set[Any] = set()
    converge_order: list[Any] = []

    def _make_handler(key: Any, after: float) -> Any:
        """One handler closing over its own entity: requeue-after once, then converge."""

        async def handler(ctx: Context) -> ReconcileResult:
            got = await ctx.runtime.get(*key)
            assert got is not None and isinstance(got.state, Actuator)
            if key not in requeued:
                requeued.add(key)
                return ReconcileResult.requeue(after=after)
            await ctx.runtime.edit(
                key[0],
                key[1],
                expected_version=got.version,
                state=Actuator(current=got.state.desired, desired=got.state.desired),
            )
            converge_order.append(key)
            return ReconcileResult.done()

        return handler

    registry.register("device_slow", _make_handler(slow_key, 5.0))
    registry.register("device_fast", _make_handler(fast_key, 1.0))
    await rt.create(slow_node)
    await rt.create(fast_node)

    rec = Reconciler(rt, registry, clock, resync_interval=10_000.0, backoff=1.0)
    run_task: asyncio.Task[None] = asyncio.ensure_future(rec.run())

    # Let both keys dispatch once and PARK their off-worker backoff timers on the
    # clock (single-flight: the worker dispatches one at a time, so both reach their
    # requeue-after over successive steps — but neither blocks the worker).
    await asyncio.wait_for(
        _await_backoff_parked(rec, clock, {slow_key, fast_key}), timeout=2.0
    )
    assert requeued == {slow_key, fast_key}
    assert set(rec._delayed) == {slow_key, fast_key}
    assert not run_task.done()

    # Advance to 1.0: the FAST timer fires first; its key re-dispatches and converges.
    await clock.advance(1.0)

    async def _fast_converged() -> None:
        while fast_key not in converge_order:
            await asyncio.sleep(0)

    await asyncio.wait_for(_fast_converged(), timeout=2.0)
    assert converge_order == [fast_key]  # shorter delay converged first
    assert slow_key in rec._delayed  # slow still parked on its backoff
    assert not run_task.done()

    # Advance to 5.0: the SLOW timer fires; its key re-dispatches, converges, idle.
    await clock.advance(4.0)
    await asyncio.wait_for(run_task, timeout=2.0)

    assert converge_order == [fast_key, slow_key]
    assert rec.is_idle() is True
    for key in (slow_key, fast_key):
        got = await rt.get(*key)
        assert got is not None and isinstance(got.state, Actuator)
        assert _converged(got.state)
