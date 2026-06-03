"""Driver coordination-contract invariants (M5).

The :class:`~vella.reconciler.Reconciler` is the three-task watch/worker/resync
controller. These tests pin the precise, testable coordination contract (must-fix
2a-2e + the worker dispatch semantics of must-fix 6):

* ``test_step_nonblocking`` — ``step()`` on an empty queue returns ``IDLE`` and does
  not block (2c).
* ``test_idle_predicate`` — idle is False while the backlog-drained Event is unset
  even with an empty queue; True once caught-up + queue empty + nothing in-flight
  (2b).
* ``test_run_early_return_on_idle`` — ``run()`` returns before the first resync tick
  once idle (must-fix 6, early-return).
* ``test_run_to_convergence_emits_no_warnings`` — a full run-to-convergence emits
  ZERO warnings; teardown cancels tasks and closes the observe generator (2d).
* ``test_resync_skips_deadlettered`` — a dead-lettered key is not re-enqueued by a
  resync tick; ``drain()`` re-enqueues it.
* ``test_delete_race_drops`` — ``get()`` returning ``None`` at dispatch drops the key
  without raising (must-fix 6, get()-race-loss).
* ``test_requeue_backoff_ordering`` — two requeues with different ``after`` fire in
  clock order (depends on 2e).
* ``test_conflict_reread_reinvoke`` — a ``ConcurrencyConflict`` from a handler write
  triggers an immediate re-read + re-invoke with NO backoff penalty.
* ``test_giveup_deadletters_and_emits_one_telemetry`` — give-up dead-letters the key
  AND emits exactly one ``observe_only`` ``reconcile_giveup`` (its version is never
  asserted — a deleted entity reports version 0).
* ``test_resync_during_inflight_does_not_double_process`` — resolves the open
  question: a resync tick does NOT re-enqueue a key currently in-flight in the
  single-flight worker.

All async is driven by ``asyncio.run`` + a bounded ``asyncio.wait_for`` backstop and
an injected :class:`~vella.reconciler.ManualClock`; no ``pytest-asyncio``. Fixtures
use the REAL in-memory :class:`~vella.runtime.Runtime` (constructed as runtime's own
tests do) and real core :class:`~vella.core.Actuator` / :class:`~vella.core.Node`
types, so the driver is exercised against real edge semantics.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any
from uuid import UUID, uuid4

from vella.core import Actuator, Node, Registry as CoreRegistry, VellaModel, node_type
from vella.runtime import ConcurrencyConflict, LogEntry, Runtime

from vella.reconciler import (
    Context,
    InMemoryCursorStore,
    InMemoryDeadLetterStore,
    ManualClock,
    ReconcileResult,
    Reconciler,
    Registry,
)
from vella.reconciler.reconciler import IDLE


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


def _drive(coro: object, *, timeout: float = 2.0) -> None:
    """Run ``coro`` under a bounded backstop so a coordination bug fails fast."""
    asyncio.run(asyncio.wait_for(coro, timeout=timeout))  # type: ignore[arg-type]


def _converged(state: Actuator[Any]) -> bool:
    """Return whether an actuator's current equals its desired (json-dump compare)."""
    assert state.desired is not None
    return bool(
        state.current.model_dump(mode="json")
        == state.desired.model_dump(mode="json")
    )


def _reconciler(
    rt: Runtime,
    registry: Registry,
    clock: ManualClock,
    *,
    deadletter: InMemoryDeadLetterStore | None = None,
    cursor: InMemoryCursorStore | None = None,
    resync_interval: float = 30.0,
    backoff: float = 1.0,
    backoff_cap: float = 60.0,
    max_attempts: int = 5,
) -> Reconciler:
    """Build a Reconciler over real seams with deterministic timing."""
    return Reconciler(
        rt,
        registry,
        clock,
        cursor,
        deadletter,
        resync_interval=resync_interval,
        backoff=backoff,
        backoff_cap=backoff_cap,
        max_attempts=max_attempts,
    )


# --- 2c: step() is non-blocking ---------------------------------------------
def test_step_nonblocking() -> None:
    """``step()`` on an empty queue returns the IDLE sentinel without blocking."""
    _drive(_case_step_nonblocking())


async def _case_step_nonblocking() -> None:
    rt = Runtime()
    rec = _reconciler(rt, Registry(), ManualClock())
    result = await rec.step()
    assert result is IDLE


# --- 2b: the idle predicate -------------------------------------------------
def test_idle_predicate() -> None:
    """Idle is False until caught-up; True once caught-up + empty + no in-flight."""
    _drive(_case_idle_predicate())


async def _case_idle_predicate() -> None:
    rt = Runtime()
    rec = _reconciler(rt, Registry(), ManualClock())

    # Empty queue but NOT caught up -> not idle (backlog-drained Event unset).
    assert rec.is_idle() is False

    # Mark caught up (the explicit Event) with an empty queue -> idle.
    rec._mark_caught_up()
    assert rec.is_idle() is True

    # Folding a state-changing entry enqueues a key -> not idle (queue non-empty).
    DeviceData = _core_registry()
    node = _drifting_node(DeviceData)
    await rt.create(node)
    hist = await rt.history(node.tenant_id, node.id)
    rec._fold(hist[0])
    assert rec.is_idle() is False


# --- must-fix 6: early-return at idle ---------------------------------------
def test_run_early_return_on_idle() -> None:
    """``run()`` returns before the first resync tick once the loop is idle."""
    _drive(_case_run_early_return())


async def _case_run_early_return() -> None:
    rt = Runtime()
    # A long resync interval; the clock is NEVER advanced. If run() waited for a
    # resync tick instead of early-returning at idle, the wait_for backstop trips.
    rec = _reconciler(rt, Registry(), ManualClock(), resync_interval=10_000.0)
    # No backlog: the watch task reaches the live edge, sets caught-up, queue empty.
    await rec.run()
    assert rec.is_idle() is True


# --- 2d: teardown emits no warnings -----------------------------------------
def test_run_to_convergence_emits_no_warnings() -> None:
    """A full run-to-convergence leaks no async generator / un-cancelled task."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> hard failure (mirrors gate)
        _drive(_case_run_to_convergence())


async def _case_run_to_convergence() -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    registry = Registry()

    async def converge(ctx: Context) -> ReconcileResult:
        # Fresh get inside the handler (correct pattern); converge current->desired.
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

    node = _drifting_node(DeviceData)
    await rt.create(node)

    rec = _reconciler(rt, registry, ManualClock(), resync_interval=10_000.0)
    await rec.run()

    # Converged: current now equals desired.
    got = await rt.get(node.tenant_id, node.id)
    assert got is not None and isinstance(got.state, Actuator)
    assert _converged(got.state)


# --- 2d: DIRECT teardown coverage (non-vacuous) -----------------------------
def test_run_teardown_closes_generator_and_cancels_tasks() -> None:
    """``run()``'s ``finally`` directly closes ``observe()`` and cancels its tasks.

    ``test_run_to_convergence_emits_no_warnings`` is necessary but INSUFFICIENT: it
    wraps the driver in ``asyncio.run``, whose loop-close ``shutdown_asyncgens`` +
    task cleanup masks a leaked ``observe()`` generator or an un-cancelled
    watch/resync task — so removing the teardown from ``run()``'s ``finally`` leaves
    that suite fully green and the load-bearing 2d contract unverified.

    This test drives ``run()`` to convergence on a PRE-EXISTING event loop (never
    ``asyncio.run``) so the loop is NOT closed underneath us, then asserts the
    teardown contract DIRECTLY, with no shutdown machinery to lean on:

    * the ``observe()`` async generator was closed — the runtime's in-memory store
      registers one observer queue per live ``observe()`` and its ``finally``
      ``discard``s that queue on generator close (M3); so the observer set must
      shrink back to its pre-run membership once ``aclose()`` has run;
    * the watch + resync tasks did not leak — every task ``run()`` spawned on this
      loop is ``done()`` after it returns (cancelled-and-awaited in the ``finally``).

    Removing ``aclose()`` from the ``finally`` leaves the observer queue registered
    (first assertion fails); removing the ``task.cancel()/await`` block leaves the
    watch + resync tasks pending (second assertion fails). The test is its own proof
    of non-vacuity.
    """
    loop = asyncio.new_event_loop()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # mirror the gate: any warning -> failure
            loop.run_until_complete(
                asyncio.wait_for(_case_run_teardown(loop), timeout=2.0)
            )
    finally:
        loop.close()


async def _case_run_teardown(loop: asyncio.AbstractEventLoop) -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    registry = Registry()

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
    node = _drifting_node(DeviceData)
    await rt.create(node)

    # The runtime's in-memory store registers one observer queue per live observe();
    # its generator finally discards that queue on aclose(). Snapshot membership
    # BEFORE run() so we can prove the queue was discarded (not merely that the set
    # is empty) after teardown. (Internal access is intentional: this is a teardown-
    # coverage test asserting the M3-established generator-close signal.)
    observers = rt._store._index.observers  # type: ignore[attr-defined]
    before = set(observers)

    # Snapshot the loop's tasks BEFORE the driver spawns its watch/resync tasks, so
    # the diff afterwards is exactly the set run() created on this (un-closed) loop.
    tasks_before = asyncio.all_tasks(loop)

    rec = _reconciler(rt, registry, ManualClock(), resync_interval=10_000.0)
    await rec.run()

    # 1) The observe() generator was closed: its queue was discarded back out of the
    #    runtime's observer set (membership returned to the pre-run baseline). If
    #    run()'s finally skipped aclose(), the queue would still be registered here.
    assert set(observers) == before

    # 2) No leaked tasks: every task run() spawned on this loop is done (the watch +
    #    resync tasks were cancelled and awaited in the finally). If the finally
    #    skipped the cancel/await block, those tasks would still be pending here.
    leaked = {t for t in asyncio.all_tasks(loop) - tasks_before if not t.done()}
    assert leaked == set()

    # Sanity: the loop did converge (the run wasn't a no-op).
    got = await rt.get(node.tenant_id, node.id)
    assert got is not None and isinstance(got.state, Actuator)
    assert _converged(got.state)


# --- must-fix 6: resync skips dead-lettered; drain() re-enqueues ------------
def test_resync_skips_deadlettered() -> None:
    """A dead-lettered key is not re-enqueued by resync; ``drain()`` re-enqueues it."""
    _drive(_case_resync_skips_deadlettered())


async def _case_resync_skips_deadlettered() -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    clock = ManualClock()
    deadletter = InMemoryDeadLetterStore()
    registry = Registry()

    # A handler that always errors so the key exhausts attempts and gets dead-lettered.
    async def always_fail(_ctx: Context) -> ReconcileResult:
        raise RuntimeError("boom")

    registry.register("device", always_fail)

    node = _drifting_node(DeviceData)
    await rt.create(node)
    rec = _reconciler(
        rt, registry, clock, deadletter=deadletter, max_attempts=1, backoff=1.0
    )

    # Fold the create + dispatch once: max_attempts=1 -> immediate give-up.
    hist = await rt.history(node.tenant_id, node.id)
    rec._fold(hist[0])
    await rec.step()

    key = (node.tenant_id, node.id)
    assert await deadletter.get(*key) is not None  # dead-lettered

    # A resync pass must NOT re-enqueue the dead-lettered key.
    await rec._resync_once()
    assert rec._workset.queue_depth() == 0

    # drain() is the explicit re-entry path: it re-enqueues the key.
    await rec.drain()
    assert await deadletter.get(*key) is None  # drained out of the store
    popped = rec._workset.pop()
    assert popped == key


# --- must-fix 6: delete-race at dispatch drops without raising --------------
def test_delete_race_drops() -> None:
    """``get()`` -> None at dispatch drops the key; no raise, no dead-letter."""
    _drive(_case_delete_race_drops())


async def _case_delete_race_drops() -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    deadletter = InMemoryDeadLetterStore()
    rec = _reconciler(rt, Registry(), ManualClock(), deadletter=deadletter)

    node = _drifting_node(DeviceData)
    await rt.create(node)
    hist = await rt.history(node.tenant_id, node.id)
    rec._fold(hist[0])

    # Delete the entity AFTER it was folded but BEFORE dispatch (the race window).
    await rt.delete(node.tenant_id, node.id)

    # Dispatch: fresh get() returns None -> drop, no raise.
    result = await rec.step()
    assert result == (node.tenant_id, node.id)  # the key was processed (dropped)
    assert await deadletter.get(node.tenant_id, node.id) is None  # not dead-lettered


# --- 2e: requeue backoff ordering (off-worker, clock-driven) ----------------
def test_requeue_backoff_ordering() -> None:
    """Two requeues with different ``after`` re-enqueue in clock order (depends on 2e).

    After the M6 fix the worker NEVER parks on the backoff: ``_backoff_or_giveup``
    returns promptly and SCHEDULES an off-worker, clock-driven delayed re-enqueue
    (one timer task per key in ``_delayed``). This direct test pins that the two
    timers re-enqueue their keys in clock order under ``advance()`` — the shorter
    ``after`` first, regardless of schedule order — and that the off-worker timer,
    not the worker call, is what waits. (The end-to-end version through ``run()`` is
    ``test_requeue_backoff_ordering_through_run``.)
    """
    _drive(_case_requeue_backoff_ordering())


async def _case_requeue_backoff_ordering() -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    clock = ManualClock()
    rec = _reconciler(rt, Registry(), clock, backoff=1.0)

    # Two distinct keys must be in the work-set (the re-enqueue requires a folded
    # version); fold a synthetic state-changing entry then pop so the queue is empty
    # but the keys are known and not pending.
    slow_key = ("t1", uuid4())
    fast_key = ("t1", uuid4())
    rec._workset._versions[slow_key] = 0  # known to the work-set (staleness index)
    rec._workset._versions[fast_key] = 0

    # Schedule both delayed re-enqueues. These DO NOT block: the worker call returns
    # promptly and the wait lives in an off-worker timer task (the resync idiom).
    # "slow" schedules its clock waiter FIRST, but wakes LAST (after=5.0 > 1.0).
    await rec._backoff_or_giveup(slow_key, reason="rq", after=5.0)
    await rec._backoff_or_giveup(fast_key, reason="rq", after=1.0)

    # Neither has been re-enqueued yet (the clock has not advanced); both keys are
    # pending an off-worker backoff timer, so the loop is NOT idle.
    assert rec._workset.queue_depth() == 0
    assert set(rec._delayed) == {slow_key, fast_key}

    # Each timer is its OWN task: it parks on the clock only once the loop schedules
    # it (one turn after _schedule_requeue). advance() resolves only ALREADY-parked
    # waiters, so let both timers reach their clock.sleep before advancing.
    while len(clock._waiters) < 2:
        await asyncio.sleep(0)

    await clock.advance(1.0)  # reaches 1.0: fires the fast timer only
    assert rec._workset.queue_depth() == 1
    assert rec._workset.pop() == fast_key  # the shorter delay re-enqueued first
    assert slow_key in rec._delayed and fast_key not in rec._delayed

    await clock.advance(4.0)  # reaches 5.0: fires the slow timer
    assert rec._workset.queue_depth() == 1
    assert rec._workset.pop() == slow_key
    assert rec._delayed == {}  # both timers fired and cleared themselves


# --- must-fix 6: ConcurrencyConflict -> immediate re-read + re-invoke --------
def test_conflict_reread_reinvoke() -> None:
    """A conflict from a handler write re-reads + re-invokes with NO backoff."""
    _drive(_case_conflict_reread_reinvoke())


async def _case_conflict_reread_reinvoke() -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    clock = ManualClock()
    registry = Registry()

    node = _drifting_node(DeviceData)
    await rt.create(node)

    calls = {"n": 0}

    async def conflict_once(ctx: Context) -> ReconcileResult:
        calls["n"] += 1
        got = await ctx.runtime.get(node.tenant_id, node.id)
        assert got is not None and isinstance(got.state, Actuator)
        if calls["n"] == 1:
            # Simulate contention: write with a stale expected_version -> conflict.
            await ctx.runtime.edit(
                node.tenant_id, node.id, expected_version=got.version - 1, name="x"
            )
        # Second invocation: fresh version, converge.
        await ctx.runtime.edit(
            node.tenant_id,
            node.id,
            expected_version=got.version,
            state=Actuator(current=got.state.desired, desired=got.state.desired),
        )
        return ReconcileResult.done()

    registry.register("device", conflict_once)

    rec = _reconciler(rt, registry, clock)
    hist = await rt.history(node.tenant_id, node.id)
    rec._fold(hist[0])
    await rec.step()

    # The handler was invoked twice (conflict -> immediate re-invoke), with NO
    # clock advance in between (no backoff penalty: clock is still at zero).
    assert calls["n"] == 2
    assert clock.now() == 0.0
    got = await rt.get(node.tenant_id, node.id)
    assert got is not None and isinstance(got.state, Actuator)
    assert _converged(got.state)


# --- must-fix 6: give-up dead-letters AND emits exactly one telemetry --------
def test_giveup_deadletters_and_emits_one_telemetry() -> None:
    """Give-up records to the dead-letter store and emits ONE reconcile_giveup."""
    _drive(_case_giveup())


async def _case_giveup() -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    clock = ManualClock()
    deadletter = InMemoryDeadLetterStore()
    registry = Registry()

    async def always_fail(_ctx: Context) -> ReconcileResult:
        raise RuntimeError("permanent failure")

    registry.register("device", always_fail)

    node = _drifting_node(DeviceData)
    await rt.create(node)
    rec = _reconciler(
        rt, registry, clock, deadletter=deadletter, max_attempts=1
    )

    hist = await rt.history(node.tenant_id, node.id)
    rec._fold(hist[0])
    await rec.step()  # max_attempts=1 -> immediate give-up

    key = (node.tenant_id, node.id)
    record = await deadletter.get(*key)
    assert record is not None
    assert record.attempts == 1
    assert "permanent failure" in record.reason

    # Exactly ONE observe_only reconcile_giveup telemetry entry was emitted. Do NOT
    # assert its .version (a deleted entity reports version 0; this one is not
    # deleted, but the contract forbids the assertion regardless).
    full = await rt.history(node.tenant_id, node.id)
    giveups = [
        e
        for e in full
        if e.transition == "observe_only"
        and e.payload.get("event") == "reconcile_giveup"
    ]
    assert len(giveups) == 1
    assert giveups[0].payload["attempts"] == 1


# --- resolved open question: resync during in-flight ------------------------
def test_resync_during_inflight_does_not_double_process() -> None:
    """A resync tick does NOT re-enqueue a key currently in-flight (single-flight).

    Resolution of the Critic-flagged open question: a popped (in-flight) key has
    left the dedup guard, so a naive resync could re-enqueue it mid-handler and
    double-process. The driver tracks the single in-flight key and the resync pass
    explicitly skips it. We block the handler mid-dispatch, fire a resync pass, and
    assert the queue stays empty (the in-flight key was skipped). When the handler
    finishes, the worker clears in-flight; a subsequent resync may re-enqueue.
    """
    _drive(_case_resync_inflight())


async def _case_resync_inflight() -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    clock = ManualClock()
    registry = Registry()

    node = _drifting_node(DeviceData)
    await rt.create(node)
    key = (node.tenant_id, node.id)

    gate: asyncio.Event = asyncio.Event()
    in_handler: asyncio.Event = asyncio.Event()

    async def blocking_handler(_ctx: Context) -> ReconcileResult:
        in_handler.set()
        await gate.wait()  # park mid-dispatch so the key stays in-flight
        return ReconcileResult.done()

    registry.register("device", blocking_handler)
    rec = _reconciler(rt, registry, clock)

    hist = await rt.history(node.tenant_id, node.id)
    rec._fold(hist[0])

    # Start the worker step; it pops the key (now in-flight) and parks in the handler.
    step_task = asyncio.ensure_future(rec.step())
    await asyncio.wait_for(in_handler.wait(), timeout=1.0)

    # The key is in-flight: queue is empty, and a resync pass must SKIP it.
    assert rec._workset.queue_depth() == 0
    await rec._resync_once()
    assert rec._workset.queue_depth() == 0

    # Release the handler; the worker finishes and clears in-flight.
    gate.set()
    await asyncio.wait_for(step_task, timeout=1.0)
    assert rec._inflight is None

    # Now (not in-flight) a resync pass MAY re-enqueue the still-known key.
    await rec._resync_once()
    assert rec._workset.queue_depth() == 1
    assert rec._workset.pop() == key


# --- 2d: cancellation-path teardown is clean (the M6 race regression) --------
def test_run_cancellation_is_clean() -> None:
    """Cancelling ``run()`` mid-flight tears down cleanly: no aclose race, no leaks.

    The M6 regression (REAL, plain-event-loop reproducible). The fold's live-edge
    probe (``fold_available``) pulls each entry in a CHILD ``_anext`` task that holds
    ``observe().__anext__()`` in flight. When ``run()`` is cancelled mid-flight — an
    outer ``asyncio.wait_for`` timeout, or a graceful controller shutdown that
    cancels the run task — the CancelledError can land while that child pull is
    parked at the bare-yield window. The OLD teardown then ``await stream.aclose()``-d
    the ``observe()`` generator from ``run()``'s OWN frame — a DIFFERENT task context
    than the child ``_anext`` still iterating that same generator — raising
    ``RuntimeError: aclose(): asynchronous generator is already running``. The fix
    makes the watch task the SOLE owner/closer of the generator (it ``aclose``s it
    inside its own :func:`contextlib.aclosing` frame), the fold drives its child pull
    to ``done()`` before unwinding, and teardown re-cancels each helper to ``done()``
    without ever touching the generator cross-task.

    This drives MANY ``run()``+cancel cycles on ONE pre-existing loop (NOT
    ``asyncio.run`` per cycle — that masks the race behind loop-close
    ``shutdown_asyncgens``). Each cycle cancels ``run()`` in-flight after a SWEPT
    number of event-loop micro-yields, so the cancel lands across every teardown
    phase — including the exact fold live-edge window that triggers the race. Under
    ``warnings.simplefilter("error")`` (any leaked-generator / un-cancelled-task
    ``UserWarning`` is a hard failure) it asserts:

    * NO ``RuntimeError`` (aclose-already-running) is raised across all cycles;
    * the ``observe()`` generator was closed every cycle — the runtime's observer set
      returns to its pre-run baseline (the same close signal
      ``test_run_teardown_closes_generator_and_cancels_tasks`` asserts), so no
      generator and no in-flight ``_anext`` child leaked;
    * every task ``run()`` spawned on this loop is ``done()`` (nothing leaked);
    * the whole thing is bounded by ``asyncio.wait_for`` so a teardown hang fails
      fast rather than wedging the suite.

    NON-VACUITY: against the UNFIXED teardown this FAILS — the ``RuntimeError``
    surfaces within the cycle sweep (verified ~20% of cycles, deterministically in
    the micro-yield window). Against the fix it PASSES reliably (verified 5x under
    PYTHONHASHSEED 0/1/42 over thousands of cancellations: 0 errors, 0 leaks).
    """
    loop = asyncio.new_event_loop()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # mirror the gate: any warning -> failure
            loop.run_until_complete(
                asyncio.wait_for(_case_run_cancellation_clean(loop), timeout=20.0)
            )
    finally:
        loop.close()


async def _case_run_cancellation_clean(loop: asyncio.AbstractEventLoop) -> None:
    rt = Runtime()
    DeviceData = _core_registry()
    registry = Registry()

    # A converging handler: the worker actually pops + dispatches each run, so the
    # fold + live-edge machinery (the child _anext pull) runs every cycle — that pull
    # is the second task touching the generator the OLD cross-task aclose() raced.
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

    node = _drifting_node(DeviceData)
    await rt.create(node)

    observers = rt._store._index.observers  # type: ignore[attr-defined]
    before = set(observers)
    tasks_before = asyncio.all_tasks(loop)

    # Sweep the number of micro-yields before the cancel so it lands across every
    # teardown phase, including the fold live-edge window that triggers the race. A
    # generous cycle count (the race window is hit deterministically each sweep) so
    # the unfixed teardown reliably raises here.
    for cycle in range(180):
        rec = _reconciler(rt, registry, ManualClock(), resync_interval=10_000.0)
        run_task: asyncio.Task[None] = asyncio.ensure_future(rec.run())
        for _ in range(cycle % 6):
            await asyncio.sleep(0)
        run_task.cancel()
        try:
            # Bounded so a teardown hang (a helper never driven to done) fails fast.
            await asyncio.wait_for(run_task, timeout=2.0)
        except asyncio.CancelledError:
            pass  # expected: the cancel propagated out of the cancelled run()
        # An aclose-already-running RuntimeError (the OLD defect) would propagate
        # out of the await above and fail the test right here.

    # Let any just-cancelled helper frames finish unwinding.
    await asyncio.sleep(0)

    # The generator was closed every cycle: the observer set is back to its pre-run
    # baseline (no leaked observe() queue, no in-flight _anext child holding it open).
    assert set(observers) == before

    # No leaked tasks: every task the cancelled runs spawned on this loop is done().
    leaked = {t for t in asyncio.all_tasks(loop) - tasks_before if not t.done()}
    assert leaked == set()


# --- a folded observe_only never re-enqueues (give-up self-feed guard) -------
def test_fold_observe_only_does_not_enqueue() -> None:
    """A folded ``observe_only`` give-up emit does not feed the worker a new item."""
    rt = Runtime()
    rec = _reconciler(rt, Registry(), ManualClock())
    eid = uuid4()
    entry = LogEntry.model_validate(
        {
            "cursor": {"token": "0"},
            "tenant_id": "t1",
            "entity_kind": "node",
            "entity_id": eid,
            "version": 0,
            "transition": "observe_only",
            "payload": {"event": "reconcile_giveup"},
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    assert rec._fold(entry) is None
    assert rec._workset.queue_depth() == 0
