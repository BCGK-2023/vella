"""DIRECT teardown coverage for ``GraphFollower`` (M6) — non-vacuous (2d).

The package gate's ``filterwarnings = ["error::UserWarning"]`` is NECESSARY but
INSUFFICIENT on its own: an ``asyncio.run``-wrapped follower would have its leaked
``observe()`` generator and un-cancelled watch task swept up by the loop-close
``shutdown_asyncgens`` + task cleanup, leaving a broken teardown fully green. So
this is the LOAD-BEARING gate — it drives the follower on a PRE-EXISTING event loop
(``asyncio.new_event_loop()`` + ``run_until_complete``, NEVER ``asyncio.run``) so
the loop is not closed underneath us, then asserts the teardown contract DIRECTLY:

* (a) the ``observe()`` generator was closed — the runtime's in-memory store
  registers one observer queue per live ``observe()`` and its ``finally`` discards
  that queue on generator close, so the observer set must return to its pre-run
  baseline once ``aclose()`` has run;
* (b) the watch task did not leak — every task the follower spawned on this loop is
  ``done()`` after ``run()`` returns (cancelled-and-awaited in the ``finally``).

This mirrors the reconciler's direct teardown test verbatim, adapted to the
follower. Non-vacuity (which mutation each assertion catches), with a DEVIATION
from the plan's stated coverage that is forced by Python's async-generator
finalization semantics and proven empirically (see the module-level note below):

* **mut-m6-cross-task-aclose** (``aclose`` the generator from the teardown frame
  while the watch's carried pull holds ``__anext__`` in flight) →
  ``RuntimeError: aclose(): asynchronous generator is already running``. Proven
  CONSTRUCTIVELY by :func:`test_cross_task_aclose_raises_already_running` below —
  this is what makes the single-task generator-ownership discipline load-bearing.
* assertion (b) **task-done** catches a teardown that forgets to cancel/await the
  watch task: the watch would stay pending here.

DEVIATION (reported, not hidden): the plan expects assertion (a) [observer-set
baseline] to turn RED under **mut-m6-vacuous-aclose** and the watch-task-done
assertion to turn RED under **mut-m6-oneshot-cancel**. Neither holds for THIS
follower over the in-memory runtime, because:

* the in-memory ``observe()`` generator runs its ``finally`` (the observer-queue
  ``discard``) whenever its ``__anext__`` is FINALIZED — by exhaustion OR by the
  cancellation of the in-flight pull at teardown — NOT only by ``aclose()``. So the
  observer set returns to baseline even if the ``contextlib.aclosing`` is a no-op
  (mut-m6-vacuous-aclose); ``aclosing`` remains correct, idiomatic, defensive
  discipline (and the cross-task case above proves single-task ownership matters),
  but the observer-baseline cannot OBSERVE its presence — empirically verified.
* this follower blocks on a single carried ``await fetch`` (not a re-parking live
  ``async for``), so a single ``task.cancel()`` terminates the watch cleanly; the
  re-cancel-to-done loop is belt-and-suspenders here rather than strictly required
  (also empirically verified). The loop is kept for robustness + reconciler parity.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any
from uuid import uuid4

from vella.core import EdgeTypes
from vella.runtime import Runtime

from vella.graph import GraphFollower, GraphProjection, GraphView, ManualClock

from _fixtures import make_node, thing_registry


def test_run_teardown_closes_generator_and_cancels_tasks() -> None:
    """``run()``'s ``finally`` directly closes ``observe()`` and cancels its task.

    Drives the follower to quiescence on a PRE-EXISTING event loop (never
    ``asyncio.run``) so the loop is NOT closed underneath us, then asserts the
    teardown contract DIRECTLY, with no shutdown machinery to lean on. The test is
    its own proof of non-vacuity (see module docstring for the three mutations).
    """
    loop = asyncio.new_event_loop()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # mirror the gate: any warning -> failure
            loop.run_until_complete(
                asyncio.wait_for(_case_run_teardown(loop), timeout=3.0)
            )
    finally:
        loop.close()


async def _case_run_teardown(loop: asyncio.AbstractEventLoop) -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b = uuid4(), uuid4()
    for nid in (a, b):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    # The runtime's in-memory store registers one observer queue per live observe();
    # its generator finally discards that queue on aclose(). Snapshot membership
    # BEFORE run() so we can prove the queue was discarded (not merely that the set
    # is empty) after teardown. (Internal access is the documented teardown-coverage
    # signal — the M3/M4-established generator-close discipline.)
    observers = rt._store._index.observers  # type: ignore[attr-defined]
    before = set(observers)

    # Snapshot the loop's tasks BEFORE the follower spawns its watch task, so the
    # diff afterwards is exactly the set run() created on this (un-closed) loop.
    tasks_before = asyncio.all_tasks(loop)

    follower = GraphFollower(rt, tenant, mode="full", clock=ManualClock())
    # max_steps=1: the run loop returns after the FIRST caught-up pass, so the watch
    # task is parked on the LIVE EDGE (carrying __anext__ in flight) when the finally
    # tears it down — the teardown must cancel-and-await it to done() here.
    await follower.run(max_steps=1)

    # (a) The observe() generator was closed: its queue was discarded back out of the
    #     runtime's observer set (membership returned to the pre-run baseline). See the
    #     module docstring DEVIATION: this proves the generator was finalized + no leak;
    #     it does NOT specifically prove aclose() ran (cancellation finalizes it too).
    assert set(observers) == before

    # (b) No leaked tasks: every task run() spawned on this loop is done (the watch
    #     task was cancelled and awaited in the teardown finally). A teardown that
    #     forgot to cancel/await the watch would leave it pending here.
    leaked = {t for t in asyncio.all_tasks(loop) - tasks_before if not t.done()}
    assert leaked == set()

    # Sanity: the run actually folded the backlog (not a no-op).
    assert b in {r.to_id for r in follower.view()._internal_index().neighbors(a, "out")}


def test_cross_task_aclose_raises_already_running() -> None:
    """Single-task ownership is load-bearing: cross-task ``aclose()`` raises (2d).

    This is the CONSTRUCTIVE proof of ``mut-m6-cross-task-aclose``. The follower's
    watch task owns its ``observe()`` generator and carries one ``__anext__`` pull in
    flight while parked on the live edge. If teardown ``aclose()``d that generator
    from a DIFFERENT task (the bug the single-task discipline forbids) while the pull
    is running, CPython raises ``RuntimeError: aclose(): asynchronous generator is
    already running``. We reproduce exactly that race directly on a never-closed loop
    to prove the discipline is necessary (the follower NEVER does this — its
    ``aclose()`` touches only the task, never the generator).
    """
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(_case_cross_task_aclose(), timeout=3.0))
    finally:
        loop.close()


async def _case_cross_task_aclose() -> None:
    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b = uuid4(), uuid4()
    for nid in (a, b):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    stream = rt.observe(since=None)

    async def _pull() -> Any:
        return await stream.__anext__()

    # Drain the backlog, then leave a pull parked on the live edge (running __anext__
    # in this `fetch` task) — exactly the follower's carried-pull state at teardown.
    fetch: "asyncio.Task[Any]" = asyncio.ensure_future(_pull())
    while True:
        await asyncio.sleep(0)
        if fetch.done():
            try:
                fetch.result()
            except StopAsyncIteration:
                break
            fetch = asyncio.ensure_future(_pull())
            continue
        break  # live edge: fetch is parked, running __anext__

    raised = False
    try:
        await stream.aclose()  # cross-task aclose while fetch's __anext__ is running
    except RuntimeError as exc:
        raised = "already running" in str(exc)
    assert raised, "cross-task aclose did not raise the 'already running' RuntimeError"

    # Clean up the parked pull (the follower drives this to done() in its watch frame).
    fetch.cancel()
    while not fetch.done():
        try:
            await fetch
        except (asyncio.CancelledError, StopAsyncIteration):
            if fetch.done():
                break


def test_aclose_is_idempotent_and_safe_before_run() -> None:
    """``aclose()`` on a never-run follower is a no-op (no task, no error/warning)."""
    loop = asyncio.new_event_loop()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loop.run_until_complete(asyncio.wait_for(_case_aclose_unstarted(), timeout=2.0))
    finally:
        loop.close()


async def _case_aclose_unstarted() -> None:
    rt = Runtime()
    follower = GraphFollower(rt, "t", mode="full", clock=ManualClock())
    await follower.aclose()  # no watch task spawned yet — must be a clean no-op
    await follower.aclose()  # idempotent


# --- regression guard: fold + refresh need NO Clock and NO running task -----
def test_fold_and_refresh_need_no_clock_or_task() -> None:
    """The cold fold + pull ``refresh()`` are one-shot — no follower, Clock, or task.

    The follower is OPT-IN: ``GraphProjection.fold`` and ``GraphView.refresh`` must
    keep working as plain coroutines with no ``Clock`` injected and no background
    task spawned. Guards against M6 accidentally coupling the pull path to the
    follower's async lifecycle.
    """
    loop = asyncio.new_event_loop()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loop.run_until_complete(
                asyncio.wait_for(_case_pull_path_no_task(loop), timeout=2.0)
            )
    finally:
        loop.close()


async def _case_pull_path_no_task(loop: asyncio.AbstractEventLoop) -> None:
    # Snapshot INSIDE the case coroutine: by now the wait_for wrapper task and this
    # case task already exist, so any task that appears AFTER this point is genuinely
    # spawned by the pull path (a follower would add a watch task; the pull path adds
    # none).
    tasks_before = asyncio.all_tasks(loop)

    rt = Runtime()
    thing = thing_registry()
    tenant = "t"
    a, b = uuid4(), uuid4()
    for nid in (a, b):
        await rt.create(make_node(thing, tenant_id=tenant, node_id=nid))
    await rt.link(tenant, a, b, EdgeTypes.KNOWS)

    # No Clock anywhere on the pull path.
    view: GraphView = await GraphProjection().fold(rt, tenant, mode="full")
    c = uuid4()
    await rt.create(make_node(thing, tenant_id=tenant, node_id=c))
    await rt.link(tenant, a, c, EdgeTypes.OWNED_BY)
    refreshed = await view.refresh(rt)
    assert sorted(str(r.to_id) for r in refreshed._internal_index().neighbors(a, "out")) == sorted(
        [str(b), str(c)]
    )

    # The pull path spawned NO background task.
    spawned = {t for t in asyncio.all_tasks(loop) - tasks_before}
    assert spawned == set(), f"pull path spawned background tasks: {spawned}"
