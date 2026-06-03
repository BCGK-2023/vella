"""The opt-in background follower — ``GraphFollower`` (M6).

``GraphFollower`` is the ONLY async-lifecycle surface in the package. The cold
fold (``GraphProjection.fold``) and the pull ``refresh()`` are one-shot coroutines
that need no running task and no :class:`~vella.graph.Clock`; the follower is the
opt-in alternative for non-pull consumers that want a :class:`~vella.graph.GraphView`
kept *current* by a background task watching ``observe()``'s live tail.

It ports the reconciler's lifecycle discipline, adapted to a single long-lived
``observe()`` generator:

* **Single-task generator ownership.** The follower owns its ``observe()``
  generator inside its OWN watch task frame via :func:`contextlib.aclosing`; only
  that frame ``aclose()``s it. No other task (notably teardown / :meth:`aclose`)
  ever touches the generator — a cross-task ``aclose()`` while the generator's
  ``__anext__`` is in flight raises ``RuntimeError: aclose(): asynchronous generator
  is already running`` (proven load-bearing by ``test_follower_teardown``). The
  watch's cancellation-robust ``finally`` drives the in-flight pull to ``done()`` so
  the owning frame's ``aclose()`` never races a still-running generator.
* **Carried pull (a forced deviation from ``bounded_drain``).** The generic
  ``bounded_drain`` (used by the one-shot ``fold`` / ``refresh``) ``cancel()``s its
  in-flight ``__anext__`` probe at each live edge, which FINALIZES the in-memory
  ``observe()`` generator (its ``finally`` runs) — fine for callers that then reopen,
  fatal for a long-lived follower that must keep the SAME generator to see live
  entries. So the watch CARRIES one persistent pull across the live edge instead of
  cancelling it (see :meth:`_watch`); the pull is cancelled only at teardown.
* **Re-cancel-to-done teardown.** :meth:`aclose` cancels the watch task and
  re-cancels each loop turn until ``task.done()`` (reconciler parity + robustness).
* **Quiescence is an explicit Event, not sleep-and-hope.** The watch sets
  :attr:`caught_up` (and bumps a monotonic generation) each time the live edge is
  reached AFTER the view reflects every drained entry; callers/tests
  ``await follower.caught_up.wait()`` and assert equivalence AT that signal.

The current state is updated incrementally via the SAME copy-on-write delta path as
``GraphView.refresh`` (``_fold.authority_pass`` + ``GraphIndex.apply_delta``), so a
followed view at quiescence is byte-identical to a fresh fold to the same cursor.

Note on ``Clock``: the v0.1 loop is purely event-driven on ``observe()`` (it blocks
on the live tail and wakes per entry; it has NO timer behaviour). The public
``Clock``/``ManualClock`` surface is kept as the plan-specified injectable-time seam
and supported test driver (and the structural Protocol-conformance proof), but the
follower's loop does not call ``clock.sleep`` — the clock is effectively vestigial in
the loop. This is reported as a deviation rather than padded with fabricated timers.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, Callable, Optional
from uuid import UUID

from vella.core import Edge
from vella.runtime import Runtime

from vella.runtime import Cursor, LogEntry

from ._fold import authority_pass, seed_state
from ._index import GraphIndex
from .clock import Clock, SystemClock
from .mode import MaterializationMode
from .view import GraphView

if TYPE_CHECKING:
    from typing import AsyncIterator


class GraphFollower:
    """Keeps a :class:`~vella.graph.GraphView` current from ``observe()``'s tail.

    Opt-in: construct one, ``await run()`` (typically as a task) to fold the backlog
    and then track the live tail, ``await caught_up.wait()`` to observe quiescence,
    read :meth:`view` for the current snapshot, and ``await aclose()`` to tear down.
    The cold fold and ``refresh()`` need NONE of this — the follower is the only
    place a :class:`~vella.graph.Clock` or a running task enters the package.

    Examples:
        >>> from vella.graph import GraphFollower, ManualClock
        >>> from vella.runtime import Runtime
        >>> follower = GraphFollower(Runtime(), "tenant", clock=ManualClock())
        >>> type(follower).__name__
        'GraphFollower'
    """

    def __init__(
        self,
        runtime: Runtime,
        tenant_id: str,
        *,
        mode: MaterializationMode = "full",
        weight: Optional[Callable[[Edge[Any, Any]], float]] = None,
        lru_capacity: int = 1024,
        clock: Optional[Clock] = None,
    ) -> None:
        """Create a follower for ``tenant_id`` over ``runtime`` (does not start it).

        Args:
            runtime: The runtime whose log is followed (read-only ``observe``/``get``).
            tenant_id: The tenant this follower projects; other tenants are skipped.
            mode: ``"full"`` holds node+edge bodies resident; ``"lean"`` holds none.
            weight: Optional pure ``Edge -> float`` baked onto each edge record.
            lru_capacity: The bound for the ``lean`` hydration LRU on each view.
            clock: The injectable time seam; defaults to a :class:`SystemClock`.
                A :class:`ManualClock` is injected in tests. The v0.1 loop is purely
                event-driven on ``observe()`` (it has no timer behaviour), so the
                clock is carried as the documented injectable-time seam rather than
                driving any sleep — see the module docstring.
        """
        self._runtime = runtime
        self._tenant_id = tenant_id
        self._mode: MaterializationMode = mode
        self._weight = weight
        self._lru_capacity = lru_capacity
        self._clock: Clock = clock if clock is not None else SystemClock()
        # The current folded state, tracked on the follower (not reached out of the
        # view) so updates stay package-internal: an empty index + no high-water until
        # run() folds anything. :meth:`view` snapshots these into a frozen GraphView.
        self._index: GraphIndex = GraphIndex.build([], {})
        self._high_water: Optional[Cursor] = None
        self._resident: dict[UUID, Any] = {}
        # The explicit quiescence signal callers await (NOT sleep-and-hope): SET by
        # the watch task whenever the live edge is reached (the view reflects every
        # entry up to that point) and CLEARED by the watch when it picks up a new
        # live entry. Only the watch task ever touches it, so there is no
        # producer/consumer clear race.
        self.caught_up: asyncio.Event = asyncio.Event()
        # Monotonic count of caught-up passes the watch has reached; ``run`` waits on
        # this (never on clearing the Event) so a bounded run cannot lose a wakeup.
        self._generation: int = 0
        self._watch_task: Optional[asyncio.Task[None]] = None

    @property
    def clock(self) -> Clock:
        """The injected :class:`~vella.graph.Clock` (the injectable-time seam)."""
        return self._clock

    def view(self) -> GraphView:
        """The current followed :class:`~vella.graph.GraphView`.

        Updated incrementally via the same copy-on-write delta path as
        ``GraphView.refresh`` — untouched index buckets are shared by identity, so
        reading the view between updates is cheap and the prior view is never
        mutated. At quiescence (``await caught_up.wait()``) this equals a fresh fold
        to the same cursor.

        Returns:
            The latest :class:`~vella.graph.GraphView` the follower has folded.
        """
        return GraphView(
            index=self._index,
            mode=self._mode,
            high_water=self._high_water,
            resident=dict(self._resident),
            tenant_id=self._tenant_id,
            runtime=self._runtime,
            weight=self._weight,
            lru_capacity=self._lru_capacity,
        )

    async def run(self, max_steps: Optional[int] = None) -> None:
        """Fold the backlog, then track the live tail until cancelled or bounded out.

        Spawns a single watch task that OWNS the ``observe()`` generator end to end
        (opened inside :func:`contextlib.aclosing` in its own frame). The run loop
        yields to let the watch task drain entries and incrementally update the view,
        returning when ``max_steps`` live-edge passes have been observed (a bounded
        test driver) or running until cancelled (``max_steps=None``). On EVERY exit
        path — normal return OR an outer cancellation — the ``finally`` tears the
        watch task down (see :meth:`aclose`) so no task and no generator leaks.

        Args:
            max_steps: Optional bound on the number of caught-up (live-edge) passes
                to observe before returning; ``None`` runs until cancelled. Bounds
                the no-``pytest-asyncio`` test driver so a regression fails fast
                under ``asyncio.wait_for`` rather than hanging.
        """
        # SINGLE-TASK GENERATOR OWNERSHIP: the watch task opens, iterates, AND closes
        # the observe() generator inside its own frame (see :meth:`_watch`). run() and
        # aclose() NEVER touch the generator — so even when run() is cancelled
        # mid-flight, the only aclose() runs as part of the watch task's own unwind,
        # never cross-task, so "aclose(): already running" cannot occur. The watch
        # task blocks forever on the live edge; run() decides when to stop and the
        # finally cancels the (still-parked) watch task — exactly the reconciler's
        # run()/teardown split, which is what makes the re-cancel-to-done discipline
        # load-bearing (a one-shot cancel cannot terminate a live-edge-parked task).
        self._watch_task = asyncio.ensure_future(self._watch())
        try:
            observed = 0
            while max_steps is None or observed < max_steps:
                if self._watch_task.done():
                    # The watch task ended (stream exhausted / errored / externally
                    # cancelled via aclose()). Surface a genuine error but treat a
                    # cancellation as a normal stop — the watch being cancelled (e.g.
                    # a concurrent aclose()) is teardown, not a failure to propagate.
                    try:
                        await self._watch_task
                    except asyncio.CancelledError:
                        pass
                    return
                # Wait for the NEXT caught-up pass. Quiescence is counted by the
                # monotonic generation the watch bumps (NOT by clearing the Event,
                # which would race the watch's own clear) — yield until it advances
                # past what we have already observed. The yield (sleep(0)) is what lets
                # the watch task make progress (fold the backlog / block on the tail).
                while self._generation <= observed and not self._watch_task.done():
                    await asyncio.sleep(0)
                if self._generation > observed:
                    observed = self._generation
        finally:
            await self.aclose()

    async def _watch(self) -> None:
        """Own the ``observe()`` generator; fold the backlog, then track the live tail.

        Opens ``observe(since=None)`` inside an :func:`contextlib.aclosing` block so
        the generator is ``aclose``d by THIS task's own frame unwinding — whether the
        loop completes, raises, or this task is cancelled. No other task ever calls
        ``aclose()`` on it, which is what frees teardown from the cross-task
        "aclose(): already running" race.

        **Why this re-implements the drain rather than calling ``bounded_drain``
        (a documented, forced deviation):** ``bounded_drain`` (used by the one-shot
        ``fold`` / ``refresh``) ``cancel()``s its in-flight ``__anext__`` probe at
        every live edge. For the in-memory runtime that ``cancel()`` throws into the
        generator's parked ``await queue.get()`` and FINALIZES the generator — the
        one-shot callers then ``aclose()`` and reopen, so they never notice. A
        long-lived follower CANNOT reopen per pass (it would miss live entries and
        re-fold the whole backlog), so it must CARRY the parked pull across the live
        edge instead of cancelling it: a single persistent ``fetch`` task holds
        ``__anext__`` in flight; an immediately-available entry resolves it on a bare
        ``sleep(0)`` (drain), and at the live edge the SAME parked ``fetch`` is
        awaited (block) to deliver the next live entry — never cancelled until
        teardown. The cancellation-robust ``finally`` drives ``fetch`` to ``done()``
        on EVERY exit (normal, error, or an outer cancellation thrown into a bare
        ``await``), exactly like ``bounded_drain``'s, so the owning frame's
        ``aclose()`` never races a still-running generator.

        Each caught-up pass folds the batch drained since the last live edge into ONE
        delta accumulator (seeded from the current view's live set — the ``refresh()``
        delta discipline), COW-updates the view, and ONLY THEN signals quiescence (so
        a waiter woken by the Event always observes the already-updated view, never a
        stale snapshot). The Event is cleared (by THIS task only — no consumer clear,
        no race) when a new live entry is picked up after a caught-up pass.
        """
        async with contextlib.aclosing(self._runtime.observe(since=None)) as stream:
            # One persistent pull held in flight across the whole watch — carried over
            # the live edge (never cancelled until teardown) so the in-memory
            # generator is not finalized by a cancelled __anext__ (see method doc).
            fetch: asyncio.Task[Any] = asyncio.ensure_future(_anext(stream))
            try:
                state = self._seed_state()
                drained = 0
                signalled_once = False
                while True:
                    # Bare yield: an immediately-available entry resolves `fetch` on
                    # this turn; otherwise `fetch` parks on the live edge (we block on
                    # it below). This is bounded_drain's race kernel, inlined so the
                    # losing pull is CARRIED rather than cancelled.
                    await asyncio.sleep(0)
                    if fetch.done():
                        try:
                            entry = fetch.result()
                        except StopAsyncIteration:
                            # Finite/exhausted stream (e.g. a test stream): apply the
                            # final batch, signal, and stop.
                            if drained:
                                await self._apply(state)
                            self._signal_caught_up()
                            return
                        state.apply(entry)
                        drained += 1
                        fetch = asyncio.ensure_future(_anext(stream))
                        continue
                    # Live edge: the batch drained since the last edge is complete.
                    if drained:
                        await self._apply(state)
                    elif signalled_once:
                        # No new entries since the last caught-up pass: re-arm the
                        # Event so a consumer that appended an entry blocks on the
                        # NEXT genuine quiescence rather than seeing a stale set.
                        self.caught_up.clear()
                    self._signal_caught_up()
                    signalled_once = True
                    # Block on the carried pull for the next live entry, then start a
                    # fresh batch. The Event is cleared on pick-up so a waiter sees the
                    # post-apply view, not this pre-apply one.
                    entry = await fetch
                    self.caught_up.clear()
                    state = self._seed_state()
                    state.apply(entry)
                    drained = 1
                    fetch = asyncio.ensure_future(_anext(stream))
            finally:
                # Drive the carried pull to done() unconditionally — on normal return,
                # error, OR a CancelledError thrown into a bare await above — so no
                # pull leaks past this frame and the aclose() (the async with exit)
                # never races a still-running generator. Cancellation-robust: an outer
                # cancel re-thrown into us must not abandon `fetch` still-pending.
                fetch.cancel()
                while not fetch.done():
                    try:
                        await fetch
                    except (asyncio.CancelledError, StopAsyncIteration):
                        if fetch.done():
                            break

    def _signal_caught_up(self) -> None:
        """Mark one caught-up pass: bump the monotonic generation and set the Event.

        Called by the watch task only, AFTER the view reflects every entry up to the
        live edge — so a waiter woken by the Event always observes a current view and
        ``run``'s generation wait counts a real quiescence (never a stale one).
        """
        self._generation += 1
        self.caught_up.set()

    def _seed_state(self) -> Any:
        """Seed a fresh delta accumulator from the current folded live set.

        A ``delete`` / ``unlink`` in the next drained slice removes a prior-live
        entity and a ``create`` / ``link`` adds one — exactly the ``refresh()`` delta
        discipline, carrying the current high-water forward unchanged on an empty
        slice.

        Returns:
            A ``_FoldState`` seeded from the current index's live node/edge ids.
        """
        return seed_state(
            self._tenant_id,
            live_nodes=set(self._index.node_types),
            live_edges=set(self._index.live_edges),
            high_water=self._high_water,
        )

    async def _apply(self, state: Any) -> None:
        """Copy-on-write the follower's index from a drained delta accumulator.

        Runs the shared authority pass over the delta's recomputed live set and folds
        it into a NEW index via ``GraphIndex.apply_delta`` (untouched buckets shared
        by identity), then swaps in the new index + high-water + resident bodies. The
        prior index is never mutated, so a :meth:`view` snapshot taken before this
        apply keeps querying its own frozen index.

        Args:
            state: The drained ``_FoldState`` carrying the live id-sets + high-water.
        """
        records, node_types, resident = await authority_pass(
            self._runtime,
            self._tenant_id,
            state,
            mode=self._mode,
            weight=self._weight,
        )
        self._index = self._index.apply_delta(records, node_types)
        self._high_water = state.high_water
        self._resident = resident

    async def aclose(self) -> None:
        """Cancel the watch task and await it to ``done()`` (re-cancel-to-done).

        The watch task ``aclose``s the ``observe()`` generator inside its OWN frame
        as it unwinds — this method NEVER touches the generator (a cross-task
        ``aclose()`` while ``__anext__`` is in flight raises ``RuntimeError:
        aclose(): asynchronous generator is already running``).

        The await is cancellation-robust: if the follower itself is being cancelled,
        a :class:`asyncio.CancelledError` re-thrown into our own ``await`` must NOT
        leave the watch task pending (a leaked task surfaces as a ``UserWarning`` and
        turns the gate red). So the task is RE-``cancel()``-ed each loop turn until it
        reports ``done()`` — a single cancel can be transiently swallowed (the live
        ``async for`` re-parks on the live edge), so re-cancelling drives it to
        ``done()`` for certain. Idempotent: a no-op once the task is done / unstarted.
        """
        task = self._watch_task
        if task is None:
            return
        while not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Either the watch task's own cancellation completing, OR an outer
                # cancellation re-thrown into this frame. If the task is now done the
                # loop exits; otherwise re-cancel + re-await so it cannot leak.
                if task.done():
                    break
                continue
            except Exception:  # noqa: BLE001 - the watch task's terminal error is
                # surfaced by awaiting it; aclose swallows it so the original run()
                # outcome (return or the outer cancellation) is preserved.
                break


async def _anext(stream: "AsyncIterator[LogEntry]") -> LogEntry:
    """Pull the next entry from ``stream`` (a typed ``anext`` wrapper for tasks).

    Args:
        stream: The ``observe`` async iterator to advance.

    Returns:
        The next ``LogEntry``.
    """
    return await stream.__anext__()
