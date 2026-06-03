"""The controller: the watch / worker / resync driver (M5).

:class:`Reconciler` is the three-task control loop (decision A1) — a watch/fold
task draining ``runtime.observe(since)`` into the work-set and dedup queue, a
single-flight worker task draining the queue, and a resync ticker on the injected
:class:`~vella.reconciler.clock.Clock`. Idle is an explicit, race-free predicate
(decision B1), ``step()`` is non-blocking, and ``run(max_steps=...)`` bounds worker
iterations.

Coordination contract (the precise, testable semantics)
--------------------------------------------------------
**Three-task split (2a).** :meth:`run` starts three coroutines: the watch/fold
task (drains :meth:`~vella.runtime.Runtime.observe` into the work-set — folding the
backlog via :func:`~vella.reconciler.workset.fold_available`, which sets the
backlog-drained Event, then continuing to fold live entries); the worker task
(drains the queue one :meth:`step` at a time — SINGLE-FLIGHT); and the resync
ticker (``await clock.sleep(resync_interval)`` → re-enqueues still-drifting
work-set keys, SKIPPING dead-lettered and in-flight keys). ``run(max_steps=N)``
bounds WORKER iterations only.

**Idle predicate (2b).** ``idle ≡ queue empty ∧ no known drift ∧ watch caught up to
the live edge``. "Caught up" is the EXPLICIT backlog-drained Event from the fold
(M3), never "the observe for-loop ended" (it never ends — ``observe`` blocks on the
live edge). "No known drift" means the worker is not mid-dispatch (single-flight: an
in-flight key has been popped but not yet resolved). The driver evaluates idle as a
predicate; :meth:`run` early-returns the moment it holds.

**Non-blocking ``step()`` (2c).** :meth:`step` pops at most one key via
``pop()``; on an empty queue it returns the :data:`IDLE` sentinel without blocking.
The only thing that blocks on the live edge is the watch task.

**Teardown (2d).** The watch task OWNS the ``observe`` generator: it opens, iterates,
and ``aclose``s it inside its own frame (via :func:`contextlib.aclosing`), so the
generator is always closed by the single task that touches it — even when that task
is cancelled. :meth:`run`'s ``finally`` only cancels the watch + resync tasks and
awaits them to ``done()`` (never calling ``aclose()`` cross-task), and that await is
cancellation-robust: if ``run`` itself is cancelled mid-flight the helpers are still
awaited to completion. So the ``filterwarnings=error::UserWarning`` gate sees zero
leaked generators / un-cancelled tasks, and — because no other task ever ``aclose``s
the generator — no ``RuntimeError: aclose(): asynchronous generator is already
running`` can arise from a cross-task close racing the watch task's ``anext``.

**Worker dispatch (must-fix 6).** Pop a key → FRESH ``runtime.get`` (the freshness
contract; the folded version is stale). ``get`` is ``None`` (deleted/gone) → DROP,
never raise. Not an :class:`~vella.core.Actuator` with a non-``None`` ``desired``,
or ``current == desired`` (compared via ``model_dump(mode="json")``) → not drifting,
clear. Drifting + a registered handler → build a :class:`Context`, dispatch by
``got.type``. A :class:`~vella.runtime.ConcurrencyConflict` from the handler's write
is expected contention → immediate re-read + re-invoke, NO backoff, bounded by
:data:`_MAX_IMMEDIATE_RETRIES`. A handler raising (other) or returning
``requeue(after)`` → capped exponential backoff (``min(base*2**n, cap)``) or the
explicit ``after``, per-key attempt tracking, re-enqueue after ``clock.sleep``.
After ``max_attempts`` → record to the dead-letter store AND emit exactly one
``observe_only`` ``reconcile_giveup`` telemetry entry (skipped by the fold, so it
cannot re-enqueue). ``done`` clears drift; ``drop`` discards without dead-lettering.

**Resync vs. in-flight (resolved open question).** A resync tick re-enqueues a
still-drifting key only if it is neither dead-lettered nor currently in-flight in
the single-flight worker. The dedup guard already covers a key that is queued; a
popped (in-flight) key has left the guard, so resync explicitly skips the in-flight
key (tracked in :attr:`_inflight`) to avoid double-processing. Even if a converged
key were re-enqueued, the worker's fresh-get drift recheck makes the re-process a
safe no-op — but skipping in-flight keys keeps the single-flight invariant exact.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
from typing import Any, Optional, cast

from vella.core import Actuator
from vella.runtime import ConcurrencyConflict, Cursor, LogEntry, Runtime

from .clock import Clock, SystemClock
from .context import Context
from .cursor_store import CursorStore
from .deadletter import DeadLetterRecord, DeadLetterStore
from .registry import Registry
from .result import ReconcileResult
from .workset import WorkKey, WorkSet, fold_available


class _Idle(enum.Enum):
    """Singleton sentinel type for :data:`IDLE` (a distinct, well-typed value)."""

    IDLE = enum.auto()


IDLE = _Idle.IDLE
"""Sentinel :meth:`Reconciler.step` returns when the queue is empty (non-blocking)."""

# Bound on immediate (no-backoff) retries after a ConcurrencyConflict. Contention
# is expected and resolved by re-reading the fresh version and re-invoking the
# handler; this caps the loop so pathological, unwinnable contention eventually
# falls through to the normal backoff/give-up path rather than spinning forever.
_MAX_IMMEDIATE_RETRIES = 5


class Reconciler:
    """Watch/worker/resync controller over the runtime contract.

    The controller observes the log, computes drift against fresh ``get`` reads,
    and drives convergent actions through the runtime's write verbs. It owns no
    storage and no clock; both are injected.
    """

    def __init__(
        self,
        runtime: Runtime,
        registry: Registry,
        clock: Optional[Clock] = None,
        cursor_store: Optional[CursorStore] = None,
        deadletter_store: Optional[DeadLetterStore] = None,
        *,
        resync_interval: float = 30.0,
        backoff: float = 1.0,
        backoff_cap: float = 60.0,
        max_attempts: int = 5,
    ) -> None:
        """Wire the controller to its injected runtime, registry, and seams.

        Args:
            runtime: The runtime contract the loop reads and writes through.
            registry: The entity-kind -> handler map the worker dispatches with.
            clock: The injected time source for resync ticks and backoff. Defaults
                to a :class:`~vella.reconciler.clock.SystemClock`.
            cursor_store: Persists the resume cursor handed to ``observe``. Optional
                — when ``None`` the watch task observes from the start each run.
            deadletter_store: Records keys the loop has given up on. Optional — when
                ``None`` give-up records only emit telemetry.
            resync_interval: Seconds of clock time between resync ticks.
            backoff: The base backoff (seconds) for the capped exponential schedule.
            backoff_cap: The maximum backoff delay (seconds) per attempt.
            max_attempts: Attempts before a key is dead-lettered and given up on.
        """
        self._runtime = runtime
        self._registry = registry
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._cursor_store = cursor_store
        self._deadletter_store = deadletter_store
        self._resync_interval = resync_interval
        self._backoff = backoff
        self._backoff_cap = backoff_cap
        self._max_attempts = max_attempts

        # The fold (work-set + dedup queue + high-water + backlog-drained Event).
        self._workset = WorkSet()
        # Per-key attempt counter, used for the backoff schedule and give-up bound.
        self._attempts: dict[WorkKey, int] = {}
        # The single in-flight key, if the worker is mid-dispatch. Resync skips it.
        self._inflight: Optional[WorkKey] = None

    # -- idle predicate (2b) -------------------------------------------------
    def is_idle(self) -> bool:
        """Return whether the loop has gone quiet (the race-free idle predicate).

        ``idle ≡ queue empty ∧ no known drift ∧ watch caught up to the live edge``.
        "Caught up" is the EXPLICIT backlog-drained Event from the fold (never
        inferred from a loop that happened not to fire); "no known drift" means no
        key is currently in-flight in the single-flight worker.

        Returns:
            ``True`` once the queue is empty, nothing is in-flight, and the watch
            task has reached the live edge; ``False`` otherwise.
        """
        return (
            self._workset.queue_depth() == 0
            and self._inflight is None
            and self._workset.backlog_drained.is_set()
        )

    # -- non-blocking worker step (2c) ---------------------------------------
    async def step(self) -> Any:
        """Process at most one queued work item without blocking.

        Pops one key from the dedup queue; on an empty queue returns the
        :data:`IDLE` sentinel without blocking. A popped key is dispatched per the
        worker semantics (fresh ``get`` → drift gate → handler → policy). The key is
        marked in-flight for the duration so a concurrent resync tick skips it.

        Returns:
            The :data:`IDLE` sentinel if the queue was empty, otherwise the
            :data:`~vella.reconciler.workset.WorkKey` that was processed.
        """
        key = self._workset.pop()
        if key is None:
            return IDLE
        self._inflight = key
        try:
            await self._dispatch(key)
        finally:
            self._inflight = None
        return key

    async def _dispatch(self, key: WorkKey) -> None:
        """Dispatch one reconcile pass for ``key`` (the worker's core, single-flight).

        Performs the fresh-get freshness read, the drift gate, the handler lookup,
        and applies the resulting policy (done / drop / requeue / conflict / error).

        Args:
            key: The ``(tenant_id, entity_id)`` to reconcile.
        """
        tenant_id, entity_id = key
        # FRESHNESS: read the live entity at dispatch, not the folded version.
        got = await self._runtime.get(tenant_id, entity_id)
        if got is None:
            # get()-race-loss: deleted/gone between fold and dispatch -> DROP.
            self._clear(key)
            return

        if not self._is_drifting(got):
            # Converged (or not an actuator with a target) -> clear drift.
            self._clear(key)
            return

        handler = self._registry.lookup(got.type)
        if handler is None:
            # Unregistered kind: explicit miss, skip (never crash).
            self._clear(key)
            return

        await self._invoke(key, handler)

    async def _invoke(self, key: WorkKey, handler: Any) -> None:
        """Invoke ``handler`` for ``key``, retrying contention and applying policy.

        A :class:`~vella.runtime.ConcurrencyConflict` raised by the handler's write
        is expected contention: re-read and re-invoke immediately (no backoff),
        bounded by :data:`_MAX_IMMEDIATE_RETRIES`. Any other exception, or a
        ``requeue`` verdict, routes to the backoff/give-up path. ``done`` clears
        drift; ``drop`` discards without dead-lettering.

        Args:
            key: The key being reconciled.
            handler: The async reconcile handler for the entity's kind.
        """
        ctx = Context(self._runtime, self._clock)
        for _ in range(_MAX_IMMEDIATE_RETRIES + 1):
            try:
                result: ReconcileResult = await handler(ctx)
            except ConcurrencyConflict:
                # Expected contention: another writer won the version race. Re-read
                # happens implicitly on the next handler call (it does a fresh get).
                continue
            except Exception as exc:  # noqa: BLE001 - policy: any error -> backoff.
                await self._backoff_or_giveup(key, reason=repr(exc))
                return
            # A verdict came back without a conflict: apply the policy and stop.
            if result.kind == "done":
                self._clear(key)
            elif result.kind == "drop":
                self._clear(key)
            else:  # "requeue"
                await self._backoff_or_giveup(
                    key, reason="handler requested requeue", after=result.after
                )
            return
        # Exhausted the immediate-retry budget on pure contention: treat as an
        # error and fall through to the normal backoff/give-up path.
        await self._backoff_or_giveup(
            key, reason="persistent ConcurrencyConflict (contention)"
        )

    async def _backoff_or_giveup(
        self, key: WorkKey, *, reason: str, after: Optional[float] = None
    ) -> None:
        """Re-enqueue ``key`` after a delay, or give up once attempts are exhausted.

        Bumps the per-key attempt counter. If it reaches ``max_attempts`` the key is
        dead-lettered and exactly one ``reconcile_giveup`` telemetry entry is
        emitted; otherwise the worker sleeps the explicit ``after`` (or the capped
        exponential ``min(base * 2**n, cap)``) on the injected clock and re-enqueues.

        Args:
            key: The key to back off or give up on.
            reason: A human-readable reason recorded on give-up.
            after: An explicit requeue delay (seconds); ``None`` uses the schedule.
        """
        attempts = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempts
        if attempts >= self._max_attempts:
            await self._giveup(key, reason=reason, attempts=attempts)
            return
        if after is not None:
            delay = after
        else:
            # Capped exponential: base * 2**(attempts-1), clamped to the cap.
            delay = min(self._backoff * (2 ** (attempts - 1)), self._backoff_cap)
        await self._clock.sleep(delay)
        self._enqueue(key)

    async def _giveup(self, key: WorkKey, *, reason: str, attempts: int) -> None:
        """Dead-letter ``key`` and emit exactly one ``reconcile_giveup`` telemetry.

        Records the give-up in the dead-letter store (so resync skips the key) and
        emits a single ``observe_only`` telemetry entry. The fold skips
        ``observe_only`` entries, so the give-up emit cannot re-enqueue the key.

        Args:
            key: The key being given up on.
            reason: The final reason recorded.
            attempts: The number of attempts made before giving up.
        """
        tenant_id, entity_id = key
        if self._deadletter_store is not None:
            await self._deadletter_store.record(
                DeadLetterRecord(
                    tenant_id=tenant_id,
                    entity_id=entity_id,
                    reason=reason,
                    attempts=attempts,
                )
            )
        # Exactly one observe_only give-up telemetry. Do NOT assert its version:
        # a deleted entity reports version 0.
        await self._runtime.emit_telemetry(
            tenant_id,
            entity_id,
            {"event": "reconcile_giveup", "attempts": attempts, "error": reason},
        )
        # Clear drift bookkeeping; the key stays dead-lettered until drain().
        self._clear(key)

    def _is_drifting(self, got: Any) -> bool:
        """Return whether ``got`` is an actuator whose current diverges from desired.

        Drift is defined only for an :class:`~vella.core.Actuator` state with a
        non-``None`` ``desired``; comparison is via ``model_dump(mode="json")``,
        never Python ``==`` (core's ``_vella_registry`` PrivateAttr breaks ``==``).

        Args:
            got: The freshly read entity.

        Returns:
            ``True`` when the entity has actuator state with a target and the
            current state differs from the desired state.
        """
        if not isinstance(got.state, Actuator):
            return False
        # ``isinstance`` narrows the erased state to ``Actuator[Unknown]`` (the
        # TState is erased), whose ``current``/``desired`` are Unknown under pyright
        # --strict. Cast to ``Actuator[Any]`` so both ``model_dump`` calls type
        # cleanly; mypy infers the precise type and would call the cast redundant —
        # the two checkers disagree, so the cast carries a mypy-side ignore (exactly
        # the runtime's erased-generic idiom). Compare via the JSON dump, never
        # ``==`` (core's ``_vella_registry`` PrivateAttr breaks ``==``).
        actuator = cast("Actuator[Any]", got.state)  # type: ignore[redundant-cast]
        desired = actuator.desired
        if desired is None:
            return False
        current_dump: dict[str, Any] = actuator.current.model_dump(mode="json")
        desired_dump: dict[str, Any] = desired.model_dump(mode="json")
        return current_dump != desired_dump

    def _enqueue(self, key: WorkKey) -> None:
        """Re-enqueue ``key`` through the work-set's dedup guard.

        Args:
            key: The key to re-enqueue (deduped — a no-op if already pending).
        """
        self._workset.enqueue(key)

    def _clear(self, key: WorkKey) -> None:
        """Clear a key's retry bookkeeping after it converges, drops, or gives up.

        Args:
            key: The key whose attempt counter to reset.
        """
        self._attempts.pop(key, None)

    # -- the three-task run loop (2a) ----------------------------------------
    async def run(self, max_steps: Optional[int] = None) -> None:
        """Drive watch + worker + resync to convergence.

        Starts the watch/fold task, the resync ticker, and a worker loop that calls
        :meth:`step` until the loop is idle (early-return: the moment the idle
        predicate holds, rather than waiting for the next resync tick) or until
        ``max_steps`` worker iterations have run. The ``finally`` tears down the
        watch + resync tasks (the watch task closes the ``observe`` generator it owns
        as part of its own unwind — teardown never ``aclose``s it cross-task) (2d).

        Args:
            max_steps: Optional bound on worker iterations; ``None`` runs until idle.
        """
        since: Optional[Cursor] = (
            await self._cursor_store.load()
            if self._cursor_store is not None
            else None
        )
        # SINGLE-TASK GENERATOR OWNERSHIP (2d): the watch task opens, iterates, AND
        # closes the ``observe`` generator inside its own frame (see :meth:`_watch`).
        # ``run``/``_teardown`` never touch the generator — so even when ``run`` is
        # cancelled mid-flight, the only ``aclose()`` runs as part of the watch task's
        # own unwind, never cross-task, so "aclose(): already running" cannot occur.
        watch_task = asyncio.ensure_future(self._watch(since))
        resync_task = asyncio.ensure_future(self._resync_loop())
        try:
            steps = 0
            while max_steps is None or steps < max_steps:
                # Let the watch task make progress folding the backlog / live edge.
                await asyncio.sleep(0)
                result = await self.step()
                if result is IDLE:
                    # Early-return at idle: queue empty, nothing in-flight, caught up.
                    if self.is_idle():
                        return
                    # Not yet caught up (backlog still draining): yield and re-poll.
                    await asyncio.sleep(0)
                    continue
                steps += 1
        finally:
            await self._teardown(watch_task, resync_task)

    async def _watch(self, since: Optional[Cursor]) -> None:
        """Open, fold (setting the caught-up Event), then fold live entries — and close.

        OWNS the ``observe`` generator end to end: opens ``observe(since)`` inside an
        :func:`contextlib.aclosing` block so the generator is ``aclose``d by THIS
        task's own frame unwinding — whether the fold completes, raises, or this task
        is cancelled. No other task ever calls ``aclose()`` on it, which is what makes
        the teardown free of the cross-task "aclose(): already running" race (2d).

        Drives :func:`~vella.reconciler.workset.fold_available` to consume the known
        backlog and set the backlog-drained Event, then continues pulling live
        entries off the generator, folding each and persisting the resume cursor.
        Blocks on the live edge (the only task that does).

        Args:
            since: The resume cursor handed to ``observe`` (``None`` from the start).
        """
        async with contextlib.aclosing(self._runtime.observe(since=since)) as stream:
            await fold_available(self._workset, stream)
            async for entry in stream:
                self._workset.apply(entry)
                if self._cursor_store is not None:
                    await self._cursor_store.save(entry.cursor)

    async def _resync_loop(self) -> None:
        """Periodically re-enqueue still-drifting keys (skipping dead-letter/in-flight).

        Sleeps ``resync_interval`` on the injected clock, then re-enqueues every
        work-set key that is still known (drift is rechecked at dispatch via a fresh
        ``get``), SKIPPING keys that are dead-lettered (so a deliberate give-up is
        not defeated) or currently in-flight in the single-flight worker (so a key
        being processed is never double-enqueued). Loops until cancelled.
        """
        while True:
            await self._clock.sleep(self._resync_interval)
            await self._resync_once()

    async def _resync_once(self) -> None:
        """Run a single resync pass: re-enqueue eligible still-known keys.

        A key is eligible if it is not currently in-flight and not dead-lettered.
        Keys are visited in sorted order so the re-enqueue sequence is deterministic
        under the manual clock.
        """
        for key in sorted(self._workset.keys(), key=lambda k: (k[0], str(k[1]))):
            if key == self._inflight:
                continue
            if self._deadletter_store is not None:
                tenant_id, entity_id = key
                if await self._deadletter_store.get(tenant_id, entity_id) is not None:
                    continue
            self._enqueue(key)

    async def _teardown(
        self, watch_task: "asyncio.Future[None]",
        resync_task: "asyncio.Future[None]",
    ) -> None:
        """Cancel the watch + resync tasks and await them to ``done()`` (2d).

        Cancels both helper tasks and awaits each to completion. The ``observe``
        generator is NOT touched here — the watch task ``aclose``s it inside its own
        frame (see :meth:`_watch`), so teardown never calls ``aclose()`` cross-task.

        The await is made cancellation-robust: if ``run`` itself is being cancelled,
        the :class:`asyncio.CancelledError` raised at our own ``await`` must NOT leave
        a helper task pending (a leaked task surfaces as a ``UserWarning`` and turns
        the gate red). So each task is re-awaited in a loop until it reports
        ``done()``, swallowing both the helper's own ``CancelledError`` and any
        ``CancelledError`` thrown into us mid-await. This guarantees that after
        ``run`` returns OR is cancelled, every helper task is ``done()`` and the
        generator has been closed by the watch task's unwind — with no RuntimeError
        and no leaked-task / un-awaited-generator warning.

        Args:
            watch_task: The watch/fold task to cancel.
            resync_task: The resync ticker task to cancel.
        """
        # Await each helper to ``done()``, robust to ``run`` being cancelled mid-await:
        # a ``CancelledError`` re-thrown into us must not abandon a still-pending task.
        # We RE-``cancel()`` on each loop turn: a helper can transiently swallow a
        # single cancellation (the fold's live-edge ``await fetch`` catches and re-
        # parks in the live ``async for``), so a one-shot cancel is not guaranteed to
        # terminate it — re-cancelling each turn drives it to ``done()`` for certain.
        for task in (watch_task, resync_task):
            while not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    # Either the helper's own cancellation completing, OR an outer
                    # cancellation re-thrown into this frame. If the task is now done
                    # the loop exits; otherwise we re-cancel + re-await so it cannot
                    # leak (and the generator it owns stays unclosed).
                    if task.done():
                        break
                    continue
                except Exception:  # noqa: BLE001 - the helper's terminal error is
                    # surfaced by awaiting it; teardown swallows it so the original
                    # ``run`` outcome (return or the outer cancellation) is preserved.
                    break

    # -- explicit re-entry for dead-lettered keys ----------------------------
    async def drain(self) -> None:
        """Re-enqueue dead-lettered keys for an explicit retry pass.

        The documented re-entry path for keys the loop gave up on: drains the
        dead-letter store and re-enqueues every drained key (resetting its attempt
        counter), so the next worker pass reconciles it afresh. A no-op when no
        dead-letter store is wired.
        """
        if self._deadletter_store is None:
            return
        drained = await self._deadletter_store.drain()
        for record in drained:
            key: WorkKey = (record.tenant_id, record.entity_id)
            self._attempts.pop(key, None)
            self._enqueue(key)

    # -- synchronous fold seam for tests -------------------------------------
    def _fold(self, entry: LogEntry) -> Optional[WorkKey]:
        """Fold one ``LogEntry`` directly into the work-set (a synchronous test seam).

        Mirrors what the watch task does per entry, without the live-edge blocking —
        tests drive this plus :meth:`step` to exercise the worker deterministically.

        Args:
            entry: The log entry to fold.

        Returns:
            The newly-enqueued :data:`~vella.reconciler.workset.WorkKey`, or ``None``
            if the entry was non-state-changing or deduped.
        """
        return self._workset.apply(entry)

    def _mark_caught_up(self) -> None:
        """Mark the watch task caught up to the live edge (a synchronous test seam).

        Sets the backlog-drained Event directly so tests can exercise the idle
        predicate without running the async watch task.
        """
        self._workset.mark_backlog_drained()
