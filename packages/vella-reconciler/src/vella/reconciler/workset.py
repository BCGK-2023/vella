"""The work-set fold (internal mechanism, M3).

The watch task drains ``runtime.observe(since)`` into this fold. The fold is the
reconciler's working memory between the log and the worker: it answers "which keys
need a fresh look, and have we caught up to the live edge yet?" — never "what does
the entity look like." That second question is resolved later, at dispatch (M5),
via a fresh ``runtime.get()``; the fold deliberately never reconstructs entity
state.

What the fold maintains
-----------------------
* **Work-set** — a map ``(tenant_id, entity_id) -> last-seen version``. The version
  is a wake-up / staleness index, NOT reconstructed state. Per TRAP-1 the fold
  reads ONLY :class:`~vella.runtime.LogEntry`'s typed top-level fields
  (``tenant_id``, ``entity_id``, ``version``, ``transition``); it NEVER touches
  ``entry.payload`` and never calls ``runtime.get()``.
* **Dedup work-queue** — a FIFO queue of keys plus a companion guard ``set`` (the
  B1 / TRAP-3 design). A key is enqueued at most once while pending: an entity
  edited N times enqueues ONCE. The M5 worker pops a key and clears it from the
  guard so a later edit can re-enqueue it.
* **Monotonic high-water (int, B1)** — a counter incremented by exactly one per
  drained ``LogEntry``, INCLUDING ``observe_only``. This is the only "how far have
  we read" measure the reconciler compares; comparisons are integer. It is NEVER a
  :class:`~vella.runtime.Cursor` or ``cursor.token`` (``Cursor`` carries no
  ordering, and the in-memory token is ``str(offset)`` so a lexical compare breaks
  past offset 9). The resume ``Cursor`` lives verbatim in the ``CursorStore``.
* **Backlog-drained signal** — an :class:`asyncio.Event` the watch task sets once
  the known backlog has been folded (the "caught up to the live edge" signal M5's
  idle predicate consumes). See the contract below.

Transition handling
--------------------
The fold SKIPS ``observe_only`` (telemetry) and any non-state-changing transition:
such an entry still advances the high-water (it WAS drained from the stream) but
does NOT enter the work-set and does NOT enqueue a key. This is the
"no self-re-enqueue from the reconciler's own give-up emit" guarantee — the worker
emitting telemetry on give-up must not feed itself a new work item.

State-changing transitions (``create`` / ``edit`` / ``set_desired`` / ``upsert`` /
``delete`` / ``link`` / ``unlink``) upsert the key's version and enqueue the key
(deduped). ``delete`` is NOT special-cased here: the fold records the key like any
state-changing entry, and the DROP-on-``get()``-is-``None`` happens at dispatch
(M5), never in the fold.

The backlog-drained contract (M5 depends on this — stated precisely)
--------------------------------------------------------------------
``runtime.observe()`` is catch-up-then-live and BLOCKS at the live edge (the
in-memory store parks on ``await queue.get()``), so "caught up" can NEVER be "the
``async for`` ended" — it never ends. The fold instead defines:

    The known backlog is "every entry already appended to the log at the moment
    the watch task starts draining." The watch task folds those entries via
    :func:`fold_available`, which pulls entries that are immediately available
    WITHOUT blocking on the live edge, then sets :attr:`WorkSet.backlog_drained`
    the first time a pull would have to block (i.e. the live edge is reached).

Concretely, :func:`fold_available` probes the stream with a zero-delay
``anext`` race: it schedules ``anext(it)`` and a single event-loop yield; if the
yield wins, no entry was immediately available, the live edge is reached, and the
Event is set. Every entry that arrives before the live edge is folded and counted
in the high-water first, so when the Event fires the high-water already reflects
the entire known backlog. M5's idle predicate may therefore treat
``backlog_drained.is_set()`` as "the watch task has caught up to the live edge."

This keeps the fold itself synchronous and pure (:meth:`WorkSet.apply` ->
``Optional[WorkKey]``); only the draining/Event-setting is async, and it lives in a
small helper the M5 watch task drives. The seam is clean: M5 may use
:func:`fold_available` as-is, or drive :meth:`WorkSet.apply` directly and call
:meth:`WorkSet.mark_backlog_drained` on its own schedule.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import AsyncIterator, Optional
from uuid import UUID

from vella.runtime import LogEntry, TransitionKind

# A work-set / queue key: one entity within one tenant.
WorkKey = tuple[str, UUID]

# Transitions that do NOT change state: drained + counted, but never enqueued.
# ``observe_only`` is the telemetry channel (the reconciler's own give-up emit
# rides this), so skipping it is the no-self-re-enqueue guarantee.
_NON_STATE_CHANGING: frozenset[TransitionKind] = frozenset({"observe_only"})


class WorkSet:
    """The fold's working memory: work-set, dedup queue, high-water, idle signal.

    Construct one per watch task. :meth:`apply` is the synchronous, pure-ish fold
    step (one ``LogEntry`` in, an optional newly-enqueued :data:`WorkKey` out);
    the async draining that sets :attr:`backlog_drained` lives in
    :func:`fold_available`. Nothing here is added to ``vella.reconciler.__all__``:
    the work-set is an internal mechanism, not public surface.
    """

    def __init__(self) -> None:
        """Create an empty fold positioned before the first entry."""
        # Work-set: last-seen version per key (a staleness index, never state).
        self._versions: dict[WorkKey, int] = {}
        # Dedup work-queue: a FIFO of keys + a companion guard set. A key is in
        # the guard iff it is currently pending (enqueued, not yet popped by the
        # worker). The worker clears it from the guard on pop so a later edit can
        # re-enqueue. Together they give O(1) dedup (TRAP-3).
        self._queue: deque[WorkKey] = deque()
        self._pending: set[WorkKey] = set()
        # Monotonic high-water: entries drained so far (int, B1). NEVER a Cursor.
        self._high_water: int = 0
        # The explicit "caught up to the live edge" signal for M5's idle predicate.
        self._backlog_drained: asyncio.Event = asyncio.Event()

    @property
    def high_water(self) -> int:
        """Return the monotonic count of entries drained so far.

        Returns:
            The number of ``LogEntry`` values folded via :meth:`apply`, including
            ``observe_only`` entries. Increases by exactly one per entry; compared
            only as an integer, never against a :class:`~vella.runtime.Cursor`.
        """
        return self._high_water

    @property
    def backlog_drained(self) -> asyncio.Event:
        """Return the "caught up to the live edge" event M5's idle predicate reads.

        Returns:
            An :class:`asyncio.Event` that is set once the known backlog has been
            folded (see the module docstring's backlog-drained contract). Unset
            until the watch task reaches the live edge for the first time.
        """
        return self._backlog_drained

    def mark_backlog_drained(self) -> None:
        """Signal that the known backlog has been folded (caught up to live edge).

        Idempotent: setting an already-set :class:`asyncio.Event` is a no-op. The
        watch task calls this once it has pulled every immediately-available entry;
        M5 may also call it directly when driving :meth:`apply` on its own schedule.
        """
        self._backlog_drained.set()

    def version(self, key: WorkKey) -> Optional[int]:
        """Return the last-seen version for ``key``, or ``None`` if never seen.

        Args:
            key: The ``(tenant_id, entity_id)`` to look up.

        Returns:
            The last folded ``version`` for the key, or ``None`` if no
            state-changing entry has been folded for it. This is a staleness index
            for the worker, never reconstructed entity state.
        """
        return self._versions.get(key)

    def pop(self) -> Optional[WorkKey]:
        """Pop the next pending key off the dedup queue, or ``None`` if empty.

        Clears the popped key from the guard set so a subsequent edit re-enqueues
        it. This is the worker's (M5) entry point; the fold itself only enqueues.

        Returns:
            The next ``(tenant_id, entity_id)`` to reconcile, or ``None`` when the
            queue is empty.
        """
        if not self._queue:
            return None
        key = self._queue.popleft()
        self._pending.discard(key)
        return key

    def queue_depth(self) -> int:
        """Return the number of keys currently pending in the dedup queue.

        Returns:
            The queue length. Useful for the M5 idle predicate (``queue empty``).
        """
        return len(self._queue)

    def keys(self) -> list[WorkKey]:
        """Return every key the fold has ever recorded in the work-set.

        The M5 resync ticker walks these to re-enqueue still-drifting keys (drift is
        rechecked at dispatch via a fresh ``get``). This is the work-set membership,
        independent of the dedup queue — a key converged-and-popped is still known
        here so resync can re-examine it. Iteration order is the fold's insertion
        order; the resync ticker sorts before re-enqueueing for determinism.

        Returns:
            The recorded ``(tenant_id, entity_id)`` keys.
        """
        return list(self._versions.keys())

    def enqueue(self, key: WorkKey) -> Optional[WorkKey]:
        """Re-enqueue a known ``key`` through the dedup guard (M5 resync/requeue seam).

        Unlike :meth:`apply` (which folds a fresh ``LogEntry``), this re-enqueues a
        key the worker has already seen — used by the M5 resync ticker and the
        backoff requeue path. Deduped against the guard set: a no-op if the key is
        already pending. The key must already be in the work-set (it was folded once
        to land here); this never invents a version.

        Args:
            key: The ``(tenant_id, entity_id)`` to re-enqueue.

        Returns:
            ``key`` if it was newly enqueued; ``None`` if it was already pending.
        """
        if key in self._pending:
            return None
        self._pending.add(key)
        self._queue.append(key)
        return key

    def apply(self, entry: LogEntry) -> Optional[WorkKey]:
        """Fold one ``LogEntry``: advance the high-water, maybe enqueue its key.

        Reads ONLY ``entry``'s typed top-level fields (``transition``,
        ``tenant_id``, ``entity_id``, ``version``) — NEVER ``entry.payload`` (TRAP-1)
        and never ``runtime.get()``. Every entry advances the high-water by exactly
        one. A non-state-changing entry (``observe_only``) stops there: it does not
        enter the work-set and does not enqueue (the self-re-enqueue guard). A
        state-changing entry upserts the key's version and enqueues the key, deduped
        against the guard set so N edits to one key enqueue ONCE.

        Args:
            entry: The log entry to fold.

        Returns:
            The :data:`WorkKey` if this call newly enqueued it; ``None`` if the
            entry was non-state-changing or the key was already pending (deduped).
        """
        # Every drained entry advances the high-water — observe_only included.
        self._high_water += 1

        if entry.transition in _NON_STATE_CHANGING:
            return None

        key: WorkKey = (entry.tenant_id, entry.entity_id)
        # Work-set upsert: record the latest version as a staleness index.
        self._versions[key] = entry.version

        # Dedup enqueue: only if not already pending (TRAP-3 / O(entities)).
        if key in self._pending:
            return None
        self._pending.add(key)
        self._queue.append(key)
        return key


async def fold_available(workset: WorkSet, stream: AsyncIterator[LogEntry]) -> None:
    """Fold every immediately-available entry, then mark the backlog drained.

    The watch-task helper that realizes the backlog-drained contract (see the
    module docstring). It pulls entries that are available WITHOUT blocking on the
    live edge — racing each ``anext(stream)`` against a single event-loop yield —
    folding each via :meth:`WorkSet.apply`. The first time the yield wins (no entry
    was immediately available: the live edge is reached) it calls
    :meth:`WorkSet.mark_backlog_drained` and returns, leaving the stream open for
    M5's live phase. Because every available entry is folded (and counted in the
    high-water) before the Event fires, the high-water reflects the entire known
    backlog the moment ``backlog_drained`` is set.

    Args:
        workset: The fold to drain into.
        stream: The ``runtime.observe(since)`` async iterator to drain.
    """
    while True:
        # ``fetch`` is a sub-task holding ``stream.__anext__()`` in flight. It MUST
        # always reach ``done()`` before this frame unwinds — including when THIS
        # coroutine is itself cancelled (the watch task being torn down) at the bare
        # yield below, BEFORE we reach the explicit cancel path. A leaked ``fetch``
        # keeps the generator's ``__anext__`` running, which both leaks the task and
        # makes the only safe ``aclose()`` (in the watch task's
        # :func:`contextlib.aclosing` frame) race a still-running generator — the M6
        # "aclose(): already running" defect. So the ``fetch`` lifecycle is wrapped
        # in a ``finally`` that drives it to ``done()`` for certain.
        fetch: asyncio.Task[LogEntry] = asyncio.ensure_future(_anext(stream))
        try:
            # A bare yield: if `fetch` already has an entry buffered it resolves on
            # this turn; otherwise `fetch` parks on the live edge and the yield wins.
            await asyncio.sleep(0)
            if fetch.done():
                try:
                    entry = fetch.result()
                except StopAsyncIteration:
                    # Stream exhausted (e.g. a finite test stream): backlog is fully
                    # drained — mark caught-up and stop.
                    workset.mark_backlog_drained()
                    return
                workset.apply(entry)
                continue
            # The live edge: no entry was immediately available. M5 owns the live
            # phase, so the parked pull is cancelled (in the ``finally``) and we
            # signal caught-up and hand the stream back open.
            workset.mark_backlog_drained()
            return
        finally:
            # Drive ``fetch`` to ``done()`` unconditionally — on the normal live-edge
            # return AND on a CancelledError thrown into the ``await`` above. The
            # re-await is cancellation-robust: an outer cancellation re-thrown into us
            # must not abandon ``fetch`` still-pending. ``fetch`` is ``cancel()``-ed,
            # so this terminates promptly; the terminal
            # ``CancelledError``/``StopAsyncIteration`` is swallowed either way.
            fetch.cancel()
            while not fetch.done():
                try:
                    await fetch
                except (asyncio.CancelledError, StopAsyncIteration):
                    if fetch.done():
                        break


async def _anext(stream: AsyncIterator[LogEntry]) -> LogEntry:
    """Pull the next entry from ``stream`` (a typed ``anext`` wrapper for tasks).

    Args:
        stream: The async iterator to advance.

    Returns:
        The next ``LogEntry``.
    """
    return await stream.__anext__()
