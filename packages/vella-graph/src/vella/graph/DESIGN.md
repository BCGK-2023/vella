# DESIGN.md — vella-graph

The *why* behind the graph projection. The code is the spec; this document holds
the rationale. Finalized at M6.

## Why a projection, not a store

The runtime exposes only `get`/`history`/`observe` — no list/scan, no "edges
touching X". You cannot read-through to answer a neighbor query. So the index is
**forced**: every query is answered from an in-memory adjacency index folded from
`observe()`. The graph owns no storage and performs no writes; the runtime stays
the sole authority. This is the load-bearing constraint that shapes everything
below.

## ADR (finalized at M6)

**Decision:** read-only projection folding `observe()` into an always-built,
type-partitioned, bidirectional adjacency index with `MaterializationMode(full |
lean)` residency, sorted-id deterministic queries, COW pull-refresh, and an opt-in
background follower porting reconciler lifecycle discipline.

**Drivers:**
1. No list/scan → the index is forced (read-through is impossible).
2. `Cursor` is unordered (no `__lt__`) → the resume token is opaque, stored
   verbatim, never compared; "how far" uses an internal monotonic int.
3. Mode is residency, not results → topology equivalence across modes is
   load-bearing; this forces a clean split between the mode-independent topology
   index and mode-dependent body hydration.

**Alternatives:** A2 (flat dict + frozenset — hash-seed-sensitive iteration,
sorting on the hot path); A3 (HAMT — explicit v0.1 non-goal); B2 (unbounded cache
— defeats lean); B3 (no cache — spec mandates LRU); C2 (string/Cypher DSL —
non-goal); C3 (callable predicate — breaks determinism, opaque to pruning); D-alt
(no baking / weight-per-query-only — forces body reads on every weighted SP,
defeats "spend memory to buy latency"); D2 (store whole body — forces residency in
lean, breaks the split). All rejected (see plan §1 rationales).

**Why chosen:** A1 (nested dict COW index) + B1 (bounded-LRU lean hydration) + C1
(anchored, type-pruned motif) + D1 (baked weights) maximizes expression while
keeping the gated deterministic core mode-independent; COW gives natural O(Δ)
refresh; LRU bounds lean; type-pruned motif scopes the matcher; baked weights keep
weighted SP pure in-memory.

**Consequences:** a weight change requires a re-fold; the per-query weight override
is full-mode only (lean raises a typed `WeightOverrideRequiresFullMode`, an
explicit equivalence exclusion); `Clock`/`ManualClock` are re-declared, not
imported, to respect the dependency direction; the M6 follower is independently
revertible.

**Follow-ups:** runtime batch-`get` (speeds cold fold; non-gating); topology-only
residency at scale beyond lean+LRU (v0.2); graph-level invariants / edge
cardinality (deferred by core; the index is the natural home); a shared `Clock`
conformance suite (needs a testing-utils package or accepted duplication).

## M6 follower: design notes & honest deviations

The opt-in `GraphFollower` is the *only* async-lifecycle surface; `fold` and
`refresh` stay one-shot coroutines needing no task and no `Clock`
(regression-guarded). Three M6 decisions deviate from the consensus plan; each is
recorded here rather than hidden, per the project's "report honestly" rule.

**1. Carried pull, not `bounded_drain`, for the live tail (forced).** `bounded_drain`
(used by `fold`/`refresh`) `cancel()`s its in-flight `__anext__` probe at each live
edge. For the in-memory runtime that cancellation *finalizes* the `observe()`
generator (its `finally` runs the observer-queue `discard`) — harmless for callers
that then reopen, but fatal for a long-lived follower that must keep the *same*
generator to see live entries. Verified empirically: after a `bounded_drain` pass
the same generator yields `StopAsyncIteration`. So the follower's watch CARRIES one
persistent `__anext__` pull across the live edge (resolving it on a bare `sleep(0)`
when an entry is buffered, blocking on it at the edge) and cancels it only at
teardown. Note this also means the reconciler's `fold_available` + live-`async for`
reuse pattern would not, in fact, track live entries over this runtime — a finding
worth carrying upstream.

**2. `Clock` is vestigial in the follower's loop (kept by spec).** The v0.1 loop is
purely event-driven on `observe()`: it blocks on the live tail and wakes per entry,
with no timer behaviour, so it never calls `clock.sleep`. The public
`Clock`/`ManualClock` surface is kept exactly as the plan specifies — it is the
injectable-time seam, the supported no-`pytest-asyncio` test driver, and the
structural Protocol-conformance proof — but no fabricated timer machinery was added
to manufacture a use. If a future follower grows genuine time-based behaviour
(debounced re-fold, periodic compaction) the seam is already in place.

**3. Direct-teardown non-vacuity is narrower than the plan assumed.** The DIRECT
teardown test runs on a pre-existing, never-closed loop and asserts (a) the observer
set returns to baseline and (b) every spawned task is `done()`. The plan expected
`mut-m6-vacuous-aclose` to turn (a) RED and `mut-m6-oneshot-cancel` to turn (b) RED.
Empirically neither does, for principled reasons: an in-memory `observe()` generator
runs its `finally` whenever its `__anext__` is finalized — by exhaustion OR by the
teardown cancellation of the carried pull — *not only* by `aclose()`; so (a) holds
even with a vacuous `aclosing` (the `aclosing` stays as correct, idiomatic, defensive
discipline, but its presence is unobservable via the observer set). And because the
watch blocks on a single carried `await fetch` (not a re-parking `async for`), a
single `task.cancel()` terminates it, making the re-cancel-to-done loop
belt-and-suspenders here (kept for robustness + reconciler parity). The teardown
discipline that IS load-bearing and proven non-vacuous: **single-task generator
ownership** — `mut-m6-cross-task-aclose` (aclosing the generator from the teardown
frame while the carried pull's `__anext__` is in flight) raises `RuntimeError:
aclose(): asynchronous generator is already running`, proven constructively by
`test_cross_task_aclose_raises_already_running`. The behavioural mutations remain
fully caught: `mut-m6-early-drain` → follower≠fresh-fold equivalence RED;
`mut-m6-no-quiescence-signal` → `caught_up` never set → bounded `wait_for` times out.
