# DESIGN.md — vella-graph

The *why* behind the graph projection. The code is the spec; this document holds
the rationale. (Finalized at M6; this is the M1 ADR stub copied from the consensus
plan §6.)

## Why a projection, not a store

The runtime exposes only `get`/`history`/`observe` — no list/scan, no "edges
touching X". You cannot read-through to answer a neighbor query. So the index is
**forced**: every query is answered from an in-memory adjacency index folded from
`observe()`. The graph owns no storage and performs no writes; the runtime stays
the sole authority. This is the load-bearing constraint that shapes everything
below.

## ADR (stub — finalized in this file at M6)

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
