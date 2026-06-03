# CLAUDE.md — vella-graph

The graph projection (v0.1): a read-only traversal view over the published
`vella.runtime` contract. The runtime is *physics* (log + store + write verbs);
the graph is a *read-only projection* that folds `observe()` into an always-built,
type-partitioned, bidirectional adjacency index and answers graph/traversal
queries from memory. It owns no storage and performs no writes — the runtime is
the sole authority. Inherits the monorepo CLAUDE.md; this adds graph specifics.

## Scope & deps
- `vella-graph` only. Depends on `vella-runtime` and `vella-core`, **never** the
  reverse. Both lower layers are unaware of this one.
- Published deps = `pydantic + typing_extensions + vella-core + vella-runtime`
  only (asserted by `tests/test_deps.py`, exact spellings). 0.x is lockstep: pin
  `vella-core==0.1.*` and `vella-runtime==0.1.*`.
- **Import allow-list (asserted by `tests/test_import_boundary.py`):** only the 7
  symbols in `vella.runtime.__all__` — `ConcurrencyConflict`, `Cursor`,
  `LogEntry`, `Runtime`, `Store`, `StoreTxn`, `TransitionKind`. Never a private
  `vella.runtime._*` symbol. From `vella.core`, only its public surface.

## Invariants
- **The index is forced; read-through is impossible.** The runtime exposes only
  `get`/`history`/`observe` — no list/scan. Every query is answered from an
  in-memory adjacency index folded from `observe()`, never read-through.
- **Determinism is a property, not a hope.** Every query returns `sorted()` ids;
  the gated determinism artifact is topology-derived and byte-identical across
  `PYTHONHASHSEED {0,1,42}` (from M3). Any set-derived serialized value is
  `sorted()`.
- **TRAP-1 (payload-independence).** Fold on typed top-level `LogEntry` fields
  only; `get()` for authority; never depend on `.payload` shape.
- **TRAP-2 (bounded drain).** Drain past `observe()`'s blocking live edge with a
  bounded drain that stops at the live edge; never `async for`-to-completion.
- **Cursor is opaque and unordered.** The view's high-water resume token is an
  opaque `Cursor` stored verbatim and passed back to `observe(since=)`; any "how
  far" comparison uses an internal monotonic int, never `cursor.token` (`Cursor`
  has no `__lt__`).
- **Mode is residency, not results.** `MaterializationMode(full|lean)` controls
  only body source; the same query engine runs over the same sorted topology in
  both modes, so topology results are byte-identical across modes.
- **No `pytest-asyncio`.** Async tests drive the follower manually (`asyncio.run`
  / a direct `run_until_complete`) under an injected `ManualClock`.

## Gate (before every commit; mirrors CI; run from `packages/vella-graph/`)
`pytest -q` (incl. doctests + any Hypothesis invariants) · `mypy` · `pyright` ·
`ruff check src/vella/graph` ·
`interrogate -c pyproject.toml src/vella/graph` · `mkdocs build --strict` ·
`python scripts/export_graph_surface.py --check` (public-surface
breaking-change tripwire: snapshots `__all__` + exported error MROs + model
field-types + `Literal` values + public method signatures; fails closed).
`filterwarnings = ["error::UserWarning"]` is load-bearing — a leaked async
generator or un-cancelled task surfaces as a `UserWarning` and turns the gate red.
Local commits only.
