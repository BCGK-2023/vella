# CLAUDE.md — vella-reconciler

The reconciliation loop (v0.1): a controller-runtime over the published
`vella.runtime` contract. The runtime is *physics* (log + store + write verbs);
the reconciler is a *control loop* that observes the log, computes drift (desired
vs. current), and drives convergent actions back through the runtime's write
verbs. It owns no storage and no clock of its own — both are injected. Inherits
the monorepo CLAUDE.md; this adds reconciler specifics.

## Scope & deps
- `vella-reconciler` only. Depends on `vella-runtime` and `vella-core`, **never**
  the reverse. Both lower layers are unaware of this one.
- Published deps = `pydantic + typing_extensions + vella-core + vella-runtime`
  only (asserted by `tests/test_deps.py`, exact spellings). 0.x is lockstep: pin
  `vella-core==0.1.*` and `vella-runtime==0.1.*`.
- **Import allow-list (asserted by `tests/test_import_boundary.py`):** only the 7
  symbols in `vella.runtime.__all__` — `ConcurrencyConflict`, `Cursor`,
  `LogEntry`, `Runtime`, `Store`, `StoreTxn`, `TransitionKind`. Never a private
  `vella.runtime._*` symbol. From `vella.core`, only its public surface.

## Invariants
- **The runtime is physics; the reconciler is a control loop.** It never persists
  world state; it reconciles toward desired state through the runtime's verbs.
- **Determinism is a property, not a hope.** Every ordering — resync ticks,
  backoff wakes, dead-letter serialization — is explicit and reproducible under
  the injected `ManualClock`. Any set-derived serialized value is `sorted()`.
- **Idle is observable.** Convergence is a race-free predicate the driver
  evaluates (queue empty ∧ no drift ∧ caught up to the live edge), never inferred
  from a loop that happened not to fire. `observe()` blocks at the live edge, so
  loop termination can never stand in for idle.
- **High-water is an integer.** "Caught up to the live edge" uses the fold-side
  monotonic entry counter, NEVER `Cursor`/`cursor.token` (`Cursor` has no
  ordering; the in-memory token is `str(offset)`, so lexical compare breaks past
  9). The resume `Cursor` is stored verbatim in the `CursorStore`, never compared.
- **Non-blocking `step()`.** Only the watch task blocks on the live edge; `step()`
  processes ≤1 queued item and returns an idle sentinel on an empty queue.
- **No `pytest-asyncio`.** Async tests drive the loop manually (`asyncio.run` / a
  tiny `run(coro)` helper) under an injected `ManualClock`. Mirror
  `packages/vella-runtime/tests/test_observe.py`.
- **`ManualClock` is PUBLIC** — a supported testing seam, part of `__all__` and
  the surface tripwire baseline.

## Gate (before every commit; mirrors CI; run from `packages/vella-reconciler/`)
`pytest -q` (incl. doctests + any Hypothesis invariants) · `mypy` · `pyright` ·
`ruff check src/vella/reconciler` ·
`interrogate -c pyproject.toml src/vella/reconciler` · `mkdocs build --strict` ·
`python scripts/export_reconciler_surface.py --check` (public-surface
breaking-change tripwire: snapshots `__all__` + exported error MROs + model
field-types + `Literal` values; fails closed). `filterwarnings =
["error::UserWarning"]` is load-bearing — a leaked async generator or
un-cancelled task surfaces as a `UserWarning` and turns the gate red.
