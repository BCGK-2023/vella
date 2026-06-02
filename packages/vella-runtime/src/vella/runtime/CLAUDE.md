# CLAUDE.md — vella-runtime

The physics / state substrate (v0.1): the sole agent-agnostic entry point for
changing world state. CQRS write side — an append-only **log** plus a canonical
**state-table** that is a fold of it. Agents, APIs, humans, CLIs, and MCP servers
are all just *consumers* of one action contract; nothing mutates the world except
through here. Inherits the monorepo CLAUDE.md; this adds runtime specifics.

## Scope & deps
- `vella-runtime` only. Depends on `vella-core`, **never** the reverse.
- Published deps = `pydantic + typing_extensions + vella-core` only (asserted by
  `tests/test_deps.py`). 0.x is lockstep: pin `vella-core==0.1.*`.
- v0.1 is the **state substrate only**: Store + log + state-table + write verbs
  (`create`/`edit`/`set_desired`/`upsert`/`delete`/`link`/`unlink`) + reads
  (`get`/`history`/`observe`) + `emit_telemetry`. `run_tool` and the
  reconciliation loop (`vella.reconciler`) are out of scope.

## Invariants
- **Async-first.** All verbs are `async`; `observe()` is an async iterator
  (global by design — the stream every projection rebuilds from).
- **Atomicity.** Read-modify-write verbs run inside one `store.transaction()`
  scope. `edit`/`set_desired` use optimistic concurrency (`expected_version` →
  `ConcurrencyConflict`); `upsert` is idempotent via the transaction lock alone.
- **Stream/table duality.** `state-table == fold(log)`, provable two ways that
  must agree: in-memory typed replay via `hydrate` on model-instance payload, and
  whole-table canonical bytes. Compare entities via `model_dump(mode="json")`,
  **never** Python `==` (core's `_vella_registry` PrivateAttr breaks `==`).
- **`LogEntry.payload` holds model-instance fields**
  (`{k: getattr(e, k) for k in type(e).model_fields}`) — never
  `model_dump(mode="python")` (it dict-ifies nested models, breaking `hydrate`).
- **Never sort core model fields** (`integrations`, `contributes_to` are
  order-semantic). Sort only runtime's own set-derived serialized structures.
- `Cursor` is an opaque `{token: str}` — consumers pass it back to `observe`,
  never compare by value. Tombstones are runtime-side (`get` → `None`, retained in
  `history`); core has no delete concept.

## Gate (before every commit; mirrors CI; run from `packages/vella-runtime/`)
`pytest -q` (incl. doctests + Hypothesis fold/tenancy invariants) · `mypy` ·
`pyright` · `ruff check src/vella/runtime` ·
`interrogate -c pyproject.toml src/vella/runtime` · `mkdocs build --strict` ·
`python scripts/export_runtime_surface.py --check` (breaking-change tripwire:
snapshots the Store/StoreTxn Protocol + Runtime verb signatures + `Cursor`/
`LogEntry`/`TransitionKind`/`ConcurrencyConflict`; fails closed).
Determinism: serialized output is byte-identical across `PYTHONHASHSEED`
(`tests/test_determinism.py`, subprocess-based). The conformance suite
(`tests/conformance/store_suite.py`) is the contract every future Store adapter
runs unchanged.
