# CLAUDE.md — vella-agent

The cognition core (v0.1): a self-hosted, data-configured agent interpreter over
the published `vella.runtime` and `vella.graph` contracts. The runtime is *physics*
(log + store + write verbs); the graph is a *read-only projection* answering
traversal queries from memory; the agent is the *cognition core* that acts ONLY
through the runtime's verbs and perceives ONLY through the graph. It owns no storage
and takes no privileged path — an agent run, its steps, tool calls, messages, and
policy are ordinary registered core node types. Inherits the monorepo CLAUDE.md;
this adds agent specifics.

## Scope & deps
- `vella-agent` only. Depends on `vella-runtime`, `vella-graph`, and `vella-core`,
  **never** the reverse and **never** `vella-reconciler` (a sibling, not a
  dependency — the agent is off the reconciler's graph entirely). All three lower
  layers are unaware of this one.
- Published deps = `pydantic + typing_extensions + vella-core + vella-runtime +
  vella-graph` only (asserted by `tests/test_deps.py`, exact spellings). 0.x is
  lockstep: pin `vella-core==0.1.*`, `vella-runtime==0.1.*`, `vella-graph==0.1.*`.
  Heavy I/O transports live in extras only: `[openrouter]` (`httpx`) and `[mcp]`
  (`mcp`) — excluded from the published frozenset, so importing the core never
  pulls them.
- **Import allow-list (asserted by `tests/test_import_boundary.py`, AST-based):**
  from `vella.runtime`, only the 7 symbols in `vella.runtime.__all__`
  (`ConcurrencyConflict`, `Cursor`, `LogEntry`, `Runtime`, `Store`, `StoreTxn`,
  `TransitionKind`); from `vella.graph`, only the symbols in `vella.graph.__all__`;
  from `vella.core`, only its public surface. Never a private `vella.*._*` symbol.
  **Any `import vella.reconciler` is forbidden** — the check is AST-based and never
  executes the import, so the forbid holds even though `vella-reconciler` is not
  installed on this branch.

## Invariants
- **Self-hosting is the substrate, not a feature.** The agent acts only through the
  runtime's public verbs; cognition is nodes/edges and `observe_only` telemetry.
  No new `Node` subclass; no edit pushed down into core/runtime/graph.
- **Three Protocol seams.** `ModelProvider` / `ToolInvoker` / `ContextAssembler`
  each have a deterministic in-gate reference impl (MockProvider /
  InMemoryToolInvoker / GraphContextAssembler) and optional out-of-gate real
  adapters; heavy I/O never enters the gate.
- **Determinism is a property, not a hope.** The interpreter is network-free under
  its reference impls; the gated determinism artifact is byte-identical across
  `PYTHONHASHSEED {0,1,42}`. Any set-derived serialized value is `sorted()`.
- **Loop-as-data.** Behavior is configured by a frozen `loop_policy` typed FSM
  schema (budgets, stop conditions, tool gating, planning, sub-agents, compaction),
  interpreted — never hard-coded into control flow.
- **Sub-agents are bounded.** `max_depth`/`max_fanout` are computed from the
  **graph** (the durable authority, never an in-memory counter) and checked
  **before** any child `agent.run` node is created; runaway spawning is provably
  impossible (closed-form run-tree cardinality + per-run token budgets).
- **No `pytest-asyncio`.** Async tests drive the interpreter/followers manually
  (`asyncio.run` / bounded `run(max_steps)`) under an injected `Clock`/`ManualClock`.

## Gate (before every commit; mirrors CI; run from `packages/vella-agent/`)
`pytest -q` (incl. doctests + any Hypothesis invariants) · `mypy` · `pyright` ·
`ruff check src/vella/agent` ·
`interrogate -c pyproject.toml src/vella/agent` · `mkdocs build --strict` ·
`python scripts/export_agent_surface.py --check` (public-surface
breaking-change tripwire: snapshots `__all__` + exported error MROs + model
field-types + `Literal` values + public method signatures; fails closed).
`filterwarnings = ["error::UserWarning"]` is load-bearing — a leaked async
generator or un-cancelled task surfaces as a `UserWarning` and turns the gate red.
Local commits only.
