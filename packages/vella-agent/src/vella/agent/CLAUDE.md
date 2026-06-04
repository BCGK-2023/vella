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
- **Self-hosting purity.** The agent acts ONLY through the runtime's public verbs;
  cognition is nodes/edges and `observe_only` telemetry. No new `Node` subclass; no
  edit pushed down into core/runtime/graph. Every write in `_writeback.py` /
  `interpreter.py` goes through `create`/`link`/`edit`/`emit_telemetry` — there is no
  direct store access and no private `vella.*._*` import.
- **`observe_only` never bumps the version.** The token-level reasoning trace is
  emitted via `emit_telemetry` (locked decision #1): it reaches the log and live
  observers but never touches the run's state-table version, so the trace is
  replayable yet non-bloating. A trace written as an `edit` would bump the version —
  that is the mutation the `test_observe_only_trace` invariant catches.
- **The canonical turn is frozen.** `Message` / `AssistantTurn` / the `ContentBlock`
  union / `Usage` / `StopReason` and the `ModelProvider` streaming events are the
  contract every provider and the interpreter speak; the interpreter pattern-matches
  ONLY canonical shapes, never a provider-specific field (provider-agnostic). The
  streaming delta assembler keys strictly by integer `content_block` index and parses
  only at `content_block_stop`, so streaming ≡ non-streaming byte-for-byte.
- **Tools-are-nodes.** A tool is an `agent.tool` node (its `ToolData` declaration +
  discriminated `Binding`); discovery is a graph read over the `HAS_TOOL` edge plus an
  idempotent `upsert` baseline seed — no privileged toolset. `run_tool` lives in the
  agent layer (the `ToolInvoker` seam owns dispatch + capped retries via an injected
  `Clock`); the runtime stays pure. Dispatch routes strictly by `binding.kind`.
- **Loop-as-data.** Behavior is configured by a frozen `loop_policy` typed FSM schema
  (budgets, stop conditions, `tool_choice`, planning modes, sub-agents, compaction),
  interpreted — never hard-coded into control flow. `step_budget` is checked at the
  turn boundary BEFORE the next turn (N steps ⇒ ≤ N `agent.step` nodes); the HARD
  `token_budget` always wins over the SOFT `compaction_threshold` and is terminal.
- **Sub-agents are runaway-bounded.** `max_depth`/`max_fanout` are computed from the
  **graph** (the durable authority, never an in-memory counter) with explicit
  `direction="out"` for depth and `direction="in"` for fanout (never `"both"`), and
  checked **before** any child `agent.run` node is created. Runaway spawning is
  provably impossible: closed-form run-tree cardinality `Σ fⁱ`
  (`max_run_tree_size`) AND per-run token budgets (no pooling) bound aggregate cost.
- **Determinism is a property, not a hope.** The interpreter is network-free under its
  reference impls; the gated determinism artifacts (surface snapshot + loop-policy
  `model_dump`) are byte-identical across `PYTHONHASHSEED {0,1,42}` (subprocess test).
  Any set-derived serialized value (`stop_conditions`, restricted `types`, discovery
  order, sub-agent run-tree digest) is `sorted()` at the serialization boundary.
- **No `pytest-asyncio`.** Async tests drive the interpreter manually (`asyncio.run` +
  bounded `run(max_steps)`) under an injected `Clock`/`ManualClock`;
  `filterwarnings = ["error::UserWarning"]` turns a leaked generator/task red.
- **Fresh-`Registry` discipline.** The package registers its nine `agent.*` types ONCE
  at import (idempotent, stable names); `@node_type` requires a **frozen** data class
  and silently overwrites on re-registration, so every test/doctest constructs nodes
  through `agent_registry()` (a fresh `Registry()`), NEVER core's process-wide
  `default_registry` — the real defense against cross-test pollution.
- **Extras are quarantined.** The published deps are exactly the 5-tuple
  `{pydantic, typing_extensions, vella-core, vella-runtime, vella-graph}`
  (`test_deps.py`); `[openrouter]` (`httpx`) and `[mcp]` (`mcp`) adapters import lazily
  and are EXCLUDED from the core gate via the `extras` marker — importing the core
  never pulls heavy I/O, and `__all__` stays at 84 with the adapters out of it.

## Gate (before every commit; mirrors CI; run from `packages/vella-agent/`, venv active)
Fail-closed, in order — any non-zero exit is RED:
1. `pytest -q` — doctests (`README.md` + every public docstring) + Hypothesis
   invariants + the subprocess determinism tests. `addopts` pins
   `-m "not extras"` + `--doctest-glob=*.md --doctest-modules`, so the deterministic
   CORE gate never collects the `httpx`/`mcp` adapter smoke tests.
2. `mypy` (strict; `src` + `tests`) · `pyright` (strict; `src`).
3. `ruff check src/vella/agent` (pydocstyle Google convention).
4. `interrogate -c pyproject.toml src/vella/agent` — 100% docstring coverage over the
   public surface (a public symbol missing a docstring is RED).
5. `mkdocs build --strict` — the API reference renders from docstrings (`::: vella.agent`);
   any cross-ref/strict warning is RED.
6. `python scripts/export_agent_surface.py --check` — the public-surface
   breaking-change tripwire: snapshots `__all__` (**frozen at 84**) + exported error
   MROs + model field-types + `Literal` values + public method signatures; fails
   closed on any drift.
7. `tests/test_deps.py` (the 5-dep frozenset) · `tests/test_import_boundary.py`
   (AST-based: runtime+graph+core public `__all__` only; forbid `_*` and any
   `vella.reconciler` import) · the determinism subprocess tests across
   `PYTHONHASHSEED {0,1,42}`.

`filterwarnings = ["error::UserWarning"]` is load-bearing — a leaked async generator
or un-cancelled task surfaces as a `UserWarning` and turns the gate red. The
out-of-gate adapters are exercised SEPARATELY: install the extras
(`uv pip install -e ".[dev,openrouter,mcp]"`) and run `pytest -m extras`. Local
commits only.
