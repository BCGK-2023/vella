# DESIGN.md — vella-agent

The *why* behind the cognition core. The code is the spec; this document holds the
rationale and the two pinned contracts the code freezes. Finalized at M8.

## Self-hosted cognition: the substrate, not a feature

`vella.runtime` is *physics* — an append-only log, an optimistic-concurrency store,
and the write verbs (`create`/`edit`/`set_desired`/`upsert`/`delete`/`link`/`unlink`/
`emit_telemetry`) that move world state forward. `vella.graph` is a *read-only
projection* that folds the runtime's `observe()` stream into an in-memory adjacency
index and answers traversal queries from memory. `vella.agent` is the *cognition
core*: a data-configured interpreter that ACTS only through the runtime's verbs and
PERCEIVES only through the graph.

The load-bearing decision is that an agent is not a privileged object with its own
storage — it is *self-hosted on the substrate*. An agent run, its steps, tool calls,
messages, summaries, its loop policy, its providers, and its tools are all ordinary
registered **core node types** (`@node_type` over frozen data classes). There is no
new `Node` subclass and no edit pushed down into core/runtime/graph. Cognition is
nodes/edges; the token-level reasoning trace is `observe_only` telemetry that reaches
the log and live observers but never bumps the run's state-table version. This is what
buys full replayability, observability, and multi-agent topology *for free*: a run is
a durable record you can fold from zero, a sub-agent tree is a `PART_OF` subgraph, and
"how far did this run get" is a graph query — never an in-memory counter (TRAP-1: the
authority is the graph/runtime, never a local mutable).

## Three act-modalities

The cognition core exposes three distinct ways for an agent to *act*, each landing on
the substrate differently:

1. **Imperative** — a tool call. The model emits a `ToolUseBlock`; the `ToolInvoker`
   seam dispatches it (by `binding.kind`), owns capped retries (via an injected
   `Clock`), and returns a single `ToolResult`. The loop never re-invokes — the
   invoker surfaces a final `is_error` only once its own retries are exhausted.
2. **Declarative** — desired state. The agent can write a *desired* projection via
   `runtime.set_desired` and let a reconciler (a SIBLING package, off this package's
   graph entirely) drive convergence. The agent NEVER imports `vella.reconciler`; it
   only writes desired state through the runtime's public verb. This keeps the
   declarative modality available without coupling cognition to controller-runtime.
3. **Cognition** — the loop itself. The FSM interpreter over a frozen `loop_policy`
   IS the third modality: assemble context → `provider.turn()` → record message +
   trace → invoke tools → evaluate stop/budgets → next turn or halt.

## Tools-are-nodes + the invoker/registry split

A tool is an `agent.tool` node: a `ToolData` payload carrying a core `ToolDeclaration`
(`name`/`description`/`parameters`/`returns`), a discriminated `Binding` union
(`BuiltinBinding` / `MCPBinding` / `HTTPBinding`, keyed by `kind`), `ToolHints`, and an
optional `RetryPolicy`. Discovery is a **graph read** over the custom `HAS_TOOL` edge
plus an idempotent `upsert` baseline "system" seed — there is no privileged toolset and
re-seeding is a no-op. (`HAS_TOOL` is a custom edge string, NOT a core `EdgeTypes`
constant; `Edge(type="has_tool")` emits zero warnings because it clears core's
`difflib` cutoff — a positive guard test pins this so a future core change that starts
warning is caught. No core edit is needed or permitted.)

The deliberate split is **declaration (a node) vs dispatch (an invoker)**. The node
declares *what* a tool is and how to bind it; the `ToolInvoker` seam decides *how* to
run it. `run_tool` therefore lives in the AGENT layer, not the runtime — the runtime
stays pure physics. The in-gate `InMemoryToolInvoker` is a deterministic registry
(`registry_key → callable`) with capped backoff routed through the injected `Clock`, so
the whole loop is network-free and replayable under a `ManualClock`.

## The three Protocol seams

Cognition is parameterized over three structural `Protocol` seams, each with a
deterministic **in-gate reference impl** and an optional **out-of-gate real adapter**,
so heavy I/O never enters the deterministic gate:

| Seam | In-gate ref impl (gated) | Out-of-gate adapter (extras) |
|---|---|---|
| `ModelProvider` | `MockProvider` (scriptable, fragmented streaming) | `OpenRouterProvider` (`[openrouter]`, `httpx`) |
| `ToolInvoker` | `InMemoryToolInvoker` (registry + `Clock` backoff) | MCP invoker (`[mcp]`, `mcp`) |
| `ContextAssembler` | `GraphContextAssembler` (graph recall + compaction) | — (future vectorstore assembler) |

The out-of-gate adapters translate a live dialect into the SAME canonical turn via the
SAME M2 streaming assembler (OpenRouter) or execute a real remote tool (MCP). They are
imported lazily and excluded from the published 5-dep frozenset, so importing the core
never pulls `httpx`/`mcp`.

## The two pinned contracts

### 2.1 `loop_policy` typed FSM schema (frozen `node_type("loop_policy")`)

`LoopPolicy` is the loop-as-data substrate — a frozen pydantic model the interpreter
reads as a node. Its knobs, each non-vacuous and mutation-tested:

- **Budgets (hard halts).** `step_budget` — checked at the turn boundary BEFORE the
  next turn, so N steps produce AT MOST N `agent.step` nodes (the off-by-one matters).
  `token_budget` — a HARD cumulative halt (`input+output+reasoning`; cache tokens not
  counted) that ALWAYS wins and is terminal, with NO compaction on the halting turn.
- **`compaction` (soft watermark).** `compaction_threshold` is a SOFT watermark: when
  cumulative tokens reach it (but NOT `token_budget`), older turns fold into an
  `agent.summary` node on the next assemble and the loop CONTINUES. A model validator
  enforces `compaction_threshold < token_budget` when both are set, so the soft
  watermark always fires before the hard halt — they can never breach simultaneously.
- **`stop_conditions`.** A `sorted()` tuple of `StopCondition`
  (`no_tool_calls` / `max_steps` / `max_tokens` / `refusal` / `explicit_stop_node`);
  the first firing in sorted order is the recorded halt reason (deterministic).
- **`planning`.** Three DISTINCT FSM transition tables: `off` (straight loop),
  `single` (one leading planning turn — its own `agent.step`, counts against
  `step_budget`), `replan_on_failure` (a planning turn after a non-retryable tool
  error). Collapsing `single` and `off` to one table is the mutation the tests catch.
- **`tool_choice`.** A discriminated union: `model` (all tools), `forced` (model MUST
  emit ≥1 `tool_use`), `restricted(types)` (offered set FILTERED to declarations whose
  name ∈ `types`, BEFORE the request; a `tool_use` naming an out-of-set tool is
  rejected). `types` is `sorted()` before serialization.
- **`require_tool_intent`.** When `True`, a `ToolUseBlock` missing a non-empty
  ≤1-sentence `intent` is a policy violation — the UX-legibility contract, enforced.

### 2.2 Sub-agent mechanics (frozen `SubAgentSpawn`)

`SubAgentSpawn` is `deny` or `allow(max_depth, max_fanout)` (both ≥1). A child run is
linked `child --PART_OF--> parent` via `runtime.link`. Each child gets its OWN budget
from its own `loop_policy` — **no pooling** (pooling would require a shared mutable
counter, violating TRAP-1).

**Runaway spawning is provably impossible — cardinality AND cost bounded.** The
pre-spawn gate checks two INDEPENDENT bounds from the DURABLE graph BEFORE any child
node is created:

1. `child_depth = parent_depth + 1 ≤ max_depth`, where `parent_depth` is the length of
   the `PART_OF` chain walked `direction="out"` (upward) — never `"both"`.
2. `parent_fanout + 1 ≤ max_fanout`, counted via
   `neighbors(parent, edge_type="part_of", direction="in")` — never `"both"`.

A spawn that would breach EITHER bound is refused before any child node/edge is written
(a bounded refusal recorded as a `tool_result`; the parent continues), so a
replay/resume cannot resurrect a phantom over-spawn. The total reachable run count is
bounded by the closed form `N_max = Σ_{i=0..d} fⁱ` (`max_run_tree_size`), and aggregate
token spend by `N_max × per_run_token_budget`. Result propagation is a graph read: a
child folds to a terminal status + final message, and the parent's NEXT
`ContextAssembler` pass reads that output via `PART_OF` — never an in-memory handoff, so
it survives replay/resume.

## ADR (finalized at M8)

**Title:** vella.agent v0.1 — cognition core as a self-hosted interpreter over the
Vella substrate.

**Decision.** Build `vella.agent` as a fourth downward-only sibling
(`{core, runtime, graph}` deps; **not** reconciler) whose
run/step/tool_call/message/summary/loop_policy/provider/tool/mcp_server are
**registered core node types** (no new envelopes), driven by a **data-configured FSM
interpreter** over a frozen `loop_policy` schema, acting solely through the runtime's
public verbs, perceiving via the graph, with three `Protocol` seams
(`ModelProvider` / `ToolInvoker` / `ContextAssembler`) each having a deterministic
in-gate reference impl and optional out-of-gate real adapters. Sequence: contracts-up,
interpreter-late (Option A), with self-hosting proven at M1.

**Drivers.**
1. **Contract dependency depth** forces contracts-before-interpreter: the interpreter
   depends on provider turns, tool invocation, and context, which must each have a
   deterministic reference impl before the interpreter consumes it.
2. **The two highest-risk surfaces** — deterministic streaming partial-JSON assembly
   and bounded sub-agent recursion — must each be isolated in their own milestone so
   the separate mutation-verifier can attack them without confound.
3. **Surface-freeze timing:** baseline an empty `__all__` at M0, grow it additively,
   and freeze the canonical-turn / Protocol / node-type / loop_policy surfaces at
   M2/M3/M4/M5/M6 — never retyping a shipped field. The surface is frozen at **84**.

**Alternatives considered.**
- **Option B (vertical thin-slice first)** — REJECTED. A minimal end-to-end slice at
  M1 hard-codes shapes the frozen contracts forbid (canonical turn, discriminated
  bindings, typed knobs), so it ships throwaway code the surface tripwire then churns;
  and mixing streaming-assembler determinism with loop logic in one slice CONFOUNDS the
  M2 mutation-verifier (driver #2 violated).
- **Option C (two parallel tracks joined at the interpreter)** — INVALIDATED. This is a
  single-builder lineage; parallel tracks need a frozen inter-track interface contract
  *before* either starts, which is exactly M2/M3 — so the tracks cannot truly start in
  parallel without first doing M2+M3 sequentially. Option C collapses into Option A
  with coordination overhead.

**Why chosen.** Option A is the only decomposition where every milestone compiles and
gates green in isolation (each contract has a deterministic ref impl before the
interpreter consumes it), the two risk surfaces are individually attackable by the
separate verifier, and it maps 1:1 onto the spec's acceptance-criteria clusters and the
team's proven graph/reconciler milestone cadence.

**Consequences.**
- The end-to-end loop is not demonstrable until M5 — mitigated by the M1 write-back
  smoke (self-hosting proven at M1) and interpreter-shaped consumer stubs at M2/M3/M4
  that prove the frozen surface composes correctly before M5 wires the real interpreter.
- A contract error found late forces a surface re-baseline — mitigated by freezing
  contracts at M2/M3 and the stubs catching shape mismatches early.
- A near-zero-dep core with heavy I/O quarantined in extras: importing the core never
  pulls `httpx`/`mcp`.
- Full replayability / observability / multi-agent topology comes for free via
  self-hosting; the cost is that EVERY behaviour must be expressible as nodes/edges +
  `observe_only` telemetry through the public verbs — no shortcut path is available.

**Follow-ups (v0.2 / out of scope now).**
- A vector-similarity / vectorstore `ContextAssembler` (the seam is defined; v0.1 recall
  is graph-relationship only, no vector).
- Persisted run/cursor/state backends (the seam is defined; v0.1 is in-memory).
- `vella.effects` — a transport extraction if it earns its own distribution.
- An HTTP `ToolInvoker` adapter (the `HTTPBinding` shape is reserved).
- A shared `Clock` conformance-suite testing-utils package (currently duplicated
  verbatim, as in graph/reconciler).
