# vella-agent

A self-hosted cognition core for the
[Vella](https://github.com/BCGK-2023/vella) SDK. Where `vella-runtime` is *physics*
(the append-only log, the optimistic-concurrency store, and the write verbs that
move state forward) and `vella-graph` is a *read-only projection* that answers
traversal queries from memory, `vella-agent` is the *cognition core*: a
data-configured agent interpreter that acts only through the runtime's verbs and
perceives only through the graph.

```bash
pip install vella-agent
```

The agent depends on `vella-runtime`, `vella-graph`, and `vella-core` and on
nothing higher in the stack — and **never** on `vella-reconciler`, which is a
sibling rather than a dependency. All three lower layers are unaware of it. The
agent takes no privileged path: an agent run, its steps, tool calls, messages, and
policy are ordinary registered core node types, and the agent acts solely through
the runtime's public write verbs. Its public surface is snapshotted by a surface
tripwire so accidental breaking changes fail the gate.

The public surface grew milestone by milestone and is now **frozen** at 84 symbols
(the surface tripwire enforces it). M1 added the self-hosting cognition node
type-specs (the frozen `agent.run` / `agent.step` / `agent.message` / `agent.summary`
data payloads plus the registry accessors that keep tests isolated from core's
process-wide default registry). M2 froze the **canonical turn** — the `ModelProvider`
surface every adapter and the interpreter speak: the discriminated `ContentBlock`
union (`TextBlock` / `ThinkingBlock` / `ToolUseBlock` / `ToolResultBlock`), the
`Message` / `AssistantTurn` / `Usage` / `StopReason` shapes, the streaming lifecycle
event types, and the deterministic `MockProvider` reference impl. M3 froze the
**tool-node + invoker contract**: the `ToolData` payload (a discriminated `Binding`
union, `ToolHints`, an optional `RetryPolicy`), the `ToolResult` shape, the structural
`ToolInvoker` seam with its in-gate `InMemoryToolInvoker` (builtin dispatch + capped
backoff via an injected `Clock`), graph-driven `HAS_TOOL` tool discovery, and hint
resolution. M4 froze the **`ContextAssembler` seam**: the `AssembledContext` result
(canonical messages + a cache-breakpoint marker), the `CompactionPolicy` knobs, and
the in-gate `GraphContextAssembler` (stable cacheable prefix + volatile tail +
`agent.summary` compaction at the soft watermark; graph-relationship recall, no
vector) plus the `provider` node type whose `cache_capable` flag drives the strategy.
M5 froze the **loop-as-data FSM**: the typed `LoopPolicy` schema (budgets, stop
conditions, `ToolChoice`, planning modes, compaction) and the `run` interpreter entry
point — the first demonstrable end-to-end self-hosting loop. M6 froze **bounded
sub-agents** (`SubAgentSpawn`, `max_run_tree_size`): a parent run spawns children
`PART_OF` itself, gated `max_depth`/`max_fanout` from the durable graph before any
child node exists, so runaway spawning is provably impossible. M7 added the
out-of-gate `[openrouter]` / `[mcp]` adapters (the surface stayed at 84 — adapters
import lazily and are not in `__all__`):

```pycon
>>> import vella.agent
>>> len(vella.agent.__all__)
84
>>> all(
...     name in vella.agent.__all__
...     for name in ("AssistantTurn", "MockProvider", "ToolInvoker",
...                  "ContextAssembler", "LoopPolicy", "run", "SubAgentSpawn")
... )
True
>>> sorted(vella.agent.__all__)[:4]
['AssembledContext', 'AssemblyPolicy', 'AssistantTurn', 'Binding']

```

A `MockProvider` is the deterministic, scriptable reference `ModelProvider`: it
emits a canned turn as the full streaming lifecycle (fragmented tool-call JSON
included), and the non-streaming `turn()` wrapper drains that stream through the same
deterministic accumulator the live path uses — so mock and live prove the identical
canonical shape:

```pycon
>>> import asyncio
>>> from vella.agent import MockProvider, ScriptedTurn, ScriptedText, TurnRequest
>>> p = MockProvider([ScriptedTurn(blocks=(ScriptedText(text="hi", fragments=2),))])
>>> turn = asyncio.run(p.turn(TurnRequest()))
>>> (turn.role, turn.stop_reason, turn.content[0].text)
('assistant', 'end_turn', 'hi')

```

A degenerate end-to-end run drives the FSM interpreter (`run`) over a frozen
`LoopPolicy`: it materializes its cognition through the runtime's public verbs — the
run/step/message nodes via `create`/`link`, the reasoning trace via `emit_telemetry`
(an `observe_only` entry that never bumps the state-table version) — and returns a
terminal `RunResult`. Here a single `end_turn` assistant turn with no tool calls
satisfies the `no_tool_calls` stop condition and the loop halts after one step:

```pycon
>>> import asyncio
>>> from vella.core import Node, UnresolvedRef
>>> from vella.runtime import Runtime
>>> from vella.agent import (
...     GraphContextAssembler, InMemoryToolInvoker, LoopPolicy, ManualClock,
...     MockProvider, RunData, ScriptedText, ScriptedTurn, Usage,
...     agent_registry, run,
... )
>>> from vella.agent._writeback import create_run
>>> async def demo() -> tuple[str, str | None, int]:
...     agent_registry()  # register the agent.* node types into a fresh registry
...     rt = Runtime()
...     policy = LoopPolicy(stop_conditions=("no_tool_calls",))
...     actor = UnresolvedRef(identifier="vella:demo")
...     pol = Node.from_data(policy, name="p", created_by=actor, tenant_id="t")
...     await rt.create(pol)
...     run_node = await create_run(
...         rt, RunData(goal="say hi", loop_policy_ref=pol.id), name="r", tenant_id="t"
...     )
...     provider = MockProvider([ScriptedTurn(
...         blocks=(ScriptedText(text="hi"),), stop_reason="end_turn", usage=Usage(),
...     )])
...     result = await run(
...         rt, run_node.id, tenant_id="t", provider=provider,
...         invoker=InMemoryToolInvoker(clock=ManualClock()),
...         assembler=GraphContextAssembler(), clock=ManualClock(), max_steps=8,
...     )
...     return result.status, result.halt_reason, result.steps
>>> asyncio.run(demo())
('succeeded', 'no_tool_calls', 1)

```
