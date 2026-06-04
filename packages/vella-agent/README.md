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

The public surface grows milestone by milestone. M1 added the self-hosting
cognition node type-specs (the frozen `agent.run` / `agent.step` / `agent.message` /
`agent.summary` data payloads plus the registry accessors that keep tests isolated
from core's process-wide default registry). M2 freezes the **canonical turn** — the
`ModelProvider` surface every adapter and the interpreter speak: the discriminated
`ContentBlock` union (`TextBlock` / `ThinkingBlock` / `ToolUseBlock` /
`ToolResultBlock`), the `Message` / `AssistantTurn` / `Usage` / `StopReason` shapes,
the streaming lifecycle event types, and the deterministic `MockProvider` reference
impl. The remaining two Protocol seams and the FSM interpreter land in later
milestones:

```pycon
>>> import vella.agent
>>> "AssistantTurn" in vella.agent.__all__ and "MockProvider" in vella.agent.__all__
True
>>> sorted(vella.agent.__all__)[:4]
['AssistantTurn', 'ContentBlock', 'ContentBlockDelta', 'ContentBlockStart']

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

A degenerate run materializes its cognition through the runtime's public verbs —
the run/step/message nodes via `create`/`link`, the reasoning trace via
`emit_telemetry` (an `observe_only` entry that never bumps the state-table version):

```pycon
>>> import asyncio
>>> from uuid import UUID
>>> from vella.agent import RunData, StepData
>>> from vella.agent._writeback import create_run, append_step
>>> from vella.runtime import Runtime
>>> async def demo() -> str:
...     rt = Runtime()
...     run = await create_run(rt, RunData(goal="hello"), name="r", tenant_id="t")
...     await append_step(rt, run.id, StepData(turn_index=0), name="s", tenant_id="t")
...     got = await rt.get("t", run.id)
...     return "" if got is None else got.type
>>> asyncio.run(demo())
'agent.run'

```
