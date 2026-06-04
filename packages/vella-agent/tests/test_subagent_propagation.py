"""Sub-agent result propagation (§2.2): child terminal output reaches the parent's
next turn VIA THE GRAPH — never an in-memory handoff.

The parent spawns ONE child (depth 1, fanout 1). The child folds to a terminal status
with a distinctive final message. The parent's NEXT turn must perceive that child's
terminal text — and it does so by reading the DURABLE graph (child PART_OF parent,
explicit direction), not by any in-memory return value. We prove "via the graph" two
ways:

* a recording provider captures the messages the parent's later turns receive and we
  assert the child's terminal text appears there;
* a SEPARATE, FRESH read of :func:`~vella.agent._subagent.child_terminal_messages`
  over the same runtime (no shared in-memory state with the run) returns the child's
  terminal output — exactly the read the parent's assembly performs.

Mutation (d): propagating the child result in-memory instead of via a graph read makes
the fresh-read assertion go RED (a fresh reader has no in-memory handoff to observe).

No ``pytest-asyncio``: ``asyncio.run`` + bounded ``max_steps`` + ManualClock.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from uuid import UUID

from vella.runtime import Runtime

from vella.agent import (
    GraphContextAssembler,
    InMemoryToolInvoker,
    LoopPolicy,
    ManualClock,
    MockProvider,
    ScriptedText,
    ScriptedToolUse,
    ScriptedTurn,
    SubAgentAllow,
    TurnEvent,
    TurnRequest,
    Usage,
    agent_registry,
    run,
)
from vella.agent._subagent import SPAWN_TOOL, child_terminal_messages

from _interp_helper import make_run

_TENANT = "t-propagation"
_CHILD_ANSWER = "CHILD-ANSWER-7f3a"


class RecordingProvider:
    """A provider that DELEGATES to an inner script but records every request it sees.

    Satisfies the structural :class:`~vella.agent.ModelProvider` shape. The interpreter
    feeds it the assembled request each turn; we capture those so a test can assert the
    parent's later turns perceived the child's terminal output.
    """

    def __init__(self, inner: MockProvider) -> None:
        self._inner = inner
        self.seen: list[TurnRequest] = []

    def stream(self, request: TurnRequest) -> AsyncIterator[TurnEvent]:
        self.seen.append(request)
        return self._inner.stream(request)

    async def turn(self, request: TurnRequest):  # type: ignore[no-untyped-def]
        self.seen.append(request)
        return await self._inner.turn(request)


def _script() -> list[ScriptedTurn]:
    """The shared cursor's turns in EXECUTION order.

    Execution order (single shared MockProvider cursor): the parent's spawn turn runs
    the recursive child to terminal IMMEDIATELY, so the child's turn is consumed right
    after the spawn turn — BEFORE the parent's later turns. The child is bounded to ONE
    turn via its OWN ``step_budget=1`` (carried in the spawn input — its own budget, no
    pooling), so it folds to a terminal status after producing the distinctive answer.

        0: parent spawns a child (step_budget=1 for the child)
        1: the CHILD's single terminal turn (the distinctive answer)
        2..: the parent's later turns (whose requests must carry the propagated answer)
    """
    spawn = ScriptedTurn(
        blocks=(
            ScriptedToolUse(
                id="s1",
                name=SPAWN_TOOL,
                input={"goal": "do the child task", "step_budget": 1},
                intent="spawn a child to do the task.",
            ),
        ),
        stop_reason="tool_use",
        usage=Usage(output_tokens=1),
    )
    child = ScriptedTurn(
        blocks=(ScriptedText(text=_CHILD_ANSWER),),
        stop_reason="end_turn",
        usage=Usage(output_tokens=1),
    )
    idle = ScriptedTurn(
        blocks=(ScriptedText(text="parent continues"),),
        stop_reason="end_turn",
        usage=Usage(output_tokens=1),
    )
    return [spawn, child, idle, idle, idle]


def test_child_terminal_result_propagates_into_parent_next_turn_via_graph() -> None:
    asyncio.run(asyncio.wait_for(_case_propagation(), timeout=20.0))


async def _case_propagation() -> None:
    agent_registry()
    rt = Runtime()

    # Shared script: parent turns then the child turn. The recursive child run consumes
    # the child turn from the SAME provider cursor (it runs to terminal mid-parent-turn).
    script = _script()
    inner = MockProvider(script)
    provider = RecordingProvider(inner)

    root = await make_run(
        rt,
        LoopPolicy(
            step_budget=4,
            sub_agent_spawn=SubAgentAllow(max_depth=1, max_fanout=1),
        ),
        tenant_id=_TENANT,
    )

    await run(
        rt,
        root,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=4,
    )

    # (1) A FRESH graph read — the exact read the parent's assembly performs — returns
    # the child's terminal output. This is the "via the graph" proof: a brand-new reader
    # with NO in-memory handoff still sees the child answer (mutation (d) would make
    # this empty, because an in-memory-only propagation leaves nothing in the graph).
    propagated = await child_terminal_messages(rt, root, tenant_id=_TENANT)
    assert propagated, "no child terminal output was readable from the graph"
    flat = _flatten_text(propagated)
    assert _CHILD_ANSWER in flat

    # (2) The parent's turns AFTER the spawn received the child's terminal text in their
    # assembled request — perceived through the assembler's graph read, not handed in.
    post_spawn_requests = provider.seen[1:]  # request[0] is the pre-spawn turn
    assert any(
        _CHILD_ANSWER in _flatten_text(req.messages) for req in post_spawn_requests
    ), "the child's terminal output never reached a later parent turn's request"


def _flatten_text(messages: tuple) -> str:  # type: ignore[type-arg]
    """Join all text-block text across a message tuple (for substring assertions)."""
    parts: list[str] = []
    for msg in messages:
        for block in getattr(msg, "content", ()):
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return " ".join(parts)


def test_propagation_keyed_by_child_run_id_is_deterministic() -> None:
    asyncio.run(asyncio.wait_for(_case_deterministic(), timeout=20.0))


async def _case_deterministic() -> None:
    agent_registry()
    rt = Runtime()
    script = _script()
    provider = MockProvider(script)
    root = await make_run(
        rt,
        LoopPolicy(
            step_budget=4,
            sub_agent_spawn=SubAgentAllow(max_depth=1, max_fanout=1),
        ),
        tenant_id=_TENANT,
    )
    await run(
        rt,
        root,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=4,
    )
    # Re-reading propagation twice is byte-identical (graph read is a pure function of
    # the durable record; the summary block is keyed by the child run id + status).
    a = await child_terminal_messages(rt, root, tenant_id=_TENANT)
    b = await child_terminal_messages(rt, root, tenant_id=_TENANT)
    assert [m.model_dump(mode="json") for m in a] == [
        m.model_dump(mode="json") for m in b
    ]
    # The summary block embeds the child run id — confirm it is a real child id.
    child_ids = await _children_ids(rt, root)
    assert len(child_ids) == 1
    assert str(child_ids[0]) in _flatten_text(a)


async def _children_ids(rt: Runtime, parent: UUID) -> list[UUID]:
    from vella.agent._subagent import child_runs_of

    return await child_runs_of(rt, parent, tenant_id=_TENANT)
