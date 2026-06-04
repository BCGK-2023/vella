"""Shared fixtures for the M5 interpreter tests (no ``pytest-asyncio``).

A tiny set of builders the interpreter test modules reuse: a fresh tenant actor, a
helper to attach a :class:`~vella.agent.LoopPolicy` node + create a run referencing
it, and scripted-turn shorthands. Every test drives the loop via ``asyncio.run`` with
a bounded ``max_steps`` and a :class:`~vella.agent.ManualClock` (the no-pytest-asyncio
idiom). This is a test-support module (leading underscore), not a test itself.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from vella.core import Node, UnresolvedRef
from vella.runtime import Runtime

from vella.agent import (
    LoopPolicy,
    RunData,
    ScriptedText,
    ScriptedToolUse,
    ScriptedTurn,
    Usage,
)
from vella.agent._writeback import create_run

ACTOR = UnresolvedRef(identifier="vella:test")


async def make_run(
    rt: Runtime, policy: LoopPolicy, *, tenant_id: str, goal: str = "g"
) -> UUID:
    """Create a ``loop_policy`` node + a run referencing it; return the run id.

    Both writes go through the public verbs (``create`` for the policy node;
    :func:`~vella.agent._writeback.create_run` for the run). The policy is durable
    (a node), so the interpreter reads it from authority like any other.

    Args:
        rt: The runtime to write through.
        policy: The loop policy to attach to the run.
        tenant_id: The tenant both nodes belong to.
        goal: The run's goal text.

    Returns:
        The created ``agent.run`` node id.
    """
    policy_node = Node.from_data(
        policy, name="policy", created_by=ACTOR, tenant_id=tenant_id
    )
    await rt.create(policy_node)
    run_node = await create_run(
        rt,
        RunData(goal=goal, loop_policy_ref=policy_node.id),
        name="run",
        tenant_id=tenant_id,
    )
    return run_node.id


def text_turn(text: str, *, tokens: int = 0) -> ScriptedTurn:
    """A single-text-block ``end_turn`` turn spending ``tokens`` output tokens."""
    return ScriptedTurn(
        blocks=(ScriptedText(text=text),),
        stop_reason="end_turn",
        usage=Usage(output_tokens=tokens),
    )


def tool_turn(
    *, tool_id: str, name: str, args: dict[str, Any], intent: str, tokens: int = 0
) -> ScriptedTurn:
    """A single-``tool_use`` turn (``stop_reason='tool_use'``) with the given call."""
    return ScriptedTurn(
        blocks=(ScriptedToolUse(id=tool_id, name=name, input=args, intent=intent),),
        stop_reason="tool_use",
        usage=Usage(output_tokens=tokens),
    )
