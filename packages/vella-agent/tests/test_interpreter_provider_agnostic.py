"""Provider-agnostic end-to-end: one policy, two dialect framings, one outcome.

The interpreter pattern-matches ONLY on the canonical turn — never on a provider
specific. So the SAME :class:`~vella.agent.LoopPolicy` driven over two MockProvider
scripts that frame the identical logical turns differently (block-at-a-time vs
round-robin interleaved deltas, and different argument-JSON fragment counts) must
reach the SAME terminal run projection. This is the M5 provider-agnostic acceptance
criterion at the loop level (M2 proved it at the turn-assembly level).

No ``pytest-asyncio``: ``asyncio.run`` + bounded ``max_steps`` + ManualClock.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vella.core import EdgeTypes, Node, ToolDeclaration
from vella.graph import GraphProjection
from vella.runtime import Runtime

from vella.agent import (
    BuiltinBinding,
    GraphContextAssembler,
    InMemoryToolInvoker,
    LoopPolicy,
    ManualClock,
    MockProvider,
    ScriptedText,
    ScriptedToolUse,
    ScriptedTurn,
    StepData,
    ToolData,
    ToolResult,
    Usage,
    agent_registry,
    link_run_tool,
    run,
    seed_system_tools,
)

from _interp_helper import make_run

_TENANT = "t-agnostic"


async def _impl(args: dict[str, Any]) -> ToolResult:
    return ToolResult(content={"echo": args})


async def _digest(rt: Runtime, run_id: Any) -> dict[str, Any]:
    """A canonical, provider-independent digest of the terminal projection."""
    view = await GraphProjection().fold(rt, _TENANT)
    neighbours = await view.neighbors(run_id, edge_type=EdgeTypes.PART_OF, direction="in")
    steps: list[tuple[int, str]] = []
    roles: list[str] = []
    for nb in neighbours:
        node = await rt.get(_TENANT, nb.node_id)
        if not isinstance(node, Node):
            continue
        data = node.data
        if isinstance(data, StepData):
            steps.append((data.turn_index, data.kind))
        elif getattr(data, "role", None) is not None and hasattr(data, "content"):
            roles.append(data.role)
    run_node = await rt.get(_TENANT, run_id)
    assert isinstance(run_node, Node)
    return {
        "status": run_node.data.status,
        "steps": sorted(steps),
        "roles": sorted(roles),
    }


async def _drive_with(script: list[ScriptedTurn]) -> dict[str, Any]:
    agent_registry()
    rt = Runtime()
    run_id = await make_run(
        rt, LoopPolicy(stop_conditions=("no_tool_calls",)), tenant_id=_TENANT
    )
    tool = ToolData(
        declaration=ToolDeclaration(name="search", description="search"),
        binding=BuiltinBinding(registry_key="search"),
    )
    [node] = await seed_system_tools(rt, [tool], tenant_id=_TENANT)
    await link_run_tool(rt, run_id, node.id, tenant_id=_TENANT)
    result = await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=MockProvider(script),
        invoker=InMemoryToolInvoker({"search": _impl}, clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=6,
    )
    assert result.halt_reason == "no_tool_calls"
    return await _digest(rt, run_id)


def test_two_dialect_framings_reach_same_terminal_projection() -> None:
    asyncio.run(asyncio.wait_for(_case_agnostic(), timeout=20.0))


async def _case_agnostic() -> None:
    # Framing A: a multi-block tool turn streamed block-at-a-time, args in 1 fragment.
    framing_a = [
        ScriptedTurn(
            blocks=(
                ScriptedText(text="let me search", fragments=1),
                ScriptedToolUse(
                    id="c1", name="search", input={"q": "vella"}, intent="search.", fragments=1
                ),
            ),
            stop_reason="tool_use",
            usage=Usage(output_tokens=3),
            interleave=False,
        ),
        ScriptedTurn(blocks=(ScriptedText(text="done"),), stop_reason="end_turn"),
    ]
    # Framing B: the SAME logical turn, but deltas interleaved round-robin and the tool
    # argument JSON split into many fragments + the text into several — a different
    # wire dialect that the canonical accumulator normalizes to the SAME turn.
    framing_b = [
        ScriptedTurn(
            blocks=(
                ScriptedText(text="let me search", fragments=5),
                ScriptedToolUse(
                    id="c1", name="search", input={"q": "vella"}, intent="search.", fragments=7
                ),
            ),
            stop_reason="tool_use",
            usage=Usage(output_tokens=3),
            interleave=True,
        ),
        ScriptedTurn(blocks=(ScriptedText(text="done"),), stop_reason="end_turn"),
    ]
    digest_a = await _drive_with(framing_a)
    digest_b = await _drive_with(framing_b)
    assert digest_a == digest_b
    assert digest_a["status"] == "succeeded"
    assert digest_a["steps"] == [(0, "turn"), (1, "turn")]
