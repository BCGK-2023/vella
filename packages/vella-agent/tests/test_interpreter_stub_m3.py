"""Freeze-timing consumer stub (M3): drive the tool seam as M5 will.

The freeze-timing gate (plan §3): each pre-M5 contract must be exercised through the
EXACT call-shape the M5 interpreter's turn loop will use, so a wrong-shaped frozen
surface is caught at freeze time — not at M5. This is NOT the interpreter; it is the
minimal slice proving the ``ToolInvoker`` + ``ToolData`` + discovery + hint surface
composes under the interpreter's calling convention:

    discover tools (graph + HAS_TOOL)          # seeded baseline + linked
      -> build TurnRequest.tools from the discovered tool-nodes' declarations
      -> provider.turn(req) yields a ToolUseBlock
      -> invoker.invoke(tool_node, args) -> ONE ToolResult   # retries internal
      -> resolve_hint(tool.hints, result)
      -> write agent.tool_call (tool_ref, args, intent, result, error_kind, hint)
      -> append the tool_result Message (ToolResultBlock carries the resolved hint)

No ``pytest-asyncio``: the async sequence runs via ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vella.core import Node, ToolDeclaration, UnresolvedRef
from vella.runtime import Runtime

from vella.agent import (
    BuiltinBinding,
    ErrorHint,
    InMemoryToolInvoker,
    Message,
    MockProvider,
    RunData,
    ScriptedToolUse,
    ScriptedTurn,
    StepData,
    ToolCallData,
    ToolData,
    ToolHints,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    TurnParams,
    TurnRequest,
    agent_registry,
    discover_tools,
    link_run_tool,
    resolve_hint,
    seed_system_tools,
)
from vella.agent._writeback import append_step, append_tool_call, create_run

_TENANT = "t-agent"
_ACTOR = UnresolvedRef(identifier="vella:test")


def _search_tool() -> ToolData:
    return ToolData(
        declaration=ToolDeclaration(name="search", description="search the corpus"),
        binding=BuiltinBinding(registry_key="search"),
        hints=ToolHints(
            result_hint="results are ranked; take the top hit",
            error_hints=(ErrorHint(match="RateLimited", hint="retry shortly"),),
            default_error_hint="unrecognized failure",
        ),
    )


def test_m3_tool_seam_composes_under_interpreter_call_shape() -> None:
    asyncio.run(asyncio.wait_for(_case_interpreter_shape(Runtime()), timeout=5.0))


async def _case_interpreter_shape(rt: Runtime) -> None:
    agent_registry()

    # --- setup: a run, a seeded baseline tool linked to the run via HAS_TOOL ---
    run = await create_run(rt, RunData(goal="find vella"), name="run", tenant_id=_TENANT)
    [search_node] = await seed_system_tools(rt, [_search_tool()], tenant_id=_TENANT)
    await link_run_tool(rt, run.id, search_node.id, tenant_id=_TENANT)

    # --- 1) discover tools (graph + HAS_TOOL), as M5's assemble step will ---
    tool_ids = await discover_tools(rt, run.id, tenant_id=_TENANT)
    assert tool_ids == (search_node.id,)
    tool_nodes: dict[str, Node[Any, Any]] = {}
    for tid in tool_ids:
        node = await rt.get(_TENANT, tid)
        assert isinstance(node, Node)
        tool_nodes[node.data.declaration.name] = node

    # --- 2) build TurnRequest.tools from the discovered declarations ---
    request = TurnRequest(
        messages=(Message(role="user", content=()),),
        tools=tuple(n.data.declaration for n in tool_nodes.values()),
        params=TurnParams(tool_choice="model"),
    )
    assert [t.name for t in request.tools] == ["search"]

    # --- 3) provider.turn yields a ToolUseBlock the interpreter pattern-matches ---
    provider = MockProvider(
        [
            ScriptedTurn(
                blocks=(
                    ScriptedToolUse(
                        id="c1", name="search", input={"q": "vella"}, intent="search vella"
                    ),
                ),
                stop_reason="tool_use",
            )
        ]
    )
    turn = await provider.turn(request)
    tool_uses = [b for b in turn.content if isinstance(b, ToolUseBlock)]
    assert len(tool_uses) == 1
    use = tool_uses[0]

    # --- 4) invoke -> ONE ToolResult (retries internal to the invoker) ---
    async def _search_impl(args: dict[str, Any]) -> ToolResult:
        return ToolResult(content={"top": args["q"]})

    invoker = InMemoryToolInvoker({"search": _search_impl})
    tool_node = tool_nodes[use.name]
    result = await invoker.invoke(tool_node, use.input)
    assert isinstance(result, ToolResult)
    assert result.content == {"top": "vella"}

    # --- 5) resolve the hint from the tool-node's hints ---
    hint = resolve_hint(tool_node.data.hints, result)
    assert hint == "results are ranked; take the top hit"

    # --- 6) write the agent.tool_call record (the durable, replayable trace) ---
    step = await append_step(
        rt, run.id, StepData(turn_index=0), name="step-0", tenant_id=_TENANT
    )
    call = await append_tool_call(
        rt,
        step.id,
        ToolCallData(
            tool_ref=tool_node.id,
            args=use.input,
            intent=use.intent,
            result=result.content,
            error_kind=result.error_kind,
            hint=hint,
        ),
        name="call-0",
        tenant_id=_TENANT,
    )
    got = await rt.get(_TENANT, call.id)
    assert isinstance(got, Node) and got.type == "agent.tool_call"
    assert isinstance(got.data, ToolCallData)
    assert got.data.hint == hint
    assert got.data.tool_ref == tool_node.id

    # --- 7) the tool_result Message the next turn feeds back carries the hint ---
    tool_result_message = Message(
        role="tool",
        content=(
            ToolResultBlock(
                tool_use_id=use.id,
                content=result.content,
                is_error=result.is_error,
                hint=hint,
            ),
        ),
    )
    block = tool_result_message.content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.hint == hint
    assert block.tool_use_id == use.id
