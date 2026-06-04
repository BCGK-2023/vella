"""Freeze-timing consumer stub (M4): drive the assembler as the M5 turn loop will.

The freeze-timing gate (plan §3): each pre-M5 contract must be exercised through the
EXACT call-shape the M5 interpreter's turn loop will use, so a wrong-shaped frozen
surface is caught at freeze time — not at M5. This is NOT the interpreter; it is the
minimal slice proving the ``ContextAssembler`` surface composes under the
interpreter's calling convention:

    assemble(runtime, run_node, tenant_id, provider_node, policy)   # perceive
      -> AssembledContext.messages (canonical Messages)
      -> build TurnRequest(messages, tools, params{cache: prefix is a breakpoint})
      -> provider.turn(req) -> AssistantTurn the interpreter pattern-matches on

The assembler is exercised against the SAME structural seam the interpreter binds
(``ContextAssembler``), and its output feeds straight into the frozen
``ModelProvider``/canonical-turn surface — proving the three seams compose.

No ``pytest-asyncio``: the async sequence runs via ``asyncio.run``.
"""

from __future__ import annotations

import asyncio

from vella.core import Node, ToolDeclaration, UnresolvedRef
from vella.runtime import Runtime

from vella.agent import (
    AssembledContext,
    AssemblyPolicy,
    CompactionPolicy,
    ContextAssembler,
    GraphContextAssembler,
    Message,
    MessageData,
    MockProvider,
    ProviderData,
    RunData,
    ScriptedText,
    ScriptedTurn,
    TextBlock,
    TurnParams,
    TurnRequest,
    agent_registry,
)
from vella.agent._writeback import append_message, create_run

_TENANT = "t-agent"
_ACTOR = UnresolvedRef(identifier="vella:test")


def test_m4_assembler_composes_under_interpreter_call_shape() -> None:
    asyncio.run(asyncio.wait_for(_case_interpreter_shape(), timeout=5.0))


async def _case_interpreter_shape() -> None:
    agent_registry()
    rt = Runtime()

    # --- setup: a run, a cache-capable provider node, one user turn ---
    run = await create_run(
        rt, RunData(goal="find vella"), name="run", tenant_id=_TENANT
    )
    provider_node = Node.from_data(
        ProviderData(model_id="m", cache_capable=True),
        name="provider",
        created_by=_ACTOR,
        tenant_id=_TENANT,
    )
    await rt.create(provider_node)
    await append_message(
        rt,
        run.id,
        MessageData(role="user", content=(TextBlock(text="search please"),)),
        name="u0",
        tenant_id=_TENANT,
    )

    # The interpreter binds the structural ContextAssembler seam (not a concrete).
    assembler: ContextAssembler = GraphContextAssembler()
    assert isinstance(assembler, ContextAssembler)

    # --- 1) assemble: perceive the run into canonical Messages + cache metadata ---
    ctx = await assembler.assemble(
        rt,
        run.id,
        tenant_id=_TENANT,
        provider_node=provider_node.id,
        policy=AssemblyPolicy(compaction=CompactionPolicy()),
    )
    assert isinstance(ctx, AssembledContext)
    assert all(isinstance(m, Message) for m in ctx.messages)
    # the cache breakpoint is data the interpreter reads to set the cache directive.
    assert ctx.cacheable_prefix_len >= 1

    # --- 2) build the TurnRequest exactly as M5 will (messages + tools + params) ---
    request = TurnRequest(
        messages=ctx.messages,
        tools=(ToolDeclaration(name="search", description="search the corpus"),),
        params=TurnParams(
            max_tokens=256,
            tool_choice="model",
            cache=ctx.cacheable_prefix_len > 0,
        ),
    )
    assert request.params.cache is True
    assert request.messages[0].role == "system"

    # --- 3) provider.turn yields the canonical AssistantTurn the interpreter matches ---
    provider = MockProvider(
        [ScriptedTurn(blocks=(ScriptedText(text="on it"),), stop_reason="end_turn")]
    )
    turn = await provider.turn(request)
    assert turn.stop_reason == "end_turn"
    block = turn.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "on it"
