"""Stop conditions: no_tool_calls ends; refusal ends; sorted-first reason recorded.

* ``no_tool_calls`` — an assistant ``end_turn`` with zero tool calls ends the run.
* ``refusal`` — a ``stop_reason=="refusal"`` turn ends the run.
* When several conditions could fire on one turn, the FIRST in SORTED order is the
  recorded halt reason (deterministic — the policy stores ``stop_conditions`` sorted).

No ``pytest-asyncio``: ``asyncio.run`` + bounded ``max_steps`` + ManualClock.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from vella.core import EdgeTypes, Node, ToolDeclaration
from vella.graph import GraphProjection
from vella.runtime import Runtime

from vella.agent import (
    EXPLICIT_STOP_TOOL,
    BuiltinBinding,
    GraphContextAssembler,
    InMemoryToolInvoker,
    LoopPolicy,
    ManualClock,
    MockProvider,
    RunResult,
    ScriptedText,
    ScriptedTurn,
    ToolCallData,
    ToolData,
    ToolResult,
    Usage,
    agent_registry,
    link_run_tool,
    run,
    seed_system_tools,
)

from _interp_helper import make_run, text_turn, tool_turn

_TENANT = "t-stop"


async def _drive(
    rt: Runtime, run_id: UUID, provider: MockProvider, *, max_steps: int
) -> RunResult:
    return await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=max_steps,
    )


def test_no_tool_calls_ends_the_run() -> None:
    asyncio.run(asyncio.wait_for(_case_no_tool_calls(), timeout=10.0))


async def _case_no_tool_calls() -> None:
    agent_registry()
    rt = Runtime()
    run_id = await make_run(
        rt, LoopPolicy(stop_conditions=("no_tool_calls",)), tenant_id=_TENANT
    )
    # The first turn is an end_turn with no tools => no_tool_calls fires immediately.
    provider = MockProvider([text_turn("done"), text_turn("never")])
    result = await _drive(rt, run_id, provider, max_steps=10)
    assert result.halt_reason == "no_tool_calls"
    assert result.steps == 1  # ended after the first turn, never reached the second
    assert result.status == "succeeded"


def test_refusal_ends_the_run() -> None:
    asyncio.run(asyncio.wait_for(_case_refusal(), timeout=10.0))


async def _case_refusal() -> None:
    agent_registry()
    rt = Runtime()
    run_id = await make_run(rt, LoopPolicy(stop_conditions=("refusal",)), tenant_id=_TENANT)
    refusal = ScriptedTurn(
        blocks=(ScriptedText(text="I cannot help with that"),),
        stop_reason="refusal",
        usage=Usage(),
    )
    provider = MockProvider([refusal, text_turn("never")])
    result = await _drive(rt, run_id, provider, max_steps=10)
    assert result.halt_reason == "refusal"
    assert result.steps == 1


def test_sorted_first_condition_is_the_recorded_reason() -> None:
    asyncio.run(asyncio.wait_for(_case_sorted_first(), timeout=10.0))


async def _case_sorted_first() -> None:
    # Both no_tool_calls AND refusal are configured. A refusal turn ALSO has no tool
    # calls, so both predicates could match — sorted order is ('no_tool_calls',
    # 'refusal'), so the FIRST firing in that order wins. A refusal turn's stop_reason
    # is 'refusal' (not 'end_turn'), so no_tool_calls does NOT fire; refusal does.
    agent_registry()
    rt = Runtime()
    run_id = await make_run(
        rt,
        LoopPolicy(stop_conditions=("refusal", "no_tool_calls")),
        tenant_id=_TENANT,
    )
    # Sorting is proven on the policy itself:
    refusal = ScriptedTurn(
        blocks=(ScriptedText(text="no"),), stop_reason="refusal", usage=Usage()
    )
    provider = MockProvider([refusal])
    result = await _drive(rt, run_id, provider, max_steps=10)
    # Stored sorted: ('no_tool_calls', 'refusal'); on a refusal turn only refusal fires.
    assert result.halt_reason == "refusal"


def test_stop_conditions_are_stored_sorted() -> None:
    # The set-derived serialized value is sorted at construction (deterministic bytes).
    pol = LoopPolicy(stop_conditions=("refusal", "max_tokens", "no_tool_calls"))
    assert pol.stop_conditions == ("max_tokens", "no_tool_calls", "refusal")


# --- the explicit-stop-node sentinel + refusal-not-configured pins ---


def _stop_tool() -> ToolData:
    """A discoverable ``agent.tool`` for the reserved EXPLICIT_STOP_TOOL ('stop')."""
    return ToolData(
        declaration=ToolDeclaration(
            name=EXPLICIT_STOP_TOOL, description="signal goal completion"
        ),
        binding=BuiltinBinding(registry_key=EXPLICIT_STOP_TOOL),
    )


def _other_tool() -> ToolData:
    return ToolData(
        declaration=ToolDeclaration(name="work", description="do more work"),
        binding=BuiltinBinding(registry_key="work"),
    )


async def _ok_result(_args: dict[str, Any]) -> ToolResult:
    return ToolResult(content={"ok": True})


async def _stop_step_tool_calls(rt: Runtime, run_id: UUID) -> list[ToolCallData]:
    """Every durable ``agent.tool_call`` payload across the run's steps (by step)."""
    view = await GraphProjection().fold(rt, _TENANT)
    steps = await view.neighbors(run_id, edge_type=EdgeTypes.PART_OF, direction="in")
    calls: list[ToolCallData] = []
    for step in steps:
        step_node = await rt.get(_TENANT, step.node_id)
        if not (isinstance(step_node, Node) and step_node.type == "agent.step"):
            continue
        tcs = await view.neighbors(
            step.node_id, edge_type=EdgeTypes.PART_OF, direction="in"
        )
        for tc in tcs:
            node = await rt.get(_TENANT, tc.node_id)
            if isinstance(node, Node) and isinstance(node.data, ToolCallData):
                calls.append(node.data)
    return calls


def test_explicit_stop_node_fires_through_the_interpreter() -> None:
    asyncio.run(asyncio.wait_for(_case_explicit_stop(), timeout=10.0))


async def _case_explicit_stop() -> None:
    # ONLY explicit_stop_node is configured. A turn that calls the reserved stop tool
    # halts the run AFTER the invocation is durably recorded (halt is post-invocation).
    agent_registry()
    rt = Runtime()
    run_id = await make_run(
        rt,
        LoopPolicy(stop_conditions=("explicit_stop_node",)),
        tenant_id=_TENANT,
    )
    nodes = await seed_system_tools(
        rt, [_stop_tool(), _other_tool()], tenant_id=_TENANT
    )
    for node in nodes:
        await link_run_tool(rt, run_id, node.id, tenant_id=_TENANT)

    invoker = InMemoryToolInvoker(
        {EXPLICIT_STOP_TOOL: _ok_result, "work": _ok_result}, clock=ManualClock()
    )
    # Non-vacuity: a SECOND turn that calls a NON-stop tool would NOT stop on its own;
    # if the sentinel did not fire, the run would proceed into that turn. It halting on
    # turn 0 proves it stopped specifically on the explicit_stop sentinel.
    provider = MockProvider(
        [
            tool_turn(
                tool_id="s1", name=EXPLICIT_STOP_TOOL, args={}, intent="goal complete."
            ),
            tool_turn(tool_id="w1", name="work", args={}, intent="keep working."),
        ]
    )
    result = await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=invoker,
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=5,
    )
    assert result.halt_reason == "explicit_stop_node"
    assert result.status == "succeeded"
    assert result.steps == 1  # halted on turn 0, never reached the "work" turn

    # The stop tool's call was DURABLY written before the halt (post-invocation halt).
    calls = await _stop_step_tool_calls(rt, run_id)
    stop_refs = {n.id for n in nodes if n.data.declaration.name == EXPLICIT_STOP_TOOL}
    assert any(c.tool_ref in stop_refs for c in calls)
    # And the "work" tool was never called (the run halted on turn 0).
    work_refs = {n.id for n in nodes if n.data.declaration.name == "work"}
    assert not any(c.tool_ref in work_refs for c in calls)


def test_provider_refusal_does_not_halt_when_refusal_not_configured() -> None:
    asyncio.run(asyncio.wait_for(_case_refusal_not_configured(), timeout=10.0))


async def _case_refusal_not_configured() -> None:
    # stop_conditions is EMPTY: refusal is NOT a configured terminal condition. A
    # provider that returns a refusal end-turn must NOT halt the run — the loop
    # continues and ends only via the max_steps backstop. This pins that refusal is a
    # CONFIGURED stop condition, never implicitly terminal.
    agent_registry()
    rt = Runtime()
    run_id = await make_run(rt, LoopPolicy(stop_conditions=()), tenant_id=_TENANT)
    refusal = ScriptedTurn(
        blocks=(ScriptedText(text="I cannot help with that"),),
        stop_reason="refusal",
        usage=Usage(),
    )
    # First a refusal turn, then more turns the loop should proceed into.
    provider = MockProvider(
        [refusal, text_turn("kept going"), text_turn("still going"), text_turn("more")]
    )
    # Non-vacuity: max_steps > 1 so the continuation past the refusal is observable.
    result = await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=3,
    )
    assert result.halt_reason == "max_steps"
    assert result.steps == 3  # ran past the refusal until the backstop
    assert result.steps > 1
