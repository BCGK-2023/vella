"""Replay/resume: an interrupted run resumes from the LAST recorded step (fold).

A run interrupted mid-loop (bounded by ``max_steps`` as the interruption) resumes by
folding its DURABLE recorded steps from the graph + ``runtime.get`` (NOT in-memory
state — TRAP-1) and continuing from the last recorded turn index, reaching the SAME
terminal projection (the run/step/message node set) as an uninterrupted run of the
same script.

Mutation (e): resuming from the WRONG step index (e.g. from 0, or off-by-one) makes
the fold-equivalence assertion go red — the resumed run would re-run turns it already
recorded (duplicate steps) or skip turns (missing steps), so the projection diverges.

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
    BuiltinBinding,
    GraphContextAssembler,
    InMemoryToolInvoker,
    LoopPolicy,
    ManualClock,
    MockProvider,
    ScriptedTurn,
    StepData,
    ToolCallData,
    ToolData,
    ToolResult,
    agent_registry,
    link_run_tool,
    run,
    seed_system_tools,
)

from _interp_helper import make_run, text_turn, tool_turn

_TENANT = "t-resume"

# A deterministic, index-addressable script: turn k is "turn-k" with 1 output token.
# step_budget = 4 ends the run via max_steps after exactly 4 turns.
_TOTAL = 4


def _turn(k: int) -> ScriptedTurn:
    return text_turn(f"turn-{k}", tokens=1)


async def _projection(rt: Runtime, run_id: Any) -> dict[str, Any]:
    """A deterministic digest of the run's terminal projection (step kinds + roles)."""
    view = await GraphProjection().fold(rt, _TENANT)
    neighbours = await view.neighbors(run_id, edge_type=EdgeTypes.PART_OF, direction="in")
    steps: list[tuple[int, str]] = []
    roles: list[str] = []
    for nb in sorted(neighbours, key=lambda n: str(n.node_id)):
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


def test_interrupted_run_resumes_to_same_terminal_projection() -> None:
    asyncio.run(asyncio.wait_for(_case_resume_equivalence(), timeout=20.0))


async def _case_resume_equivalence() -> None:
    agent_registry()

    # --- uninterrupted reference: one run of the full 4-turn script ---
    rt_ref = Runtime()
    ref_run = await make_run(rt_ref, LoopPolicy(step_budget=_TOTAL), tenant_id=_TENANT)
    ref_provider = MockProvider([_turn(k) for k in range(_TOTAL + 2)])
    ref_result = await run(
        rt_ref,
        ref_run,
        tenant_id=_TENANT,
        provider=ref_provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=_TOTAL,
    )
    assert ref_result.halt_reason == "max_steps"
    assert ref_result.steps == _TOTAL
    ref_projection = await _projection(rt_ref, ref_run)

    # --- interrupted run: SAME runtime, driven in two segments ---
    rt = Runtime()
    run_id = await make_run(rt, LoopPolicy(step_budget=_TOTAL), tenant_id=_TENANT)

    # Segment 1: interrupt after 2 turns (max_steps backstop = 2, below step_budget).
    seg1_provider = MockProvider([_turn(0), _turn(1)])
    seg1 = await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=seg1_provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=2,  # the interruption: the bounded driver stops mid-loop
    )
    # Interrupted mid-loop: 2 durable steps, halted by the backstop (not the budget).
    assert seg1.steps == 2

    # Count durable steps BEFORE resuming — the authority the resume folds from.
    pre_resume_steps = await _count_steps(rt, run_id)
    assert pre_resume_steps == 2

    # Segment 2: resume. A FRESH provider scripting the REMAINING turns (2,3). The
    # interpreter folds the 2 durable steps and continues from turn_index 2 — NOT 0
    # (mutation (e): resuming from 0 would re-run turns and the projection diverges).
    seg2_provider = MockProvider([_turn(2), _turn(3), _turn(4)])
    seg2 = await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=seg2_provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=_TOTAL,  # enough to reach the step_budget halt at 4 total
    )
    assert seg2.halt_reason == "max_steps"
    # Total durable steps across both segments == the budget (no duplicates, no skips).
    assert seg2.steps == _TOTAL

    resumed_projection = await _projection(rt, run_id)

    # Fold-from-log equivalence: the resumed run's terminal projection matches the
    # uninterrupted reference's exactly (same step indices, same message roles, same
    # terminal status). A wrong resume index would change the step set.
    assert resumed_projection == ref_projection
    # And the resumed step indices are exactly 0..3 — contiguous, no duplicate of 0/1.
    assert resumed_projection["steps"] == [(i, "turn") for i in range(_TOTAL)]


async def _count_steps(rt: Runtime, run_id: Any) -> int:
    view = await GraphProjection().fold(rt, _TENANT)
    neighbours = await view.neighbors(run_id, edge_type=EdgeTypes.PART_OF, direction="in")
    n = 0
    for nb in neighbours:
        node = await rt.get(_TENANT, nb.node_id)
        if isinstance(node, Node) and isinstance(node.data, StepData):
            n += 1
    return n


# --- resume of a run interrupted DURING a real tool turn ---

_TENANT_TOOL = "t-resume-tool"


async def _tool_calls_for_turn(
    rt: Runtime, run_id: UUID, *, tenant: str, turn_index: int
) -> list[ToolCallData]:
    """Every durable tool_call payload on the step whose turn_index matches."""
    view = await GraphProjection().fold(rt, tenant)
    steps = await view.neighbors(run_id, edge_type=EdgeTypes.PART_OF, direction="in")
    out: list[ToolCallData] = []
    for step in steps:
        step_node = await rt.get(tenant, step.node_id)
        if not (
            isinstance(step_node, Node)
            and isinstance(step_node.data, StepData)
            and step_node.data.turn_index == turn_index
        ):
            continue
        tcs = await view.neighbors(
            step.node_id, edge_type=EdgeTypes.PART_OF, direction="in"
        )
        for tc in tcs:
            node = await rt.get(tenant, tc.node_id)
            if isinstance(node, Node) and isinstance(node.data, ToolCallData):
                out.append(node.data)
    return out


async def _tool_projection(rt: Runtime, run_id: UUID, *, tenant: str) -> dict[str, Any]:
    view = await GraphProjection().fold(rt, tenant)
    neighbours = await view.neighbors(
        run_id, edge_type=EdgeTypes.PART_OF, direction="in"
    )
    steps: list[tuple[int, str]] = []
    roles: list[str] = []
    for nb in sorted(neighbours, key=lambda n: str(n.node_id)):
        node = await rt.get(tenant, nb.node_id)
        if not isinstance(node, Node):
            continue
        data = node.data
        if isinstance(data, StepData):
            steps.append((data.turn_index, data.kind))
        elif getattr(data, "role", None) is not None and hasattr(data, "content"):
            roles.append(data.role)
    run_node = await rt.get(tenant, run_id)
    assert isinstance(run_node, Node)
    return {"status": run_node.data.status, "steps": sorted(steps), "roles": sorted(roles)}


def test_resume_of_a_run_interrupted_during_a_tool_turn() -> None:
    asyncio.run(asyncio.wait_for(_case_resume_tool_turn(), timeout=20.0))


def _tool() -> ToolData:
    return ToolData(
        declaration=ToolDeclaration(name="probe", description="a probe tool"),
        binding=BuiltinBinding(registry_key="probe"),
    )


async def _probe_impl(_args: Any) -> ToolResult:
    return ToolResult(content={"ok": True})


async def _make_tool_run(rt: Runtime) -> UUID:
    run_id = await make_run(
        rt, LoopPolicy(stop_conditions=("no_tool_calls",)), tenant_id=_TENANT_TOOL
    )
    [node] = await seed_system_tools(rt, [_tool()], tenant_id=_TENANT_TOOL)
    await link_run_tool(rt, run_id, node.id, tenant_id=_TENANT_TOOL)
    return run_id


def _invoker() -> InMemoryToolInvoker:
    return InMemoryToolInvoker({"probe": _probe_impl}, clock=ManualClock())


async def _case_resume_tool_turn() -> None:
    agent_registry()
    # Script: turn 0 is a REAL tool call (probe); turn 1 is a text turn that ends via
    # no_tool_calls. Interrupting at max_steps=1 stops right after the tool turn's step.
    tool0 = tool_turn(tool_id="p0", name="probe", args={}, intent="probe state.")
    text1 = text_turn("done")

    # --- uninterrupted reference ---
    rt_ref = Runtime()
    ref_run = await _make_tool_run(rt_ref)
    ref = await run(
        rt_ref,
        ref_run,
        tenant_id=_TENANT_TOOL,
        provider=MockProvider([tool0, text1]),
        invoker=_invoker(),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=5,
    )
    assert ref.halt_reason == "no_tool_calls"
    ref_proj = await _tool_projection(rt_ref, ref_run, tenant=_TENANT_TOOL)

    # --- interrupted run: SAME runtime, two segments ---
    rt = Runtime()
    run_id = await _make_tool_run(rt)

    # Segment 1: interrupt after the tool turn (max_steps=1). The step + agent.tool_call
    # + observe_only trace for turn 0 are durable.
    seg1 = await run(
        rt,
        run_id,
        tenant_id=_TENANT_TOOL,
        provider=MockProvider([tool0]),
        invoker=_invoker(),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=1,
    )
    assert seg1.steps == 1
    # Non-vacuity: the interrupted turn is a REAL tool call (exactly one tool_call for
    # turn 0), not a text-only turn — so this exercises the during-a-tool-turn path.
    turn0_calls = await _tool_calls_for_turn(
        rt, run_id, tenant=_TENANT_TOOL, turn_index=0
    )
    assert len(turn0_calls) == 1

    # Segment 2: FRESH interpreter scripting only the FOLLOW-ON turn. It folds the one
    # durable step and continues from turn_index 1.
    seg2 = await run(
        rt,
        run_id,
        tenant_id=_TENANT_TOOL,
        provider=MockProvider([text1]),
        invoker=_invoker(),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=5,
    )
    assert seg2.halt_reason == "no_tool_calls"

    # No duplicate step/tool_call for turn 0: exactly ONE tool_call for turn 0 still.
    turn0_after = await _tool_calls_for_turn(
        rt, run_id, tenant=_TENANT_TOOL, turn_index=0
    )
    assert len(turn0_after) == 1

    # Final projection equals the uninterrupted reference (same steps, roles, status).
    resumed_proj = await _tool_projection(rt, run_id, tenant=_TENANT_TOOL)
    assert resumed_proj == ref_proj
