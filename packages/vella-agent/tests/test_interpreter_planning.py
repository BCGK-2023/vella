"""planning: off / single / replan_on_failure are observably-distinct FSM tables.

* ``off`` — a straight loop; NO ``agent.step`` has ``kind=="planning"``.
* ``single`` — exactly ONE leading planning step (its own ``agent.step``, ``kind==
  "planning"``, NO tool invoked), then the loop. The planning turn COUNTS against
  ``step_budget`` (so N steps => <= N step nodes still holds).
* ``replan_on_failure`` — a NON-RETRYABLE tool error (``is_error`` after ``invoke`` —
  the invoker's own retries already exhausted) transitions the FSM back to a planning
  turn before continuing.

Mutation (b): collapsing ``single`` and ``off`` to the same table makes the
single-mode planning-step assertion go red. The three modes are proven to differ by
the durable ``StepData.kind`` sequence they materialize.

No ``pytest-asyncio``: ``asyncio.run`` + bounded ``max_steps`` + ManualClock.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vella.core import EdgeTypes, Node, ToolDeclaration, UnresolvedRef
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

from _interp_helper import make_run, text_turn, tool_turn

_TENANT = "t-plan"
_ACTOR = UnresolvedRef(identifier="vella:test")


async def _step_kinds(rt: Runtime, run_id: Any) -> list[str]:
    """The durable StepData.kind sequence, in turn order (by step id)."""
    view = await GraphProjection().fold(rt, _TENANT)
    neighbours = await view.neighbors(run_id, edge_type=EdgeTypes.PART_OF, direction="in")
    steps: list[Node[Any, Any]] = []
    for nb in neighbours:
        node = await rt.get(_TENANT, nb.node_id)
        if isinstance(node, Node) and isinstance(node.data, StepData):
            steps.append(node)
    steps.sort(key=lambda n: (n.data.turn_index, str(n.id)))
    return [n.data.kind for n in steps]


def test_off_never_plans() -> None:
    asyncio.run(asyncio.wait_for(_case_off(), timeout=10.0))


async def _case_off() -> None:
    agent_registry()
    rt = Runtime()
    run_id = await make_run(
        rt,
        LoopPolicy(planning="off", stop_conditions=("no_tool_calls",)),
        tenant_id=_TENANT,
    )
    provider = MockProvider([text_turn("answer")])
    await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=5,
    )
    kinds = await _step_kinds(rt, run_id)
    assert kinds == ["turn"]
    assert "planning" not in kinds


def test_single_plans_once_first_and_counts_as_a_step() -> None:
    asyncio.run(asyncio.wait_for(_case_single(), timeout=10.0))


async def _case_single() -> None:
    agent_registry()
    rt = Runtime()
    # single + no_tool_calls. The FIRST step is the planning turn (no tools); the
    # SECOND is the working turn that ends via no_tool_calls.
    run_id = await make_run(
        rt,
        LoopPolicy(planning="single", stop_conditions=("no_tool_calls",)),
        tenant_id=_TENANT,
    )
    provider = MockProvider(
        [
            text_turn("here is my plan"),  # planning turn (recorded, no tools invoked)
            text_turn("the answer"),  # working turn -> no_tool_calls ends the run
        ]
    )
    result = await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=5,
    )
    kinds = await _step_kinds(rt, run_id)
    # Distinct from `off`: the leading step is a planning step (mutation (b)).
    assert kinds == ["planning", "turn"]
    assert result.steps == 2


def test_single_planning_counts_against_step_budget() -> None:
    asyncio.run(asyncio.wait_for(_case_single_budget(), timeout=10.0))


async def _case_single_budget() -> None:
    agent_registry()
    rt = Runtime()
    # step_budget=1 with single planning: the planning turn IS the one allowed step,
    # so the budget halts right after it — proving the planning turn counts.
    run_id = await make_run(
        rt, LoopPolicy(planning="single", step_budget=1), tenant_id=_TENANT
    )
    provider = MockProvider([text_turn("plan"), text_turn("never")])
    result = await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=5,
    )
    assert result.halt_reason == "max_steps"
    assert result.steps == 1
    assert await _step_kinds(rt, run_id) == ["planning"]


def test_replan_on_failure_transitions_to_planning_after_nonretryable() -> None:
    asyncio.run(asyncio.wait_for(_case_replan(), timeout=10.0))


async def _case_replan() -> None:
    agent_registry()
    rt = Runtime()
    run_id = await make_run(
        rt,
        LoopPolicy(planning="replan_on_failure", stop_conditions=("no_tool_calls",)),
        tenant_id=_TENANT,
    )
    failing_tool = ToolData(
        declaration=ToolDeclaration(name="flaky", description="fails"),
        binding=BuiltinBinding(registry_key="flaky"),
    )
    [node] = await seed_system_tools(rt, [failing_tool], tenant_id=_TENANT)
    await link_run_tool(rt, run_id, node.id, tenant_id=_TENANT)

    async def _always_errors(args: dict[str, Any]) -> ToolResult:
        # The invoker surfaces is_error=True AFTER its own (exhausted) retries; that is
        # the loop-level non-retryable predicate.
        return ToolResult(content="boom", is_error=True, error_kind="BoomError")

    invoker = InMemoryToolInvoker({"flaky": _always_errors}, clock=ManualClock())

    provider = MockProvider(
        [
            tool_turn(tool_id="c1", name="flaky", args={}, intent="try flaky."),  # turn 0: fails
            text_turn("replanned approach"),  # turn 1: the REPLAN planning turn
            text_turn("final answer"),  # turn 2: working turn -> no_tool_calls ends
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
        max_steps=6,
    )
    kinds = await _step_kinds(rt, run_id)
    # The failing tool turn is a normal `turn`; the NEXT step is a `planning` turn
    # (the replan), then a working `turn`. Distinct from off (no planning) and from
    # single (planning is LEADING, not after a failure).
    assert kinds == ["turn", "planning", "turn"]
    assert result.status == "succeeded"
