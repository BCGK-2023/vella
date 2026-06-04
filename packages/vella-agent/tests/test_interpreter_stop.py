"""Stop conditions: no_tool_calls ends; refusal ends; sorted-first reason recorded.

* ``no_tool_calls`` — an assistant ``end_turn`` with zero tool calls ends the run.
* ``refusal`` — a ``stop_reason=="refusal"`` turn ends the run.
* When several conditions could fire on one turn, the FIRST in SORTED order is the
  recorded halt reason (deterministic — the policy stores ``stop_conditions`` sorted).

No ``pytest-asyncio``: ``asyncio.run`` + bounded ``max_steps`` + ManualClock.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from vella.runtime import Runtime

from vella.agent import (
    GraphContextAssembler,
    InMemoryToolInvoker,
    LoopPolicy,
    ManualClock,
    MockProvider,
    RunResult,
    ScriptedText,
    ScriptedTurn,
    Usage,
    agent_registry,
    run,
)

from _interp_helper import make_run, text_turn

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
