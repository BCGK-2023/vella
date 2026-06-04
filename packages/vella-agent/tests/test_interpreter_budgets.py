"""Budget halts: step_budget (off-by-one proven), token_budget HARD, compaction SOFT.

These prove the §2.1 budget knobs are non-vacuous and independently testable:

* ``step_budget`` halts at N — the run produces AT MOST N ``agent.step`` nodes (the
  off-by-one mutation (a): checking the budget AFTER the next turn would let an
  (N+1)th step land — this asserts the durable step-node count is exactly N).
* ``token_budget`` is a HARD halt: the moment cumulative usage reaches it the run
  stops with reason ``max_tokens`` (terminal — no compaction).
* ``compaction_threshold`` is a SOFT watermark: crossing it CONTINUES the loop (the
  assembler compacts on the next assemble); it never halts the run on its own.

No ``pytest-asyncio``: every case runs under ``asyncio.run`` with a bounded
``max_steps`` backstop and a :class:`~vella.agent.ManualClock`.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from vella.core import EdgeTypes, Node
from vella.graph import GraphProjection
from vella.runtime import Runtime

from vella.agent import (
    CompactionPolicy,
    GraphContextAssembler,
    InMemoryToolInvoker,
    LoopPolicy,
    ManualClock,
    MockProvider,
    RunResult,
    StepData,
    agent_registry,
    run,
)

from _interp_helper import make_run, text_turn

_TENANT = "t-budget"


async def _count_steps(rt: Runtime, run_id: UUID) -> int:
    view = await GraphProjection().fold(rt, _TENANT)
    neighbours = await view.neighbors(run_id, edge_type=EdgeTypes.PART_OF, direction="in")
    n = 0
    for nb in neighbours:
        node = await rt.get(_TENANT, nb.node_id)
        if isinstance(node, Node) and isinstance(node.data, StepData):
            n += 1
    return n


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


def test_step_budget_halts_at_n_no_off_by_one() -> None:
    asyncio.run(asyncio.wait_for(_case_step_budget(), timeout=10.0))


async def _case_step_budget() -> None:
    agent_registry()
    rt = Runtime()
    # NO stop condition fires on these end_turn turns, so ONLY the step budget can
    # stop the loop. step_budget=3; script MORE turns than the budget.
    run_id = await make_run(rt, LoopPolicy(step_budget=3), tenant_id=_TENANT)
    provider = MockProvider([text_turn(f"t{i}") for i in range(10)])
    result = await _drive(rt, run_id, provider, max_steps=20)
    assert result.halt_reason == "max_steps"
    # Off-by-one proof: AT MOST N step nodes — exactly 3, never 4.
    assert result.steps == 3
    assert await _count_steps(rt, run_id) == 3


def test_token_budget_is_a_hard_halt() -> None:
    asyncio.run(asyncio.wait_for(_case_token_budget(), timeout=10.0))


async def _case_token_budget() -> None:
    agent_registry()
    rt = Runtime()
    # token_budget=10; each turn spends 4 output tokens. After turn 3 cumulative=12>=10
    # => HARD halt max_tokens (no compaction_threshold — pure hard-budget path).
    run_id = await make_run(rt, LoopPolicy(token_budget=10), tenant_id=_TENANT)
    provider = MockProvider([text_turn(f"t{i}", tokens=4) for i in range(10)])
    result = await _drive(rt, run_id, provider, max_steps=20)
    assert result.halt_reason == "max_tokens"
    assert result.tokens >= 10
    assert result.steps == 3  # 4+4+4=12 trips on the third turn


def test_compaction_watermark_continues_not_halts() -> None:
    asyncio.run(asyncio.wait_for(_case_compaction(), timeout=10.0))


async def _case_compaction() -> None:
    agent_registry()
    rt = Runtime()
    # Soft watermark of 6, NO hard token_budget. 4 tokens/turn -> crossed after turn 2
    # (cumulative 8 >= 6) and the loop CONTINUES (soft never halts). With a 3-turn
    # script the bounded driver, NOT max_tokens, ends the run.
    run_id = await make_run(
        rt,
        LoopPolicy(compaction=CompactionPolicy(compaction_threshold=6)),
        tenant_id=_TENANT,
    )
    provider = MockProvider([text_turn(f"t{i}", tokens=4) for i in range(3)])
    result = await _drive(rt, run_id, provider, max_steps=3)
    assert result.tokens >= 6
    assert result.halt_reason != "max_tokens"
    assert result.halt_reason == "max_steps"  # the bounded backstop, not a token halt


def test_budgets_are_independently_testable() -> None:
    asyncio.run(asyncio.wait_for(_case_independent(), timeout=10.0))


async def _case_independent() -> None:
    # A token-only policy records max_tokens, never max_steps — the gates are orthogonal.
    agent_registry()
    rt = Runtime()
    run_id = await make_run(rt, LoopPolicy(token_budget=5), tenant_id=_TENANT)
    provider = MockProvider([text_turn(f"t{i}", tokens=5) for i in range(5)])
    result = await _drive(rt, run_id, provider, max_steps=20)
    assert result.halt_reason == "max_tokens"
    assert result.steps == 1  # first turn spends 5 >= 5
