"""Token-budget cumulative resume across an interrupt (TRAP-1, Note A).

A run's cumulative token usage is DURABLE: it is folded from the run's
``observe_only`` reasoning-trace telemetry (each entry carries its turn's budgeted
``tokens``), never an in-memory counter. So a run interrupted BELOW its
``token_budget`` (by the ``max_steps`` backstop, not the budget) resumes by
re-folding the prior turns' tokens and halts ``max_tokens`` at the correct
CUMULATIVE total — exactly where an uninterrupted reference run of the same script
halts.

Mutation target: ``_resumed_progress`` (interpreter.py ~182-188) folding the
telemetry ``"tokens"``. If that fold returned 0 / dropped the prior segment's
tokens, the resumed run would under-count and halt later (or never on this script),
diverging from the reference — the non-vacuity guards below make that RED.

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

from _interp_helper import make_run

_TENANT = "t-resume-tokens"

# Each turn spends 4 budgeted tokens; budget is 10. So the run halts max_tokens at
# the boundary AFTER the 3rd turn (cumulative 12 >= 10) — two turns (8) are below
# budget, the third crosses it.
_PER_TURN = 4
_BUDGET = 10


def _turn(k: int) -> ScriptedTurn:
    return ScriptedTurn(
        blocks=(ScriptedText(text=f"turn-{k}"),),
        stop_reason="end_turn",
        usage=Usage(output_tokens=_PER_TURN),
    )


def _policy() -> LoopPolicy:
    # Only the HARD token_budget governs the halt — no stop_conditions, no step_budget.
    return LoopPolicy(token_budget=_BUDGET)


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


def test_token_budget_folds_cumulatively_across_an_interrupt() -> None:
    asyncio.run(asyncio.wait_for(_case(), timeout=20.0))


async def _case() -> None:
    agent_registry()

    # --- uninterrupted reference: one continuous run of the same script ---
    rt_ref = Runtime()
    ref_run = await make_run(rt_ref, _policy(), tenant_id=_TENANT)
    ref = await _drive(
        rt_ref, ref_run, MockProvider([_turn(k) for k in range(5)]), max_steps=10
    )
    assert ref.halt_reason == "max_tokens"
    # 3 turns of 4 tokens => cumulative 12 crosses the budget of 10 at the 3rd boundary.
    assert ref.steps == 3
    assert ref.tokens == 3 * _PER_TURN

    # --- interrupted run: SAME runtime, two segments ---
    rt = Runtime()
    run_id = await make_run(rt, _policy(), tenant_id=_TENANT)

    # Segment 1: interrupted by the backstop after ONE turn (max_steps=1), BELOW budget.
    seg1 = await _drive(rt, run_id, MockProvider([_turn(0)]), max_steps=1)
    # Non-vacuity: the interruption is the backstop, NOT the budget, and is below it —
    # so resume MUST re-fold the prior turn's tokens to reach the cumulative halt.
    assert seg1.halt_reason != "max_tokens"
    assert seg1.halt_reason == "max_steps"
    assert seg1.tokens < _BUDGET
    assert seg1.tokens == _PER_TURN  # one turn folded so far

    # Segment 2: FRESH interpreter run over the SAME run. It folds the prior turn's
    # tokens from the durable observe_only trace and continues until max_tokens.
    seg2 = await _drive(rt, run_id, MockProvider([_turn(1), _turn(2), _turn(3)]), max_steps=10)
    assert seg2.halt_reason == "max_tokens"

    # Resume-equivalence: the resumed run's final cumulative tokens AND steps EQUAL the
    # uninterrupted reference. A fold that returned 0 / dropped seg1's tokens would
    # under-count and halt later (more steps / more tokens) — diverging here.
    assert seg2.tokens == ref.tokens
    assert seg2.steps == ref.steps
