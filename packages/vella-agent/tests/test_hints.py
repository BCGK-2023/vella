"""Hint resolution: success result_hint; first matching error_hint; default fallback.

Resolution is order-sensitive: ``error_hints`` is tried first-match-wins against the
result's ``error_kind``. The resolved hint lands on the ``ToolResultBlock`` fed to the
model AND on the durable ``agent.tool_call`` node. ``asyncio.run`` (no pytest-asyncio).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from vella.core import Node, UnresolvedRef
from vella.runtime import Runtime

from vella.agent import (
    ErrorHint,
    RunData,
    StepData,
    ToolCallData,
    ToolHints,
    ToolResult,
    ToolResultBlock,
    agent_registry,
    resolve_hint,
)
from vella.agent._writeback import append_step, append_tool_call, create_run

_TENANT = "t-agent"

_HINTS = ToolHints(
    result_hint="the call succeeded; use the content",
    error_hints=(
        ErrorHint(match="RateLimited", hint="back off and retry later"),
        ErrorHint(match="NotFound", hint="the resource does not exist"),
    ),
    default_error_hint="an unclassified error occurred",
)


def test_success_resolves_result_hint() -> None:
    hint = resolve_hint(_HINTS, ToolResult(content={"ok": True}, is_error=False))
    assert hint == "the call succeeded; use the content"


def test_error_resolves_first_matching_entry() -> None:
    hint = resolve_hint(
        _HINTS, ToolResult(is_error=True, error_kind="NotFound")
    )
    assert hint == "the resource does not exist"


def test_error_match_is_order_preserving_first_wins() -> None:
    # Two entries could match the same kind; the FIRST in tuple order wins.
    hints = ToolHints(
        error_hints=(
            ErrorHint(match="Boom", hint="first"),
            ErrorHint(match="Boom", hint="second"),
        )
    )
    assert resolve_hint(hints, ToolResult(is_error=True, error_kind="Boom")) == "first"


def test_error_falls_back_to_default() -> None:
    hint = resolve_hint(
        _HINTS, ToolResult(is_error=True, error_kind="Unknown")
    )
    assert hint == "an unclassified error occurred"


def test_no_default_returns_none() -> None:
    hints = ToolHints(error_hints=())
    assert resolve_hint(hints, ToolResult(is_error=True, error_kind="x")) is None


def test_resolved_hint_lands_on_block_and_tool_call_node() -> None:
    asyncio.run(asyncio.wait_for(_case_hint_round_trip(Runtime()), timeout=5.0))


async def _case_hint_round_trip(rt: Runtime) -> None:
    agent_registry()
    result = ToolResult(is_error=True, error_kind="RateLimited")
    hint = resolve_hint(_HINTS, result)
    assert hint == "back off and retry later"

    # The resolved hint goes onto the ToolResultBlock fed back to the model...
    block = ToolResultBlock(
        tool_use_id="c1", content=result.content, is_error=result.is_error, hint=hint
    )
    assert block.hint == hint

    # ...AND is recorded on the durable agent.tool_call node.
    run = await create_run(rt, RunData(goal="g"), name="run", tenant_id=_TENANT)
    step = await append_step(
        rt, run.id, StepData(turn_index=0), name="s", tenant_id=_TENANT
    )
    call = await append_tool_call(
        rt,
        step.id,
        ToolCallData(
            tool_ref=uuid4(),
            args={},
            intent="do the thing",
            result=result.content,
            error_kind=result.error_kind,
            hint=hint,
        ),
        name="call-0",
        tenant_id=_TENANT,
    )
    got = await rt.get(_TENANT, call.id)
    assert isinstance(got, Node)
    assert isinstance(got.data, ToolCallData)
    assert got.data.hint == hint
    assert got.data.error_kind == "RateLimited"
