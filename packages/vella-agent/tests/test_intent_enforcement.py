"""require_tool_intent: an intent-less tool_use is rejected; a 1-sentence one passes.

Mutation (d): accepting an intent-less ``ToolUseBlock`` under ``require_tool_intent``
makes the rejection assertion go red. A present, ``<= 1``-sentence intent is the
UX-legibility contract; a missing/empty one (or a multi-sentence one) is a policy
violation that fails the run.

No ``pytest-asyncio``: ``asyncio.run`` + bounded ``max_steps`` + ManualClock.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from vella.core import ToolDeclaration, UnresolvedRef
from vella.runtime import Runtime

from vella.agent import (
    BuiltinBinding,
    GraphContextAssembler,
    InMemoryToolInvoker,
    LoopPolicy,
    ManualClock,
    MockProvider,
    RunResult,
    ScriptedTurn,
    ToolData,
    ToolResult,
    agent_registry,
    link_run_tool,
    run,
    seed_system_tools,
)

from _interp_helper import make_run, tool_turn

_TENANT = "t-intent"
_ACTOR = UnresolvedRef(identifier="vella:test")


async def _impl(args: dict[str, Any]) -> ToolResult:
    return ToolResult(content={"ok": True})


async def _setup(rt: Runtime, require_intent: bool) -> UUID:
    run_id = await make_run(
        rt, LoopPolicy(require_tool_intent=require_intent, stop_conditions=("no_tool_calls",)), tenant_id=_TENANT
    )
    tool = ToolData(
        declaration=ToolDeclaration(name="act", description="act"),
        binding=BuiltinBinding(registry_key="act"),
    )
    [node] = await seed_system_tools(rt, [tool], tenant_id=_TENANT)
    await link_run_tool(rt, run_id, node.id, tenant_id=_TENANT)
    return run_id


async def _drive(
    rt: Runtime, run_id: UUID, provider: MockProvider, *, max_steps: int
) -> RunResult:
    return await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker({"act": _impl}, clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=max_steps,
    )


def test_intent_less_call_is_rejected() -> None:
    asyncio.run(asyncio.wait_for(_case_rejected(), timeout=10.0))


async def _case_rejected() -> None:
    agent_registry()
    rt = Runtime()
    run_id = await _setup(rt, require_intent=True)
    provider = MockProvider(
        [tool_turn(tool_id="c1", name="act", args={}, intent="")]  # empty intent
    )
    result = await _drive(rt, run_id, provider, max_steps=2)
    assert result.status == "failed"
    assert result.halt_reason == "refusal"


def test_one_sentence_intent_passes() -> None:
    asyncio.run(asyncio.wait_for(_case_passes(), timeout=10.0))


async def _case_passes() -> None:
    agent_registry()
    rt = Runtime()
    run_id = await _setup(rt, require_intent=True)
    provider = MockProvider(
        [
            tool_turn(tool_id="c1", name="act", args={}, intent="Turn on the device."),
            # after the tool result, a no-tool turn ends the run cleanly.
            ScriptedTurn(blocks=(), stop_reason="end_turn"),
        ]
    )
    result = await _drive(rt, run_id, provider, max_steps=3)
    assert result.status == "succeeded"
    assert result.halt_reason == "no_tool_calls"


def test_multi_sentence_intent_is_rejected() -> None:
    asyncio.run(asyncio.wait_for(_case_multi(), timeout=10.0))


async def _case_multi() -> None:
    agent_registry()
    rt = Runtime()
    run_id = await _setup(rt, require_intent=True)
    provider = MockProvider(
        [tool_turn(tool_id="c1", name="act", args={}, intent="Do this. Then do that.")]
    )
    result = await _drive(rt, run_id, provider, max_steps=2)
    assert result.status == "failed"
    assert result.halt_reason == "refusal"


def test_intent_not_required_when_knob_off() -> None:
    asyncio.run(asyncio.wait_for(_case_off(), timeout=10.0))


async def _case_off() -> None:
    agent_registry()
    rt = Runtime()
    run_id = await _setup(rt, require_intent=False)
    provider = MockProvider(
        [
            tool_turn(tool_id="c1", name="act", args={}, intent=""),  # no intent, allowed
            ScriptedTurn(blocks=(), stop_reason="end_turn"),
        ]
    )
    result = await _drive(rt, run_id, provider, max_steps=3)
    assert result.status == "succeeded"
