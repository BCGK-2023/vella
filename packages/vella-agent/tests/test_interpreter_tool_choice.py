"""tool_choice: model offers all; forced requires a call; restricted filters + rejects.

* ``model`` — all discovered tools are offered; the model may or may not call one.
* ``forced`` — the provider param requires a tool call; a no-tool turn is a violation
  (the run fails with reason ``refusal``).
* ``restricted(types)`` — the offered set is FILTERED to declarations whose name is in
  ``types`` BEFORE the request (mutation (c): the filter is pre-request, not
  post-turn), AND a ``tool_use`` naming a tool OUTSIDE the set is REJECTED.

No ``pytest-asyncio``: ``asyncio.run`` + bounded ``max_steps`` + ManualClock.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vella.core import Node, ToolDeclaration, UnresolvedRef
from vella.runtime import Runtime

from vella.agent import (
    BuiltinBinding,
    GraphContextAssembler,
    InMemoryToolInvoker,
    LoopPolicy,
    ManualClock,
    MockProvider,
    ScriptedText,
    ScriptedToolUse,
    ScriptedTurn,
    ToolChoiceForced,
    ToolChoiceRestricted,
    ToolData,
    ToolResult,
    Usage,
    agent_registry,
    assistant_turn_from_blocks,
    link_run_tool,
    run,
    seed_system_tools,
)

from _interp_helper import make_run, tool_turn

_TENANT = "t-tc"
_ACTOR = UnresolvedRef(identifier="vella:test")


def _tool(name: str) -> ToolData:
    return ToolData(
        declaration=ToolDeclaration(name=name, description=f"the {name} tool"),
        binding=BuiltinBinding(registry_key=name),
    )


async def _impl(args: dict[str, Any]) -> ToolResult:
    return ToolResult(content={"ok": True})


async def _seed_and_link(rt: Runtime, run_id: Any, names: list[str]) -> None:
    nodes = await seed_system_tools(rt, [_tool(n) for n in names], tenant_id=_TENANT)
    for node in nodes:
        await link_run_tool(rt, run_id, node.id, tenant_id=_TENANT)


def _invoker() -> InMemoryToolInvoker:
    return InMemoryToolInvoker(
        {"alpha": _impl, "beta": _impl}, clock=ManualClock()
    )


def test_forced_satisfied_by_a_tool_call_proceeds() -> None:
    asyncio.run(asyncio.wait_for(_case_forced_ok(), timeout=10.0))


async def _case_forced_ok() -> None:
    # forced is honoured end-to-end: a turn that DOES call a tool satisfies the
    # provider param and the interpreter proceeds (records the call + result).
    agent_registry()
    rt = Runtime()
    run_id = await make_run(
        rt,
        LoopPolicy(tool_choice=ToolChoiceForced(), step_budget=1),
        tenant_id=_TENANT,
    )
    await _seed_and_link(rt, run_id, ["alpha"])
    provider = MockProvider(
        [tool_turn(tool_id="c1", name="alpha", args={}, intent="do alpha.")]
    )
    result = await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=_invoker(),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=2,
    )
    # The forced turn produced a call and the step budget then halted the run cleanly.
    assert result.status == "succeeded"
    assert result.halt_reason == "max_steps"
    assert result.steps == 1


def test_forced_rejects_a_no_tool_turn() -> None:
    asyncio.run(asyncio.wait_for(_case_forced_reject(), timeout=10.0))


async def _case_forced_reject() -> None:
    # A misbehaving provider returns a no-tool turn even though tool_choice=forced was
    # requested (the MockProvider enforces the contract, so use a minimal provider that
    # IGNORES the param). The INTERPRETER must reject the no-tool turn => failed run.
    agent_registry()
    rt = Runtime()
    run_id = await make_run(
        rt, LoopPolicy(tool_choice=ToolChoiceForced()), tenant_id=_TENANT
    )
    await _seed_and_link(rt, run_id, ["alpha"])

    class _IgnoresForced:
        """A provider that returns a no-tool turn regardless of the forced param."""

        async def turn(self, request: Any) -> Any:
            return assistant_turn_from_blocks(
                (ScriptedText(text="ignoring forced"),), stop_reason="end_turn"
            )

        def stream(self, request: Any) -> Any:  # pragma: no cover - turn() is used
            raise NotImplementedError

    result = await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=_IgnoresForced(),
        invoker=_invoker(),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=2,
    )
    assert result.status == "failed"
    assert result.halt_reason == "refusal"


def test_restricted_filters_offered_tools() -> None:
    asyncio.run(asyncio.wait_for(_case_restricted_filter(), timeout=10.0))


async def _case_restricted_filter() -> None:
    agent_registry()
    rt = Runtime()
    captured: dict[str, list[str]] = {}

    run_id = await make_run(
        rt,
        LoopPolicy(tool_choice=ToolChoiceRestricted(types=("alpha",))),
        tenant_id=_TENANT,
    )
    await _seed_and_link(rt, run_id, ["alpha", "beta"])

    # A provider that records the tool names it was OFFERED, then calls the allowed one.
    class _RecordingProvider(MockProvider):
        async def turn(self, request: Any) -> Any:
            captured["offered"] = [t.name for t in request.tools]
            return await super().turn(request)

    provider = _RecordingProvider(
        [tool_turn(tool_id="c1", name="alpha", args={}, intent="alpha.")]
    )
    await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=_invoker(),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=1,
    )
    # 'beta' was discovered+linked but FILTERED out before the request (mutation (c)).
    assert captured["offered"] == ["alpha"]


def test_restricted_rejects_out_of_set_call() -> None:
    asyncio.run(asyncio.wait_for(_case_restricted_reject(), timeout=10.0))


async def _case_restricted_reject() -> None:
    agent_registry()
    rt = Runtime()
    run_id = await make_run(
        rt,
        LoopPolicy(tool_choice=ToolChoiceRestricted(types=("alpha",))),
        tenant_id=_TENANT,
    )
    await _seed_and_link(rt, run_id, ["alpha", "beta"])
    # The model (mis)behaves and calls 'beta' — outside the restricted set. Even though
    # the provider could name it, the interpreter REJECTS the out-of-set call
    # (mutation (c): passing it through would breach the restriction).
    provider = MockProvider(
        [tool_turn(tool_id="c1", name="beta", args={}, intent="beta.")]
    )
    result = await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=_invoker(),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=1,
    )
    assert result.status == "failed"
    assert result.halt_reason == "refusal"


def test_model_offers_all_discovered_tools() -> None:
    asyncio.run(asyncio.wait_for(_case_model(), timeout=10.0))


async def _case_model() -> None:
    agent_registry()
    rt = Runtime()
    captured: dict[str, list[str]] = {}
    run_id = await make_run(rt, LoopPolicy(), tenant_id=_TENANT)  # default tool_choice=model
    await _seed_and_link(rt, run_id, ["alpha", "beta"])

    class _RecordingProvider(MockProvider):
        async def turn(self, request: Any) -> Any:
            captured["offered"] = sorted(t.name for t in request.tools)
            return await super().turn(request)

    provider = _RecordingProvider(
        [
            ScriptedTurn(
                blocks=(ScriptedText(text="hi"),), stop_reason="end_turn", usage=Usage()
            )
        ]
    )
    await run(
        rt,
        run_id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=_invoker(),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=1,
    )
    assert captured["offered"] == ["alpha", "beta"]
