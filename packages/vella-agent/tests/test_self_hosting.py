"""Degenerate end-to-end self-hosting: a run materializes its cognition via verbs.

A run writes ``agent.run`` + ``agent.step`` + ``agent.message`` nodes THROUGH the
runtime's public verbs (``create`` / ``link``), and the reasoning trace is an
``observe_only`` telemetry entry. The nodes are retrievable via ``runtime.get()`` /
``history()``; every write is one public ``LogEntry`` whose transition is a known
verb (never a direct store mutation). Mirrors the runtime/graph test idiom: a fresh
``Runtime()`` over the in-memory store, a fresh isolated ``Registry()`` injected for
construction, async cases driven by ``asyncio.run`` (no pytest-asyncio).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from uuid import uuid4

from vella.agent import (
    MessageData,
    RunData,
    StepData,
    TextBlock,
    ToolCallData,
    agent_registry,
)
from vella.agent._writeback import (
    append_message,
    append_step,
    append_tool_call,
    create_run,
    emit_reasoning_trace,
)
from vella.core import Edge, EdgeTypes
from vella.runtime import Runtime

_TENANT = "t-agent"

# The set of write transitions a self-hosting run is allowed to use — exactly the
# public runtime verbs. A direct store mutation would never appear here, and the
# telemetry trace is observe_only (asserted separately).
_WRITE_VERBS = frozenset({"create", "link", "observe_only"})


def _run(case: Callable[[Runtime], Any]) -> None:
    asyncio.run(asyncio.wait_for(case(Runtime()), timeout=5.0))


def test_run_materializes_nodes_via_verbs() -> None:
    _run(_case_run_materializes_nodes_via_verbs)


async def _case_run_materializes_nodes_via_verbs(rt: Runtime) -> None:
    # Fresh isolated registry — never the global default. From M3 the tool contract's
    # types (tool / mcp_server / agent.tool_call) register into the same registry; M4
    # adds the `provider` node type (the cache-strategy capability the assembler reads);
    # M5 adds the `loop_policy` FSM-schema node type the interpreter reads.
    reg = agent_registry()
    assert reg.names() == [
        "agent.message",
        "agent.run",
        "agent.step",
        "agent.summary",
        "agent.tool_call",
        "loop_policy",
        "mcp_server",
        "provider",
        "tool",
    ]

    run = await create_run(
        rt, RunData(goal="prove self-hosting"), name="run-1", tenant_id=_TENANT
    )
    step = await append_step(
        rt, run.id, StepData(turn_index=0), name="step-0", tenant_id=_TENANT
    )
    msg = await append_message(
        rt,
        run.id,
        MessageData(role="user", content=(TextBlock(text="hello"),)),
        name="msg-0",
        tenant_id=_TENANT,
    )

    # Retrievable via runtime.get() — the state-table authority, not .payload.
    got_run = await rt.get(_TENANT, run.id)
    got_step = await rt.get(_TENANT, step.id)
    got_msg = await rt.get(_TENANT, msg.id)
    assert got_run is not None and got_run.type == "agent.run"
    assert got_step is not None and got_step.type == "agent.step"
    assert got_msg is not None and got_msg.type == "agent.message"

    # Entity comparison via model_dump(mode="json"), never == (core PrivateAttr).
    assert got_run.model_dump(mode="json") == run.model_dump(mode="json")
    assert got_step.model_dump(mode="json") == step.model_dump(mode="json")
    assert got_msg.model_dump(mode="json") == msg.model_dump(mode="json")


def test_step_and_message_are_part_of_run() -> None:
    _run(_case_part_of_run)


async def _case_part_of_run(rt: Runtime) -> None:
    run = await create_run(rt, RunData(goal="g"), name="run", tenant_id=_TENANT)
    step = await append_step(
        rt, run.id, StepData(turn_index=0), name="s", tenant_id=_TENANT
    )
    msg = await append_message(
        rt,
        run.id,
        MessageData(role="assistant", content=(TextBlock(text="t"),)),
        name="m",
        tenant_id=_TENANT,
    )

    # The step/message each have exactly one PART_OF edge pointing at the run.
    step_hist = await rt.history(_TENANT, step.id)
    assert [e.transition for e in step_hist] == ["create"]

    # Every entry in the whole run's log is a public write verb (no direct store).
    # The log holds exactly six entries (3 create + 2 link + 0 telemetry here is
    # five — the run/step/msg creates and the two PART_OF links); drain the bounded
    # historical slice by count so we never block on observe()'s live edge.
    expected = 5
    seen_links = 0
    collected = 0
    async for entry in rt.observe(since=None):
        assert entry.transition in _WRITE_VERBS, entry.transition
        if entry.transition == "link":
            seen_links += 1
            # A PART_OF link's reconstructed edge points child -> run.
            edge = await rt.get(_TENANT, entry.entity_id)
            assert isinstance(edge, Edge)
            assert edge.type == EdgeTypes.PART_OF
            assert edge.to_node_id == run.id
            assert edge.from_node_id in {step.id, msg.id}
        collected += 1
        if collected == expected:
            break
    assert seen_links == 2


def test_all_writes_are_public_verb_log_entries() -> None:
    _run(_case_all_writes_public)


async def _case_all_writes_public(rt: Runtime) -> None:
    run = await create_run(rt, RunData(goal="g"), name="run", tenant_id=_TENANT)
    await append_step(rt, run.id, StepData(turn_index=0), name="s", tenant_id=_TENANT)
    await append_message(
        rt,
        run.id,
        MessageData(role="user", content=(TextBlock(text="hi"),)),
        name="m",
        tenant_id=_TENANT,
    )
    await emit_reasoning_trace(
        rt, run.id, tenant_id=_TENANT, payload={"thought": "deliberating"}
    )

    transitions: list[str] = []
    async for entry in rt.observe(since=None):
        transitions.append(entry.transition)
        if entry.transition == "observe_only":
            break
    # create(run) + create(step) + link + create(msg) + link + observe_only.
    assert sorted(set(transitions)) == ["create", "link", "observe_only"]
    assert transitions.count("create") == 3
    assert transitions.count("link") == 2
    assert transitions.count("observe_only") == 1


def test_tool_call_node_lands_part_of_step() -> None:
    _run(_case_tool_call_lands)


async def _case_tool_call_lands(rt: Runtime) -> None:
    # M3 completes self-hosting: a tool call materializes an agent.tool_call node
    # (with its resolved hint + error_kind) PART_OF the step, via runtime verbs.
    run = await create_run(rt, RunData(goal="g"), name="run", tenant_id=_TENANT)
    step = await append_step(
        rt, run.id, StepData(turn_index=0), name="s", tenant_id=_TENANT
    )
    tool_ref = uuid4()
    call = await append_tool_call(
        rt,
        step.id,
        ToolCallData(
            tool_ref=tool_ref,
            args={"q": "vella"},
            intent="search for vella",
            result={"hits": 1},
            error_kind=None,
            hint="a hit means the query matched",
        ),
        name="call-0",
        tenant_id=_TENANT,
    )

    # Retrievable as a real agent.tool_call node via the state-table authority.
    got = await rt.get(_TENANT, call.id)
    assert got is not None and got.type == "agent.tool_call"
    assert got.model_dump(mode="json") == call.model_dump(mode="json")

    # The whole log is exactly: create(run), create(step), link(step->run),
    # create(call), link(call->step) — five public-verb entries, no direct store.
    seen_call_link = False
    collected = 0
    expected = 5
    async for entry in rt.observe(since=None):
        assert entry.transition in _WRITE_VERBS, entry.transition
        if entry.transition == "link":
            edge = await rt.get(_TENANT, entry.entity_id)
            assert isinstance(edge, Edge)
            assert edge.type == EdgeTypes.PART_OF
            if edge.from_node_id == call.id:
                assert edge.to_node_id == step.id
                seen_call_link = True
        collected += 1
        if collected == expected:
            break
    assert seen_call_link
