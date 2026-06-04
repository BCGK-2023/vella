"""Graph-driven, edge-tied tool discovery: baseline seed + HAS_TOOL, idempotent.

Discovery = seeded baseline "system" tools (idempotent via ``upsert``) + per-run
tool-nodes linked by a ``HAS_TOOL`` edge (run -> tool), resolved by folding the
runtime into a graph view and querying ``neighbors`` with EXPLICIT direction. The
result is a deterministic sorted tuple of node ids; nothing is privileged. Fresh
``Runtime`` + fresh ``Registry`` per case; ``asyncio.run`` (no pytest-asyncio).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from vella.core import Node, ToolDeclaration, UnresolvedRef
from vella.runtime import Runtime

from vella.agent import (
    BuiltinBinding,
    RunData,
    ToolData,
    agent_registry,
    discover_tools,
    link_run_tool,
    seed_system_tools,
)
from vella.agent._discovery import SYSTEM_TOOL_PLUGIN
from vella.agent._writeback import create_run

_TENANT = "t-agent"
_ACTOR = UnresolvedRef(identifier="vella:test")


def _run(case: Callable[[Runtime], Any]) -> None:
    asyncio.run(asyncio.wait_for(case(Runtime()), timeout=5.0))


def _system_tool(name: str) -> ToolData:
    return ToolData(
        declaration=ToolDeclaration(name=name, description=f"{name} tool"),
        binding=BuiltinBinding(registry_key=name),
    )


def test_baseline_seed_then_has_tool_discovery() -> None:
    _run(_case_seed_and_discover)


async def _case_seed_and_discover(rt: Runtime) -> None:
    agent_registry()  # registers tool type into a fresh registry (isolation)
    run = await create_run(rt, RunData(goal="g"), name="run", tenant_id=_TENANT)

    baseline = await seed_system_tools(
        rt, [_system_tool("read"), _system_tool("write")], tenant_id=_TENANT
    )
    assert len(baseline) == 2

    # A per-run tool linked via HAS_TOOL (run -> tool).
    extra = Node.from_data(
        _system_tool("custom"), name="custom", created_by=_ACTOR, tenant_id=_TENANT
    )
    await rt.create(extra)
    await link_run_tool(rt, run.id, extra.id, tenant_id=_TENANT)

    discovered = await discover_tools(
        rt, run.id, tenant_id=_TENANT, baseline=baseline
    )
    # Baseline (2) + the one HAS_TOOL neighbour = 3, sorted by str(id).
    expected = tuple(sorted({b.id for b in baseline} | {extra.id}, key=str))
    assert discovered == expected
    assert len(discovered) == 3


def test_seeding_twice_is_idempotent() -> None:
    _run(_case_idempotent_seed)


async def _case_idempotent_seed(rt: Runtime) -> None:
    agent_registry()
    tools = [_system_tool("read"), _system_tool("write")]

    first = await seed_system_tools(rt, tools, tenant_id=_TENANT)
    second = await seed_system_tools(rt, tools, tenant_id=_TENANT)

    # Same node ids on reseed — upsert resolved the existing binding, no duplicates.
    assert {n.id for n in first} == {n.id for n in second}

    run = await create_run(rt, RunData(goal="g"), name="run", tenant_id=_TENANT)
    d1 = await discover_tools(rt, run.id, tenant_id=_TENANT, baseline=first)
    d2 = await discover_tools(rt, run.id, tenant_id=_TENANT, baseline=second)
    assert d1 == d2  # same discovered set


def test_seed_carries_idempotency_binding() -> None:
    _run(_case_binding_present)


async def _case_binding_present(rt: Runtime) -> None:
    agent_registry()
    [node] = await seed_system_tools(rt, [_system_tool("read")], tenant_id=_TENANT)
    got = await rt.get(_TENANT, node.id)
    assert got is not None
    bindings = [
        b for b in got.integrations
        if b.plugin == SYSTEM_TOOL_PLUGIN and b.external_id == "read"
    ]
    assert len(bindings) == 1


def test_discovery_is_sorted_and_has_no_privileged_tools() -> None:
    _run(_case_sorted_no_privileged)


async def _case_sorted_no_privileged(rt: Runtime) -> None:
    agent_registry()
    run = await create_run(rt, RunData(goal="g"), name="run", tenant_id=_TENANT)
    # No baseline, no links: an empty, deterministic toolset (no hidden internals).
    empty = await discover_tools(rt, run.id, tenant_id=_TENANT)
    assert empty == ()

    # With several HAS_TOOL links, the result is sorted by str(id).
    ids = []
    for name in ("a", "b", "c"):
        n = Node.from_data(
            _system_tool(name), name=name, created_by=_ACTOR, tenant_id=_TENANT
        )
        await rt.create(n)
        await link_run_tool(rt, run.id, n.id, tenant_id=_TENANT)
        ids.append(n.id)
    discovered = await discover_tools(rt, run.id, tenant_id=_TENANT)
    assert list(discovered) == sorted(ids, key=str)
