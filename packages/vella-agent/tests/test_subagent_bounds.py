"""Bounded sub-agents (M6, §2.2): runaway provably impossible — cardinality AND cost.

An ADVERSARIAL :class:`~vella.agent.MockProvider` requests a spawn on EVERY turn at
EVERY level (the script is an unbounded stream of spawn-requesting turns). The
load-bearing assertions:

* (i) the materialized run-tree node count NEVER exceeds the closed-form bound
  ``N_max = Σ_{i=0..max_depth} max_fanout^i`` (:func:`~vella.agent.max_run_tree_size`);
* (ii) AGGREGATE token spend across the WHOLE tree NEVER exceeds
  ``N_max * per_run_token_budget`` — each child has its OWN budget (no pooling).

Plus the two pre-spawn-gate refusals, each confirmed VIA THE GRAPH (no child node, no
PART_OF edge written):

* a DEPTH-breaching spawn is refused;
* a FANOUT-breaching spawn is refused.

Mutation pre-checks this file makes RED when the impl is broken: (a) depth from an
in-memory counter (replay/adversarial bound), (b) create-then-check (phantom
over-spawn), (c) dropped fanout check (closed-form tree size), (e) ``direction="both"``
for depth, (f) pooled shared budget (aggregate-cost bound).

No ``pytest-asyncio``: ``asyncio.run`` + bounded ``max_steps`` + ManualClock.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from vella.core import EdgeTypes, Node
from vella.graph import GraphProjection
from vella.runtime import Runtime

from vella.agent import (
    GraphContextAssembler,
    InMemoryToolInvoker,
    LoopPolicy,
    ManualClock,
    MockProvider,
    RunData,
    ScriptedToolUse,
    ScriptedTurn,
    StepData,
    SubAgentAllow,
    SubAgentDeny,
    Usage,
    agent_registry,
    max_run_tree_size,
    run,
)
from vella.agent._subagent import SPAWN_TOOL, gate_allows_spawn, spawn_child

from _interp_helper import make_run

_TENANT = "t-subagent"

# Each turn spends this many output tokens — the per-run token budget multiplicand in
# the aggregate cost bound. Small + fixed so the arithmetic is exact.
_TOKENS_PER_TURN = 3


def _spawn_turn(child_token_budget: int) -> ScriptedTurn:
    """A turn that REQUESTS a sub-agent spawn (the reserved SPAWN_TOOL affordance).

    The child's OWN budget is carried in the spawn input (no pooling). ``stop_reason``
    is ``tool_use`` because the turn emits a tool_use block.
    """
    return ScriptedTurn(
        blocks=(
            ScriptedToolUse(
                id="spawn-1",
                name=SPAWN_TOOL,
                input={"goal": "child", "token_budget": child_token_budget},
                intent="spawn a child sub-agent.",
            ),
        ),
        stop_reason="tool_use",
        usage=Usage(output_tokens=_TOKENS_PER_TURN),
    )


async def _run_tree_from(rt: Runtime, root: UUID) -> list[UUID]:
    """All ``agent.run`` ids reachable from ``root`` down the PART_OF tree (incl root)."""
    view = await GraphProjection().fold(rt, _TENANT)
    out: list[UUID] = []
    seen: set[UUID] = set()
    stack = [root]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = await rt.get(_TENANT, node_id)
        if not (isinstance(node, Node) and isinstance(node.data, RunData)):
            continue
        out.append(node_id)
        children = await view.neighbors(
            node_id, edge_type=EdgeTypes.PART_OF, direction="in"
        )
        for child in children:
            child_node = await rt.get(_TENANT, child.node_id)
            if isinstance(child_node, Node) and isinstance(child_node.data, RunData):
                stack.append(child.node_id)
    return out


async def _aggregate_tokens(rt: Runtime, run_ids: list[UUID]) -> int:
    """Total budgeted tokens folded across every run in the tree (durable trace)."""
    total = 0
    for run_id in run_ids:
        for entry in await rt.history(_TENANT, run_id):
            if entry.transition == "observe_only":
                value = entry.payload.get("tokens")
                if isinstance(value, int):
                    total += value
    return total


def test_spawn_every_turn_never_exceeds_cardinality_or_cost() -> None:
    asyncio.run(asyncio.wait_for(_case_adversarial(), timeout=30.0))


async def _case_adversarial() -> None:
    agent_registry()
    rt = Runtime()

    max_depth, max_fanout = 2, 2
    # Each run has its OWN token_budget (no pooling): with _TOKENS_PER_TURN spend/turn a
    # run halts max_tokens after ceil(budget/tokens) turns. The gate — NOT the budget —
    # is what caps SPAWNING; the budget caps each run's turns so the shared script is
    # finite. Aggregate cost bound = N_max * per_run_token_budget.
    per_run_token_budget = 30  # ~10 turns/run at 3 tokens/turn
    n_max = max_run_tree_size(max_depth, max_fanout)  # 1 + 2 + 4 = 7

    root_policy = LoopPolicy(
        token_budget=per_run_token_budget,
        sub_agent_spawn=SubAgentAllow(max_depth=max_depth, max_fanout=max_fanout),
    )
    root = await make_run(rt, root_policy, tenant_id=_TENANT)

    # ADVERSARIAL: a long stream of spawn-requesting turns, far more than the whole tree
    # could ever consume (every run, every turn, requests a spawn) — so the ONLY thing
    # that stops runaway spawning is the pre-spawn graph gate, not script exhaustion.
    provider = MockProvider([_spawn_turn(per_run_token_budget) for _ in range(2000)])

    await run(
        rt,
        root,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=50,
    )

    run_ids = await _run_tree_from(rt, root)
    # (i) cardinality bound: the materialized run-tree NEVER exceeds the closed form.
    assert len(run_ids) <= n_max, f"{len(run_ids)} runs > N_max={n_max}"

    # (ii) aggregate cost bound: total tokens across the whole tree <= N_max * budget.
    agg = await _aggregate_tokens(rt, run_ids)
    assert agg <= n_max * per_run_token_budget, (
        f"aggregate {agg} > N_max*budget={n_max * per_run_token_budget}"
    )
    # Sanity: spawning actually happened (more than just the root materialized).
    assert len(run_ids) > 1


def test_depth_breaching_spawn_refused_no_child_node_or_edge() -> None:
    asyncio.run(asyncio.wait_for(_case_depth_refusal(), timeout=20.0))


async def _case_depth_refusal() -> None:
    agent_registry()
    rt = Runtime()
    # max_depth=1: the root (depth 0) may spawn a depth-1 child, but that child may NOT
    # spawn a depth-2 grandchild — the depth bound refuses it.
    allow = SubAgentAllow(max_depth=1, max_fanout=5)
    root = await make_run(
        rt, LoopPolicy(sub_agent_spawn=allow), tenant_id=_TENANT
    )

    # Manually create a depth-1 child (a real PART_OF link) to set up the breach.
    child_id = await spawn_child(
        rt,
        root,
        tenant_id=_TENANT,
        goal="depth-1-child",
        child_policy_ref=None,
        provider_ref=None,
    )
    assert child_id is not None

    # The gate must REFUSE a spawn UNDER the depth-1 child (it would be depth 2 > 1).
    allowed = await gate_allows_spawn(rt, child_id, tenant_id=_TENANT, allow=allow)
    assert allowed is False

    # Confirm VIA THE GRAPH that the child has NO children (no phantom node/edge).
    grandchildren = await _children_of(rt, child_id)
    assert grandchildren == []


def test_fanout_breaching_spawn_refused_no_child_node_or_edge() -> None:
    asyncio.run(asyncio.wait_for(_case_fanout_refusal(), timeout=20.0))


async def _case_fanout_refusal() -> None:
    agent_registry()
    rt = Runtime()
    # max_fanout=1: the root may have ONE direct child; a second is refused.
    allow = SubAgentAllow(max_depth=5, max_fanout=1)
    root = await make_run(
        rt, LoopPolicy(sub_agent_spawn=allow), tenant_id=_TENANT
    )

    # First child fills the fanout budget.
    first = await spawn_child(
        rt, root, tenant_id=_TENANT, goal="c1", child_policy_ref=None, provider_ref=None
    )
    assert first is not None

    # A SECOND direct child would make fanout 2 > 1 — the gate refuses it.
    allowed = await gate_allows_spawn(rt, root, tenant_id=_TENANT, allow=allow)
    assert allowed is False

    # The root still has EXACTLY one child (no phantom second child written).
    assert len(await _children_of(rt, root)) == 1


def test_deny_policy_makes_spawn_request_a_violation() -> None:
    asyncio.run(asyncio.wait_for(_case_deny(), timeout=20.0))


async def _case_deny() -> None:
    agent_registry()
    rt = Runtime()
    # The default policy denies spawning; a spawn request is a policy violation.
    root = await make_run(
        rt, LoopPolicy(sub_agent_spawn=SubAgentDeny()), tenant_id=_TENANT
    )
    provider = MockProvider([_spawn_turn(100)])
    result = await run(
        rt,
        root,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=10,
    )
    assert result.status == "failed"
    assert result.halt_reason == "refusal"
    # No child run materialized under a deny policy.
    assert await _children_of(rt, root) == []


async def _children_of(rt: Runtime, parent: UUID) -> list[UUID]:
    """The direct child ``agent.run`` ids of ``parent`` (graph read, explicit dir)."""
    view = await GraphProjection().fold(rt, _TENANT)
    children = await view.neighbors(
        parent, edge_type=EdgeTypes.PART_OF, direction="in"
    )
    out: list[UUID] = []
    for child in children:
        node = await rt.get(_TENANT, child.node_id)
        if isinstance(node, Node) and isinstance(node.data, RunData):
            out.append(child.node_id)
    return sorted(out, key=str)


def test_parent_depth_counts_only_upward_chain_not_descendants() -> None:
    asyncio.run(asyncio.wait_for(_case_parent_depth_direction(), timeout=20.0))


async def _case_parent_depth_direction() -> None:
    """``_parent_depth`` must count ONLY the upward PART_OF chain (direction='out').

    Topology: root <- a <- b <- c (a 3-deep chain UNDER root via PART_OF edges
    child->parent). ``_parent_depth(a)`` is exactly 1 (only root is above a) — even
    though ``a`` has a DEEP subtree below it (b, c). A ``direction='both'`` walk
    (mutation (e)) would descend into that subtree and miscount the depth, breaking the
    bound; this pins the correct upward-only count.
    """
    from vella.agent._subagent import _parent_depth  # noqa: PLC0415

    agent_registry()
    rt = Runtime()
    allow = SubAgentAllow(max_depth=10, max_fanout=10)
    root = await make_run(
        rt, LoopPolicy(sub_agent_spawn=allow), tenant_id=_TENANT
    )
    a = await spawn_child(
        rt, root, tenant_id=_TENANT, goal="a", child_policy_ref=None, provider_ref=None
    )
    assert a is not None
    b = await spawn_child(
        rt, a, tenant_id=_TENANT, goal="b", child_policy_ref=None, provider_ref=None
    )
    assert b is not None
    c = await spawn_child(
        rt, b, tenant_id=_TENANT, goal="c", child_policy_ref=None, provider_ref=None
    )
    assert c is not None

    view = await GraphProjection().fold(rt, _TENANT)
    # root is depth 0; a is depth 1; b is depth 2; c is depth 3 — upward chain ONLY.
    assert await _parent_depth(view, root) == 0
    assert await _parent_depth(view, a) == 1
    assert await _parent_depth(view, b) == 2
    assert await _parent_depth(view, c) == 3


def test_n_max_closed_form() -> None:
    # The closed-form bound is Σ f^i for i in 0..d.
    assert max_run_tree_size(0, 5) == 1  # just the root
    assert max_run_tree_size(1, 1) == 2  # root + 1 child
    assert max_run_tree_size(2, 2) == 7  # 1 + 2 + 4
    assert max_run_tree_size(3, 3) == 1 + 3 + 9 + 27
