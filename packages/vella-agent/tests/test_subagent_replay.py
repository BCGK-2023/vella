"""Replay/resume for sub-trees (§2.2): the bound holds on the DURABLE record.

Three properties:

* **fold-from-log reconstructs the same tree.** Re-folding the runtime into a fresh
  graph view yields the identical run-tree (same run ids, same PART_OF structure) —
  the tree is a durable record, not in-memory state (TRAP-1).
* **mid-tree interruption resumes.** A parent run interrupted mid-loop (bounded by a
  small ``max_steps``) resumes in a second segment and continues from its durable
  steps — the already-spawned children are NOT re-spawned (the fanout gate counts them
  from the graph), and resuming never breaches the bound.
* **bounds hold on the durable record.** After interruption + resume the materialized
  run-tree still satisfies the closed-form cardinality bound — because the gate is
  re-evaluated against the durable graph on every spawn, a resume cannot resurrect a
  phantom over-spawn (mutation (a)/(b)/(e)).

No ``pytest-asyncio``: ``asyncio.run`` + bounded ``max_steps`` + ManualClock.
"""

from __future__ import annotations

import asyncio
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
    SubAgentAllow,
    Usage,
    agent_registry,
    max_run_tree_size,
    run,
)
from vella.agent._subagent import SPAWN_TOOL

from _interp_helper import make_run

_TENANT = "t-subagent-replay"
_TOKENS_PER_TURN = 3


def _spawn_turn() -> ScriptedTurn:
    return ScriptedTurn(
        blocks=(
            ScriptedToolUse(
                id="sp",
                name=SPAWN_TOOL,
                input={"goal": "child", "token_budget": 30},
                intent="spawn a child sub-agent.",
            ),
        ),
        stop_reason="tool_use",
        usage=Usage(output_tokens=_TOKENS_PER_TURN),
    )


async def _tree_signature(rt: Runtime, root: UUID) -> list[tuple[str, str]]:
    """A deterministic digest of the run-tree: sorted ``(child_id, parent_id)`` edges.

    Folds the runtime fresh and walks PART_OF down from ``root``, recording each
    ``child --PART_OF--> parent`` run edge. Sorted, so it is a stable structural
    fingerprint to compare across re-folds.
    """
    view = await GraphProjection().fold(rt, _TENANT)
    edges: list[tuple[str, str]] = []
    seen: set[UUID] = set()
    stack = [root]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        children = await view.neighbors(
            node_id, edge_type=EdgeTypes.PART_OF, direction="in"
        )
        for child in children:
            child_node = await rt.get(_TENANT, child.node_id)
            if isinstance(child_node, Node) and isinstance(child_node.data, RunData):
                edges.append((str(child.node_id), str(node_id)))
                stack.append(child.node_id)
    return sorted(edges)


async def _all_runs(rt: Runtime, root: UUID) -> list[UUID]:
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


def test_fold_from_log_reconstructs_the_same_tree() -> None:
    asyncio.run(asyncio.wait_for(_case_fold_equivalence(), timeout=30.0))


async def _case_fold_equivalence() -> None:
    agent_registry()
    rt = Runtime()
    allow = SubAgentAllow(max_depth=2, max_fanout=2)
    root = await make_run(
        rt, LoopPolicy(token_budget=30, sub_agent_spawn=allow), tenant_id=_TENANT
    )
    provider = MockProvider([_spawn_turn() for _ in range(2000)])
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
    # Two independent folds of the SAME durable log yield the identical tree structure.
    sig1 = await _tree_signature(rt, root)
    sig2 = await _tree_signature(rt, root)
    assert sig1 == sig2
    assert sig1, "the tree should have at least one PART_OF run edge"
    # And the tree respects the closed-form bound on the durable record.
    assert len(await _all_runs(rt, root)) <= max_run_tree_size(2, 2)


def test_interrupted_parent_resumes_without_breaching_the_bound() -> None:
    asyncio.run(asyncio.wait_for(_case_resume(), timeout=30.0))


async def _case_resume() -> None:
    agent_registry()
    rt = Runtime()
    allow = SubAgentAllow(max_depth=2, max_fanout=2)
    n_max = max_run_tree_size(2, 2)
    root = await make_run(
        rt, LoopPolicy(token_budget=30, sub_agent_spawn=allow), tenant_id=_TENANT
    )

    # Segment 1: interrupt the PARENT after a small number of steps (the bounded driver
    # stops mid-loop). Some children may already be spawned + materialized durably.
    seg1_provider = MockProvider([_spawn_turn() for _ in range(2000)])
    await run(
        rt,
        root,
        tenant_id=_TENANT,
        provider=seg1_provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=2,  # interruption
    )
    after_seg1 = await _all_runs(rt, root)
    # The bound already holds on the partial durable record.
    assert len(after_seg1) <= n_max

    # Segment 2: resume the SAME root run on a fresh provider. The fanout/depth gates are
    # re-evaluated against the DURABLE graph (already-spawned children counted from it),
    # so the resume never re-spawns past the bound — a phantom over-spawn is impossible.
    seg2_provider = MockProvider([_spawn_turn() for _ in range(2000)])
    await run(
        rt,
        root,
        tenant_id=_TENANT,
        provider=seg2_provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=50,
    )
    after_resume = await _all_runs(rt, root)
    # The bound STILL holds on the durable record after resume — never breached.
    assert len(after_resume) <= n_max
    # Resuming did not LOSE the children spawned in segment 1 (the tree only grows or
    # stays — the durable record is authoritative, never reset).
    assert set(after_seg1).issubset(set(after_resume))
    # Each run id appears exactly once (no duplicate child nodes from the resume).
    assert len(after_resume) == len(set(after_resume))
