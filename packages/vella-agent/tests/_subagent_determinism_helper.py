"""Subprocess fixture for the M6 sub-agent run-tree determinism artifact.

Run as a script under a fixed ``PYTHONHASHSEED`` (set by the parent test). It drives an
ADVERSARIAL sub-agent run (every turn requests a spawn) bounded by ``max_depth`` /
``max_fanout``, then folds the materialized run-tree from the graph and emits its
**structural digest** as canonical, byte-stable JSON to stdout.

The digest is hash-seed-INDEPENDENT by construction: concrete node UUIDs are random per
run, so the artifact records only the run-tree SHAPE — for each run, the sorted
``(parent_depth, direct_child_count)`` pair, aggregated and ``sorted()``. The neighbour
iteration that builds it is genuinely set-derived (graph adjacency), so a missing
``sorted()`` (or a ``direction="both"`` depth walk that miscounts) would let
construction/hash order leak into the bytes — this subprocess test across
``PYTHONHASHSEED {0,1,42}`` is what would catch it.

The mechanism MUST be a subprocess: ``PYTHONHASHSEED`` is read once at interpreter
startup, so an in-process re-import does NOT reset it; only a fresh interpreter can vary
it. This is a script, not a test module.
"""

from __future__ import annotations

import asyncio
import json
from uuid import UUID

from vella.core import EdgeTypes, Node, UnresolvedRef
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
    run,
)
from vella.agent._subagent import SPAWN_TOOL, _parent_depth
from vella.agent._writeback import create_run

_TENANT = "t-subagent-determinism"
_TOKENS_PER_TURN = 3
_PER_RUN_TOKEN_BUDGET = 30
_MAX_DEPTH = 2
_MAX_FANOUT = 2


def _spawn_turn() -> ScriptedTurn:
    """A turn that requests a sub-agent spawn (the reserved SPAWN_TOOL affordance)."""
    return ScriptedTurn(
        blocks=(
            ScriptedToolUse(
                id="sp",
                name=SPAWN_TOOL,
                input={"goal": "child", "token_budget": _PER_RUN_TOKEN_BUDGET},
                intent="spawn a child sub-agent.",
            ),
        ),
        stop_reason="tool_use",
        usage=Usage(output_tokens=_TOKENS_PER_TURN),
    )


async def _run_tree_shape(rt: Runtime, root: UUID) -> list[list[int]]:
    """The hash-seed-stable structural digest of the run-tree under ``root``.

    For every ``agent.run`` reachable down the PART_OF tree, record its
    ``[parent_depth, direct_child_count]`` (depth via the upward ``direction="out"``
    chain; child count via the ``direction="in"`` neighbours). The list is ``sorted()``,
    so it is a stable fingerprint independent of node-id (and hence hash) ordering.
    """
    view = await GraphProjection().fold(rt, _TENANT)
    shape: list[list[int]] = []
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
        children = await view.neighbors(
            node_id, edge_type=EdgeTypes.PART_OF, direction="in"
        )
        child_runs: list[UUID] = []
        for child in children:
            child_node = await rt.get(_TENANT, child.node_id)
            if isinstance(child_node, Node) and isinstance(child_node.data, RunData):
                child_runs.append(child.node_id)
        depth = await _parent_depth(view, node_id)
        shape.append([depth, len(child_runs)])
        stack.extend(child_runs)
    return sorted(shape)


async def _build() -> str:
    agent_registry()
    rt = Runtime()
    root_policy = LoopPolicy(
        token_budget=_PER_RUN_TOKEN_BUDGET,
        sub_agent_spawn=SubAgentAllow(max_depth=_MAX_DEPTH, max_fanout=_MAX_FANOUT),
    )
    # Both writes go through the public verbs (create for the policy node; create_run
    # for the root run); `run` itself never creates the root.
    policy_node = Node.from_data(
        root_policy,
        name="policy",
        created_by=UnresolvedRef(identifier="vella:test"),
        tenant_id=_TENANT,
    )
    await rt.create(policy_node)
    root_node = await create_run(
        rt,
        RunData(goal="root", loop_policy_ref=policy_node.id),
        name="root",
        tenant_id=_TENANT,
    )
    provider = MockProvider([_spawn_turn() for _ in range(2000)])
    await run(
        rt,
        root_node.id,
        tenant_id=_TENANT,
        provider=provider,
        invoker=InMemoryToolInvoker(clock=ManualClock()),
        assembler=GraphContextAssembler(),
        clock=ManualClock(),
        max_steps=50,
    )
    shape = await _run_tree_shape(rt, root_node.id)
    return json.dumps(shape, sort_keys=True, separators=(",", ":"))


def build_artifact() -> str:
    """Serialize the sub-agent run-tree structural digest to byte-stable JSON."""
    return asyncio.run(_build())


def main() -> None:
    """Print the run-tree structural digest as canonical, byte-stable JSON to stdout."""
    print(build_artifact(), end="")


if __name__ == "__main__":
    main()
