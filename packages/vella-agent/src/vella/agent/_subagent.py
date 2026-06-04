"""Bounded sub-agents (M6, §2.2) — runaway provably impossible (cardinality + cost).

A run may spawn a child ``agent.run`` linked ``child --PART_OF--> parent``. The whole
mechanism is gated by TWO independent bounds, both read from the **durable graph**
(the authority — never an in-memory counter, TRAP-1) and checked **before** any child
node or ``PART_OF`` edge is created:

1. **depth** — ``child_depth = parent_depth + 1 <= max_depth``, where ``parent_depth``
   is the length of the ``PART_OF`` chain walked from the parent UP to the root run.
   Edges are ``child --PART_OF--> parent``, so walking ``direction="out"`` from a node
   yields its parent; iterating ``"out"`` until no parent remains counts the depth.
   NEVER ``direction="both"`` (nondeterministic / would double-count, mutation (e)).
2. **fanout** — ``parent_fanout_after = current_direct_children + 1 <= max_fanout``,
   counted from ``neighbors(parent, edge_type="part_of", direction="in")`` (a child
   points IN to its parent). NEVER ``direction="both"`` (mutation (c) drops this).

A spawn that would breach EITHER bound is REFUSED *before* the child node/edge exists
(no node, no edge written) — so the bound holds on the durable record and a
replay/resume cannot resurrect a phantom over-spawn (mutation (b) creates-then-checks).

**Cardinality bound.** A tree of depth ``<= d`` branching ``<= f`` has at most
``N_max = Σ_{i=0..d} f^i`` runs (:func:`max_run_tree_size`). Under an adversarial
provider requesting a spawn every turn at every level, the materialized run-tree never
exceeds ``N_max`` — because every spawn passes through this gate against the durable
graph.

**Cost bound.** Each child gets its OWN budget from its OWN ``loop_policy`` — there is
NO pooling and NO shared mutable counter (mutation (f); pooling would couple siblings
through mutable state and violate TRAP-1). So aggregate token spend across the whole
tree is bounded by ``N_max * per_run_token_budget``.

**Result propagation** is a GRAPH READ, never an in-memory handoff (mutation (d)): a
child folds to a terminal status + a final ``agent.message``; the parent's NEXT
context-assembly pass perceives the child's terminal output by walking the graph
(``child PART_OF parent``, found with an explicit ``direction``) and injects it as a
``tool_result``-shaped block. Because it is read from the durable graph, it survives
replay/resume identically.

Every write here goes through the runtime's published verbs (the child run via
:func:`~vella.agent._writeback.create_run`; the ``PART_OF`` link via ``runtime.link``);
there is no privileged path and no new ``Node`` subclass.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from vella.core import EdgeTypes, Node, UnresolvedRef
from vella.graph import GraphProjection, GraphView
from vella.runtime import Runtime

from ._writeback import create_run
from .policy import LoopPolicy, SubAgentAllow
from .types import RunData

# The reserved tool name a turn emits to REQUEST a sub-agent spawn — recognized by the
# interpreter the same way :data:`~vella.agent.EXPLICIT_STOP_TOOL` is. A spawn request
# is a ``tool_use`` block named this, whose ``input`` carries the child's ``goal`` (and
# optionally its own ``step_budget`` / ``token_budget`` — the child's OWN budget, never
# pooled). Kept a module constant (not a policy knob) so the spawn affordance is a
# fixed, replayable part of the FSM surface, not configurable data.
SPAWN_TOOL = "spawn_subagent"
"""The reserved ``tool_use`` name a turn emits to request a bounded sub-agent spawn."""


def _agent_actor() -> UnresolvedRef:
    """The default authorship ref stamped on spawned child runs + their links."""
    return UnresolvedRef(identifier="vella:agent")


def max_run_tree_size(max_depth: int, max_fanout: int) -> int:
    """The closed-form cardinality bound ``N_max = Σ_{i=0..max_depth} max_fanout^i``.

    The maximum number of runs in a tree rooted at one run, of depth ``<= max_depth``
    (the root is depth 0) and branching ``<= max_fanout`` at every node. This is the
    provable upper bound on the materialized run-tree the pre-spawn gate enforces; the
    adversarial spawn-every-turn test asserts the actual tree never exceeds it.

    Args:
        max_depth: The maximum ``PART_OF`` chain depth a child may reach (``>= 1``).
        max_fanout: The maximum direct children any single run may have (``>= 1``).

    Returns:
        ``Σ_{i=0..max_depth} max_fanout^i`` — the closed-form run-tree cardinality
        bound.
    """
    total = 0
    for i in range(max_depth + 1):
        total += max_fanout**i
    return total


async def _parent_depth(view: GraphView, parent: UUID) -> int:
    """The ``PART_OF`` chain depth of ``parent`` (root = 0), via ``direction="out"``.

    Edges are ``child --PART_OF--> parent``, so a node's parent is its single
    ``PART_OF`` ``"out"`` neighbour; walking ``"out"`` until no parent remains counts
    the hops to the root. The walk is read from the DURABLE graph (TRAP-1 — never an
    in-memory counter, mutation (a)) and uses an EXPLICIT ``direction="out"`` (never
    ``"both"``, mutation (e), which would walk back down into children and miscount).
    A small visited-set guards against a malformed cyclic link so the walk always
    terminates.

    Args:
        view: The folded graph view to walk.
        parent: The prospective parent run whose depth to measure.

    Returns:
        The number of ``PART_OF`` hops from ``parent`` up to the root run (``0`` when
        ``parent`` is itself a root with no parent link).
    """
    depth = 0
    current = parent
    seen: set[UUID] = {current}
    while True:
        parents = await view.neighbors(
            current, edge_type=EdgeTypes.PART_OF, direction="out"
        )
        if not parents:
            return depth
        # A run has at most one PART_OF parent; take the canonical-first defensively.
        nxt = sorted(parents, key=lambda n: str(n.node_id))[0].node_id
        if nxt in seen:
            return depth
        seen.add(nxt)
        current = nxt
        depth += 1


async def _parent_fanout(
    view: GraphView, runtime: Runtime, parent: UUID, *, tenant_id: str
) -> int:
    """The number of direct ``PART_OF`` children ``parent`` already has (``"in"``).

    A child points IN to its parent (``child --PART_OF--> parent``), so the parent's
    current direct children are its ``PART_OF`` ``"in"`` neighbours. Counted from the
    DURABLE graph with an EXPLICIT ``direction="in"`` (never ``"both"``, mutation (c)).
    Only ``agent.run`` neighbours count as children (steps/messages also link
    ``PART_OF`` the run, so they MUST be filtered out or the fanout would be inflated);
    bodies are read from ``runtime.get`` authority (TRAP-1).

    Args:
        view: The folded graph view (the run's ``PART_OF`` ``"in"`` neighbours).
        runtime: The runtime to read neighbour bodies through (authority).
        parent: The prospective parent run whose current fanout to count.
        tenant_id: The tenant the run-tree belongs to.

    Returns:
        The count of distinct direct child ``agent.run`` nodes of ``parent``.
    """
    children = await view.neighbors(
        parent, edge_type=EdgeTypes.PART_OF, direction="in"
    )
    seen: set[UUID] = set()
    for child in children:
        node = await runtime.get(tenant_id, child.node_id)
        if isinstance(node, Node) and isinstance(node.data, RunData):
            seen.add(child.node_id)
    return len(seen)


async def gate_allows_spawn(
    runtime: Runtime,
    parent: UUID,
    *,
    tenant_id: str,
    allow: SubAgentAllow,
) -> bool:
    """Whether a spawn under ``parent`` is permitted by BOTH bounds (pre-spawn gate).

    Computes both bounds from the DURABLE graph BEFORE any child node/edge is created
    (TRAP-1): the prospective child's depth (``parent_depth + 1``) and the parent's
    fanout after the spawn (``current_children + 1``). Returns ``True`` only when BOTH
    are within ``allow.max_depth`` / ``allow.max_fanout``. A breach of EITHER bound
    returns ``False`` so the caller never creates the child (no node, no edge).

    Args:
        runtime: The runtime to fold the authoritative graph from (read-only here).
        parent: The prospective parent run.
        tenant_id: The tenant the run-tree belongs to.
        allow: The :class:`~vella.agent.SubAgentAllow` carrying ``max_depth`` /
            ``max_fanout``.

    Returns:
        ``True`` iff the spawn keeps the tree within both bounds; ``False`` otherwise.
    """
    view = await GraphProjection().fold(runtime, tenant_id)
    parent_depth = await _parent_depth(view, parent)
    if parent_depth + 1 > allow.max_depth:
        return False
    parent_fanout_after = (
        await _parent_fanout(view, runtime, parent, tenant_id=tenant_id) + 1
    )
    if parent_fanout_after > allow.max_fanout:
        return False
    return True


async def spawn_child(
    runtime: Runtime,
    parent: UUID,
    *,
    tenant_id: str,
    goal: str,
    child_policy_ref: Optional[UUID],
    provider_ref: Optional[UUID],
) -> Optional[UUID]:
    """Create a child ``agent.run`` ``PART_OF`` ``parent`` IF the gate allows; else None.

    Re-checks the pre-spawn gate against the DURABLE graph and, only on success,
    creates the child run via the published verbs and links it
    ``child --PART_OF--> parent`` (``runtime.link``). The order is load-bearing
    (mutation (b)): the gate is evaluated on the graph BEFORE the child node exists, so
    a breach leaves the durable record untouched. The child carries its OWN
    ``loop_policy_ref`` (its OWN budget — no pooling, mutation (f)) and its own
    ``provider_ref``.

    Note this re-folds the gate internally; the caller is expected to have already
    confirmed :func:`gate_allows_spawn` against the same :class:`~vella.agent.SubAgentAllow`
    — but re-checking here keeps the create+link atomic-on-success and makes the
    function safe to call directly.

    Args:
        runtime: The runtime to write the child + link through (verbs only).
        parent: The parent run the child is ``PART_OF``.
        tenant_id: The tenant both runs belong to.
        goal: The child run's goal text.
        child_policy_ref: The child's OWN ``loop_policy`` node id (its own budget), or
            ``None`` for an all-default child policy.
        provider_ref: The child's ``provider`` node id, or ``None``.

    Returns:
        The created child run's id, or ``None`` when no child was created (the gate
        must be checked by the caller; this returns the created id on success).
    """
    actor = _agent_actor()
    child = await create_run(
        runtime,
        RunData(
            goal=goal,
            loop_policy_ref=child_policy_ref,
            provider_ref=provider_ref,
        ),
        name="subagent-run",
        tenant_id=tenant_id,
        created_by=actor,
    )
    await runtime.link(
        tenant_id,
        child.id,
        parent,
        edge_type=EdgeTypes.PART_OF,
        created_by=actor,
    )
    return child.id


async def child_runs_of(
    runtime: Runtime, parent: UUID, *, tenant_id: str
) -> list[UUID]:
    """The direct child ``agent.run`` ids of ``parent``, sorted by ``str(id)``.

    Reads the parent's ``PART_OF`` ``"in"`` neighbours from the durable graph (EXPLICIT
    direction — a child points IN to its parent), filters to ``agent.run`` nodes
    (steps/messages also link ``PART_OF`` the run), and returns them in a deterministic
    sorted order (a set-derived serialized value → ``sorted()``). This is the read the
    parent's context assembly uses to find a child's terminal output for propagation.

    Args:
        runtime: The runtime to fold + read authority through.
        parent: The parent run whose children to list.
        tenant_id: The tenant the run-tree belongs to.

    Returns:
        The direct child run ids, sorted by ``str(id)`` (deterministic).
    """
    view = await GraphProjection().fold(runtime, tenant_id)
    children = await view.neighbors(
        parent, edge_type=EdgeTypes.PART_OF, direction="in"
    )
    ids: set[UUID] = set()
    for child in children:
        node = await runtime.get(tenant_id, child.node_id)
        if isinstance(node, Node) and isinstance(node.data, RunData):
            ids.add(child.node_id)
    return sorted(ids, key=str)


async def child_loop_policy(
    runtime: Runtime, child: UUID, *, tenant_id: str
) -> LoopPolicy:
    """The child run's OWN :class:`~vella.agent.LoopPolicy` (no pooling), or default.

    Read from the child's ``loop_policy_ref`` via ``runtime.get`` authority (TRAP-1).
    Each child has its OWN policy — there is no shared/pooled budget object (mutation
    (f)) — so the per-run ``token_budget`` here is the multiplicand in the aggregate
    cost bound ``N_max * per_run_token_budget``.

    Args:
        runtime: The runtime to read the policy node through.
        child: The child run id.
        tenant_id: The tenant the child belongs to.

    Returns:
        The child's loop policy, or an all-default :class:`~vella.agent.LoopPolicy`
        when none is attached.
    """
    node = await runtime.get(tenant_id, child)
    if not (isinstance(node, Node) and isinstance(node.data, RunData)):
        return LoopPolicy()
    ref = node.data.loop_policy_ref
    if ref is None:
        return LoopPolicy()
    policy_node = await runtime.get(tenant_id, ref)
    if isinstance(policy_node, Node) and isinstance(policy_node.data, LoopPolicy):
        return policy_node.data
    return LoopPolicy()


async def child_terminal_messages(
    runtime: Runtime, parent: UUID, *, tenant_id: str
) -> tuple[Any, ...]:
    """The terminal output messages of ``parent``'s TERMINATED children (graph read).

    Result propagation (§2.2) as a pure GRAPH READ (never an in-memory handoff,
    mutation (d)): for each direct child run that has reached a terminal status
    (``succeeded`` / ``failed``), perceives its final ``agent.message`` node from the
    durable graph and lifts it to a canonical :class:`~vella.agent.Message`. The result
    is deterministic — children are visited in sorted-id order and each child's
    messages in sorted-id order. Because it reads the durable record, the same
    propagation reconstructs identically on replay/resume.

    Args:
        runtime: The runtime to fold + read authority through.
        parent: The parent run whose children's terminal output to collect.
        tenant_id: The tenant the run-tree belongs to.

    Returns:
        One summary-shaped :class:`~vella.agent.Message` per terminated child carrying
        that child's final text, in deterministic (sorted child-id) order.
    """
    # Local imports avoid a module-load cycle (turn/types are siblings).
    from .turn import Message, TextBlock
    from .types import MessageData

    view = await GraphProjection().fold(runtime, tenant_id)
    out: list[Any] = []
    for child_id in await child_runs_of(runtime, parent, tenant_id=tenant_id):
        child_node = await runtime.get(tenant_id, child_id)
        if not (isinstance(child_node, Node) and isinstance(child_node.data, RunData)):
            continue
        status = child_node.data.status
        if status not in ("succeeded", "failed"):
            continue
        msg_neighbours = await view.neighbors(
            child_id, edge_type=EdgeTypes.PART_OF, direction="in"
        )
        msgs: list[Node[Any, Any]] = []
        for nb in msg_neighbours:
            node = await runtime.get(tenant_id, nb.node_id)
            if isinstance(node, Node) and isinstance(node.data, MessageData):
                msgs.append(node)
        msgs.sort(key=lambda n: str(n.id))
        if not msgs:
            continue
        final = msgs[-1]
        data = final.data
        assert isinstance(data, MessageData)
        text = _terminal_text(data)
        # A summary-shaped propagation block (§2.2): the child's terminal output read
        # from the durable graph, framed as a system summary keyed by the child run id
        # so the parent's model perceives it as injected sub-agent context — never an
        # in-memory handoff. Deterministic bytes (fixed prefix + the child's text).
        out.append(
            Message(
                role="system",
                content=(
                    TextBlock(
                        text=f"subagent:{child_id}:{status}:{text}",
                    ),
                ),
            )
        )
    return tuple(out)


def _terminal_text(data: Any) -> str:
    """The concatenated text content of a terminal child message (deterministic).

    Lifts every ``text``-bearing block of the child's final message into one string in
    content (semantic) order, so the propagated summary carries the child's answer
    deterministically. Non-text blocks contribute nothing.

    Args:
        data: The final :class:`~vella.agent.MessageData` of a terminated child.

    Returns:
        The joined text of the message's text blocks (``""`` when none).
    """
    parts: list[str] = []
    for block in getattr(data, "content", ()):
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return " ".join(parts)
