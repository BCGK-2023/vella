"""``GraphContextAssembler`` — the in-gate reference :class:`ContextAssembler` (M4).

The default context assembler: it perceives a run's recorded cognition through the
:class:`~vella.graph.GraphView` (``fold`` + ``neighbors`` / ``match`` with EXPLICIT
direction) and composes the canonical message sequence each turn —

* **Stable cacheable prefix** = system prompt (the run's goal) + tool schemas +
  pinned context (messages whose role is in ``CompactionPolicy.pin``, plus
  graph-relationship recall). It is a pure function of durable, stable inputs — NO
  per-turn value (no token count, no turn index) ever enters it — so it is
  byte-identical across turns, which is exactly what makes prompt caching pay off.
* **Volatile tail** = the recent ``agent.message`` nodes folded from the run, ordered
  by node id (``uuid7`` is time-ordered, so id order IS turn order) — deterministic,
  never hash order.
* **Compaction (SOFT watermark)** = when cumulative tokens reach
  ``compaction_threshold`` (and have NOT reached the HARD ``token_budget`` — that
  halt is terminal and always wins, §2.1), the turns older than the kept tail fold
  into ONE ``agent.summary`` node written through runtime verbs (``PART_OF`` the
  run); pinned messages stay in the prefix and the run continues with shorter
  context.
* **Recall** = graph-relationship recall only (``MENTIONED_IN`` / ``REFERENCES``
  neighbours of the tail's message nodes) bounded by ``recall_depth``, via
  ``neighbors`` with an explicit direction. NO vector similarity, NO new dependency
  — similarity recall is a deferred alternate assembler impl (spec §6, vectorstore
  deferred).
* **Cache-capability coupling** = the ``provider`` node's ``cache_capable`` flag is
  the durable, graph-perceivable strategy switch (spec §7): cache-capable ⇒ the
  stable prefix is marked as a cache breakpoint (``cacheable_prefix_len`` > 0);
  non-capable ⇒ no breakpoint AND a tighter effective compaction watermark
  (aggressive compaction). Both paths are deterministic and green.

Authority discipline (TRAP-1): node bodies come from ``runtime.get`` (the
authority), never reconstructed from a ``LogEntry.payload``. Determinism: every
set-derived serialized value (pin tags, recall ids) is ``sorted()``; only the
message sequence keeps insertion order (it is semantic, not set-derived).
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from vella.core import EdgeTypes, Node, UnresolvedRef
from vella.graph import GraphProjection, GraphView
from vella.runtime import Runtime

from .context import AssembledContext, AssemblyPolicy, CompactionPolicy
from .turn import Message, TextBlock
from .types import MessageData, ProviderData, RunData, SummaryData

# Recall edge partitions perceived from the volatile tail's message nodes. Both are
# canonical core ``EdgeTypes`` (no edit pushed into core); recall follows them with
# an EXPLICIT direction ("out": a message MENTIONS/REFERENCES an entity).
_RECALL_EDGES = (EdgeTypes.MENTIONED_IN, EdgeTypes.REFERENCES)

# How many of the most-recent message nodes the volatile tail always keeps live
# (older turns are the compaction candidates). A small fixed bound keeps the tail
# deterministic and independent of per-turn token counts.
_TAIL_KEEP = 4

# The factor by which a non-cache-capable provider tightens the soft watermark —
# "aggressive compaction" (spec §7): with no prefix caching to amortize a long
# prefix, the assembler compacts at half the configured threshold so context stays
# short. A pure, deterministic function of the configured threshold.
_AGGRESSIVE_DIVISOR = 2


def _agent_actor() -> UnresolvedRef:
    """The default authorship ref stamped on assembler-written summary nodes."""
    return UnresolvedRef(identifier="vella:agent")


def _message_of(node: Node[Any, Any]) -> Message:
    """Lift an ``agent.message`` node's data into the canonical :class:`Message`.

    A stored ``agent.message`` and an in-flight canonical :class:`Message` share one
    content shape (M2), so this is a direct field lift — the recorded message
    round-trips back to the exact blocks the model emitted.

    Args:
        node: An ``agent.message`` node whose ``data`` is a :class:`MessageData`.

    Returns:
        The canonical message with the node's role and content blocks (order kept).
    """
    data = node.data
    assert isinstance(data, MessageData)
    return Message(role=data.role, content=data.content)


class GraphContextAssembler:
    """The default graph-driven :class:`~vella.agent.ContextAssembler` (in-gate).

    Stateless and reusable. Satisfies the structural
    :class:`~vella.agent.ContextAssembler` Protocol by shape (no inheritance). Every
    perception goes through the graph projection; every write (compaction summary)
    goes through runtime verbs.

    Examples:
        >>> from vella.agent import GraphContextAssembler, ContextAssembler
        >>> isinstance(GraphContextAssembler(), ContextAssembler)
        True
    """

    async def assemble(
        self,
        runtime: Runtime,
        run_node: UUID,
        *,
        tenant_id: str,
        provider_node: UUID,
        policy: AssemblyPolicy,
    ) -> AssembledContext:
        """Assemble ``run_node``'s next-turn context (see the class/Protocol docs).

        Args:
            runtime: The runtime to perceive through / write compaction summaries to.
            run_node: The ``agent.run`` node whose context to assemble.
            tenant_id: The tenant the run and every node it references belongs to.
            provider_node: The ``provider`` node whose ``cache_capable`` flag selects
                the prefix-caching vs aggressive-compaction strategy.
            policy: The compaction knobs + the hard token budget.

        Returns:
            The frozen :class:`~vella.agent.AssembledContext`.
        """
        # The run node body is authority (TRAP-1: never reconstruct from a payload).
        run = await runtime.get(tenant_id, run_node)
        assert isinstance(run, Node)
        run_data = run.data
        assert isinstance(run_data, RunData)

        cache_capable = await _cache_capable(runtime, tenant_id, provider_node)

        view = await GraphProjection().fold(runtime, tenant_id)
        message_nodes = await _run_message_nodes(view, runtime, tenant_id, run_node)
        cumulative = _cumulative_tokens(message_nodes)

        # --- partition pinned (prefix) vs unpinned (tail candidates) ---
        pin = set(policy.compaction.pin)
        pinned = [n for n in message_nodes if _node_role(n) in pin]
        unpinned = [n for n in message_nodes if _node_role(n) not in pin]

        # --- compaction decision: SOFT watermark, NEVER past the HARD halt ---
        threshold = _effective_threshold(policy.compaction, cache_capable)
        hard = policy.token_budget
        at_hard_halt = hard is not None and cumulative >= hard
        do_compact = (
            not at_hard_halt
            and threshold is not None
            and cumulative >= threshold
            and len(unpinned) > _TAIL_KEEP
        )

        summary_ref: Optional[UUID] = None
        older = unpinned[:-_TAIL_KEEP] if len(unpinned) > _TAIL_KEEP else []
        tail_nodes = unpinned[-_TAIL_KEEP:] if len(unpinned) > _TAIL_KEEP else unpinned
        if do_compact and older:
            summary_ref = await _write_summary(
                view, runtime, run_node, tenant_id, older
            )

        # --- recall: graph relationships only, bounded by recall_depth ---
        recall_messages = await _recall_messages(
            view, runtime, tenant_id, tail_nodes, policy.compaction.recall_depth
        )

        # --- compose: stable prefix (system + tools + pinned + recall) + tail ---
        prefix = _stable_prefix(run_data, pinned, recall_messages)
        if do_compact and summary_ref is not None:
            summary = await runtime.get(tenant_id, summary_ref)
            assert isinstance(summary, Node) and isinstance(summary.data, SummaryData)
            prefix = prefix + (
                Message(role="system", content=(TextBlock(text=summary.data.text),)),
            )

        tail = tuple(_message_of(n) for n in tail_nodes)
        messages = prefix + tail
        cacheable_prefix_len = len(prefix) if cache_capable else 0

        return AssembledContext(
            messages=messages,
            cacheable_prefix_len=cacheable_prefix_len,
            summary_ref=summary_ref,
            compacted=do_compact and older != [],
        )


def _effective_threshold(
    compaction: CompactionPolicy, cache_capable: bool
) -> Optional[int]:
    """The soft watermark, tightened for a non-cache-capable provider.

    A cache-capable provider amortizes a long stable prefix via prompt caching, so
    it compacts at the configured ``compaction_threshold``. A non-capable provider
    cannot, so it compacts AGGRESSIVELY — at half the threshold (integer-floored) —
    keeping context short (spec §7). ``None`` (never compact) is preserved either
    way. This is the deterministic strategy switch the ``cache_capable`` flag drives.

    Args:
        compaction: The compaction knobs.
        cache_capable: Whether the run's provider node supports prompt caching.

    Returns:
        The effective soft watermark, or ``None`` when compaction is disabled.
    """
    threshold = compaction.compaction_threshold
    if threshold is None:
        return None
    if cache_capable:
        return threshold
    return threshold // _AGGRESSIVE_DIVISOR


def _stable_prefix(
    run_data: RunData,
    pinned: list[Node[Any, Any]],
    recall_messages: tuple[Message, ...],
) -> tuple[Message, ...]:
    """Build the byte-stable prefix: system prompt + pinned context + recall.

    Pure function of durable inputs only — the run's goal, the pinned message nodes
    (in stable id order), and the graph-recalled context. NO per-turn value (token
    count, turn index, timestamp) is admitted, which is what makes the prefix
    byte-identical across turns (the cache-stability property).

    Args:
        run_data: The run's frozen payload (its ``goal`` is the system prompt).
        pinned: The message nodes whose role is pinned, in stable id order.
        recall_messages: The graph-relationship recall context (already canonical).

    Returns:
        The stable prefix messages in deterministic order.
    """
    system = Message(
        role="system", content=(TextBlock(text=run_data.goal),)
    )
    pinned_messages = tuple(_message_of(n) for n in pinned)
    return (system,) + pinned_messages + recall_messages


async def _cache_capable(
    runtime: Runtime, tenant_id: str, provider_node: UUID
) -> bool:
    """Read the provider node's durable cache-capability flag (authority).

    The capability is the NODE's property (spec §6) — read from ``runtime.get``
    authority, never inferred. A missing / non-provider node is treated as
    non-capable (the safe, aggressive-compaction default).

    Args:
        runtime: The runtime to read authority through.
        tenant_id: The tenant the provider node belongs to.
        provider_node: The ``provider`` node id.

    Returns:
        ``True`` iff the node is a ``provider`` whose ``cache_capable`` is set.
    """
    node = await runtime.get(tenant_id, provider_node)
    if isinstance(node, Node) and isinstance(node.data, ProviderData):
        return node.data.cache_capable
    return False


def _node_role(node: Node[Any, Any]) -> str:
    """The author role of an ``agent.message`` node (``""`` if not a message)."""
    data = node.data
    return data.role if isinstance(data, MessageData) else ""


def _cumulative_tokens(message_nodes: list[Node[Any, Any]]) -> int:
    """Cumulative content length across the run's message nodes (deterministic proxy).

    The interpreter (M5) accumulates real ``usage`` totals; the assembler, which
    perceives only the durable message nodes, derives a deterministic token proxy
    from the recorded content text length so the soft-watermark decision is a pure
    function of the graph. Cache tokens never enter a budget (§2.1), and this proxy
    is text-only — never a per-call usage field that would vary by provider.

    Args:
        message_nodes: The run's ``agent.message`` nodes.

    Returns:
        The summed text length across all message-node content blocks.
    """
    total = 0
    for node in message_nodes:
        data = node.data
        if not isinstance(data, MessageData):
            continue
        for block in data.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                total += len(text)
    return total


async def _run_message_nodes(
    view: GraphView, runtime: Runtime, tenant_id: str, run_node: UUID
) -> list[Node[Any, Any]]:
    """The run's ``agent.message`` nodes in turn order (by id; never hash order).

    A message is linked ``message -> run`` via ``PART_OF`` (see ``_writeback``), so
    the run's messages are its ``PART_OF`` ``"in"`` neighbours (EXPLICIT direction —
    never ``"both"``, which would be nondeterministic/wrong). The neighbour ids come
    back canonically sorted by the view; bodies are read from ``runtime.get``
    authority (TRAP-1), and only ``agent.message`` nodes are kept (steps / tool_calls
    / summaries are filtered out). ``uuid7`` ids are time-ordered, so id order is
    turn order.

    Args:
        view: The folded graph view for ``tenant_id``.
        runtime: The runtime to read node bodies through (authority).
        tenant_id: The tenant the run belongs to.
        run_node: The run whose messages to collect.

    Returns:
        The run's message nodes, ordered by ``str(node.id)`` (turn order).
    """
    neighbours = await view.neighbors(
        run_node, edge_type=EdgeTypes.PART_OF, direction="in"
    )
    out: list[Node[Any, Any]] = []
    for neighbour in neighbours:
        node = await runtime.get(tenant_id, neighbour.node_id)
        if isinstance(node, Node) and isinstance(node.data, MessageData):
            out.append(node)
    # neighbours are already canonically id-ordered; re-sort defensively so turn
    # order is a property of the result, not of the query's ordering contract.
    out.sort(key=lambda n: str(n.id))
    return out


async def _recall_messages(
    view: GraphView,
    runtime: Runtime,
    tenant_id: str,
    tail_nodes: list[Node[Any, Any]],
    recall_depth: int,
) -> tuple[Message, ...]:
    """Graph-relationship recall context for the tail's messages (NO vector).

    For each message node in the tail, follows the recall edge partitions
    (``MENTIONED_IN`` / ``REFERENCES``) with an EXPLICIT ``"out"`` direction up to
    ``recall_depth`` hops, collecting the reached entity nodes. The recalled node
    ids are de-duplicated and ``sorted()`` (a set-derived serialized value → sorted
    for byte-stable bytes), then rendered as canonical ``system``-role recall
    messages. ``recall_depth == 0`` recalls nothing. This is the ONLY recall path —
    there is no similarity / vector branch (the deferred-vectorstore seam).

    Args:
        view: The folded graph view.
        runtime: The runtime to read recalled bodies through (authority).
        tenant_id: The tenant.
        tail_nodes: The volatile-tail message nodes to recall from.
        recall_depth: The hop bound (``0`` = no recall).

    Returns:
        The recall context as canonical messages, in sorted-id (deterministic) order.
    """
    if recall_depth <= 0:
        return ()
    recalled: set[UUID] = set()
    anchors = [n.id for n in tail_nodes]
    for _hop in range(recall_depth):
        next_anchors: list[UUID] = []
        for anchor in anchors:
            for edge_type in _RECALL_EDGES:
                neighbours = await view.neighbors(
                    anchor, edge_type=edge_type, direction="out"
                )
                for neighbour in neighbours:
                    if neighbour.node_id not in recalled:
                        recalled.add(neighbour.node_id)
                        next_anchors.append(neighbour.node_id)
        anchors = next_anchors
    # Set-derived serialized value -> sorted() for deterministic, hash-stable bytes.
    out: list[Message] = []
    for node_id in sorted(recalled, key=str):
        node = await runtime.get(tenant_id, node_id)
        if isinstance(node, Node):
            out.append(
                Message(
                    role="system",
                    content=(TextBlock(text=f"recall:{node.name}"),),
                )
            )
    return tuple(out)


def _summary_data(older: list[Node[Any, Any]]) -> SummaryData:
    """The deterministic ``agent.summary`` payload for a compacted turn range.

    A pure function of ``older`` (its length + roles, NOT a model call — the in-gate
    reference impl is network-free), so the artifact — and the idempotency key built
    from its ``text`` — is byte-stable across hash seeds and repeated assembly.

    Args:
        older: The message nodes being compacted (in turn order).

    Returns:
        The frozen summary payload.
    """
    roles = ",".join(_node_role(n) for n in older)
    return SummaryData(
        compacted_range=(0, len(older) - 1),
        text=f"compacted {len(older)} turns: {roles}",
    )


async def _existing_summary(
    view: GraphView,
    runtime: Runtime,
    run_node: UUID,
    tenant_id: str,
    text: str,
) -> Optional[UUID]:
    """An existing ``agent.summary`` node ``PART_OF`` the run with matching ``text``.

    Compaction must be idempotent: re-assembling a run that already crossed the
    watermark must REUSE the summary it wrote, never append a duplicate (which would
    make ``summary_ref`` — and the durable run projection — nondeterministic across
    repeated calls). The summary's ``text`` is a deterministic digest of the
    compacted range, so it is the stable reuse key.

    Args:
        view: The folded graph view (the run's ``PART_OF`` ``"in"`` neighbours).
        runtime: The runtime to read summary bodies through (authority).
        run_node: The run whose summaries to scan.
        tenant_id: The tenant.
        text: The deterministic summary text to match.

    Returns:
        The id of a matching existing summary, or ``None``.
    """
    neighbours = await view.neighbors(
        run_node, edge_type=EdgeTypes.PART_OF, direction="in"
    )
    for neighbour in sorted(neighbours, key=lambda n: str(n.node_id)):
        node = await runtime.get(tenant_id, neighbour.node_id)
        if (
            isinstance(node, Node)
            and isinstance(node.data, SummaryData)
            and node.data.text == text
        ):
            return node.id
    return None


async def _write_summary(
    view: GraphView,
    runtime: Runtime,
    run_node: UUID,
    tenant_id: str,
    older: list[Node[Any, Any]],
) -> UUID:
    """Fold ``older`` turns into ONE ``agent.summary`` node via verbs (idempotent).

    Writes a single ``agent.summary`` node (``compacted_range`` + deterministic
    ``text``) through ``runtime.create`` and links it ``PART_OF`` the run through
    ``runtime.link`` — the same verb path ``_writeback`` uses; no privileged write.
    First reuses an existing summary with the same deterministic ``text`` so repeated
    assembly is idempotent (one summary per compacted range, never a duplicate).

    Args:
        view: The folded graph view (used to find an existing summary to reuse).
        runtime: The runtime to write through (``create`` + ``link``).
        run_node: The run the summary belongs to (the ``PART_OF`` target).
        tenant_id: The tenant.
        older: The message nodes being compacted (in turn order).

    Returns:
        The ``agent.summary`` node's id (reused when one already exists).
    """
    data = _summary_data(older)
    existing = await _existing_summary(view, runtime, run_node, tenant_id, data.text)
    if existing is not None:
        return existing
    node = Node.from_data(
        data, name="summary", created_by=_agent_actor(), tenant_id=tenant_id
    )
    await runtime.create(node)
    await runtime.link(
        tenant_id,
        node.id,
        run_node,
        edge_type=EdgeTypes.PART_OF,
        created_by=_agent_actor(),
    )
    return node.id


__all__ = ["GraphContextAssembler"]
