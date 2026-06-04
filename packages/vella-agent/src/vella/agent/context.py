"""The ``ContextAssembler`` seam: the frozen perceive/memory surface (M4).

This is the third of the three Protocol seams (``ModelProvider`` /
``ToolInvoker`` / ``ContextAssembler``). Where the provider is inference and the
invoker is effects, the assembler is **perception**: each turn it folds the run's
recorded cognition out of the graph and composes the canonical
:class:`~vella.agent.Message` sequence the interpreter hands the provider — a
**stable cacheable prefix** (system prompt + tool schemas + pinned context) plus a
**volatile tail** (the recent message nodes), with older turns compacted into an
``agent.summary`` node once the soft watermark is crossed.

Why a seam (R8): assembly is the one place the caching strategy and the memory
strategy meet (spec §6/§7 — "caching ⇔ context-assembly are the same problem"). A
cache-capable provider node lets the assembler mark a stable prefix as a cache
breakpoint; a non-capable one makes the assembler compact more aggressively. Pinning
this as a Protocol is also exactly how ``vectorstore`` defers cleanly: a future
similarity-recall assembler is a different impl of THIS surface and the interpreter
never changes. The in-gate reference impl is
:class:`~vella.agent.GraphContextAssembler` (graph-relationship recall only — NO
vector similarity, NO new dependency).

The assembled result is a frozen :class:`AssembledContext`: the canonical
``tuple[Message, ...]`` PLUS the cache-breakpoint metadata (which leading messages
form the stable, cacheable prefix). The interpreter feeds ``messages`` to
``provider.turn`` and uses ``cacheable_prefix_len`` to set the request's cache
directive — so the breakpoint decision is data the assembler produced, not a guess
the interpreter makes.

All models are frozen :class:`~vella.core.VellaModel` (frozen + ``extra='forbid'``),
compared via ``model_dump(mode="json")``. ``messages`` order is **semantic** (the
conversation reads in that order) and is therefore NEVER ``sorted()``; any
set-derived value the assembler serializes (e.g. ``pin`` tags) IS ``sorted()``.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable
from uuid import UUID

from pydantic import Field
from vella.core import VellaModel
from vella.runtime import Runtime

from .turn import Message


class CompactionPolicy(VellaModel):
    """The memory/compaction knobs the assembler interprets (M4 subset of §2.1).

    The plan's ``loop_policy`` (M5) embeds exactly this frozen shape; freezing it
    here — at the milestone that owns context assembly — lets the assembler be
    written and gated against it before the interpreter lands. The three fields are
    the watermark/recall vocabulary the assembler reads each turn.

    ``compaction_threshold`` is the SOFT watermark: when cumulative tokens reach it
    (and have NOT reached the hard ``token_budget`` — that halt is terminal and
    always wins) older turns fold into one ``agent.summary`` node. ``pin`` tags name
    roles/types always kept in the stable prefix; ``recall_depth`` bounds the
    graph-relationship recall hop count.

    Attributes:
        compaction_threshold: SOFT token watermark above which older turns compact
            into an ``agent.summary`` node; ``None`` = never compact. Must be ``<``
            ``token_budget`` whenever both are set (the M5 ``LoopPolicy`` validator
            enforces this, so the soft watermark always fires before the hard halt).
        pin: Role or node-type tags whose messages are always kept in the stable
            prefix (order is not semantic — a set of tags — so it is ``sorted()``
            before serialization for byte-stable bytes).
        recall_depth: The graph-relationship recall hop bound (``neighbors`` /
            ``match`` with explicit direction); ``0`` = prefix + tail only, no recall.
    """

    compaction_threshold: Optional[int] = None
    pin: tuple[str, ...] = ()
    recall_depth: int = Field(default=0, ge=0)


class AssemblyPolicy(VellaModel):
    """The assembler's per-turn inputs: the compaction knobs + the HARD token halt.

    The hard ``token_budget`` lives on the interpreter's ``loop_policy`` (M5); the
    assembler needs it now to honour the **terminal** rule — compaction NEVER runs
    once cumulative tokens reach the hard budget (the hard halt always wins;
    §2.1). Bundling it with the :class:`CompactionPolicy` here gives the assembler a
    single frozen input it can read deterministically without reaching into a
    not-yet-frozen ``LoopPolicy``.

    Attributes:
        compaction: The soft-watermark / pin / recall knobs.
        token_budget: The HARD cumulative-token halt (terminal). When cumulative
            tokens reach it the run halts with NO compaction. ``None`` = unbounded.
    """

    compaction: CompactionPolicy = Field(default_factory=CompactionPolicy)
    token_budget: Optional[int] = None


class AssembledContext(VellaModel):
    """The frozen result of one assembly: canonical messages + cache metadata.

    The interpreter feeds :attr:`messages` straight to ``provider.turn`` and reads
    :attr:`cacheable_prefix_len` to set the request's cache directive — the
    breakpoint decision is data the assembler produced (driven by the provider
    node's ``cache_capable`` flag), never a guess the interpreter makes.

    ``messages`` order is **semantic** (it is the conversation order) and is NEVER
    ``sorted()``. The stable prefix is exactly ``messages[:cacheable_prefix_len]``;
    the volatile tail is ``messages[cacheable_prefix_len:]``.

    Attributes:
        messages: The assembled canonical conversation in order (prefix then tail).
        cacheable_prefix_len: The count of leading messages that form the stable,
            cacheable prefix; ``0`` when the provider is not cache-capable (no
            breakpoint — the aggressive-compaction path).
        summary_ref: The id of the ``agent.summary`` node this assembly wrote (or
            reused) when it compacted, or ``None`` when no compaction occurred.
        compacted: Whether this assembly folded older turns into a summary node.
    """

    messages: tuple[Message, ...] = ()
    cacheable_prefix_len: int = Field(default=0, ge=0)
    summary_ref: Optional[UUID] = None
    compacted: bool = False


@runtime_checkable
class ContextAssembler(Protocol):
    """The perceive/memory seam — assembles each turn's canonical context.

    An adapter satisfies this structurally (like ``Store`` / ``ModelProvider``), so
    the in-gate :class:`~vella.agent.GraphContextAssembler` and a future
    similarity-recall assembler are interchangeable to the interpreter without a
    common base class. :meth:`assemble` perceives the run through the graph
    projection and the runtime's authority (never a privileged path) and returns a
    frozen :class:`AssembledContext` the interpreter feeds to ``provider.turn``.
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
        """Assemble the canonical context for ``run_node``'s next turn.

        Args:
            runtime: The runtime to perceive through (folded into a graph view for
                queries; ``get`` for authority). Compaction writes go through its
                verbs.
            run_node: The ``agent.run`` node whose context to assemble.
            tenant_id: The tenant the run (and every node it references) belongs to;
                the runtime keys ``get`` / ``observe`` by tenant.
            provider_node: The ``provider`` node the run infers through; its
                ``cache_capable`` flag selects the prefix-caching vs
                aggressive-compaction strategy.
            policy: The compaction knobs + the hard token budget.

        Returns:
            The frozen assembled context (canonical messages + cache metadata).
        """
        ...


__all__ = [
    "AssembledContext",
    "AssemblyPolicy",
    "CompactionPolicy",
    "ContextAssembler",
]
