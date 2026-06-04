"""Compaction at the SOFT watermark — and NEVER past the HARD ``token_budget`` halt.

Covers the §2.1 compaction semantics: at the soft ``compaction_threshold`` the
assembler folds older turns into ONE ``agent.summary`` node written through runtime
verbs and linked ``PART_OF`` the run, then the run continues with a shorter context.
The HARD ``token_budget`` halt is terminal and always wins — once cumulative tokens
reach it the assembler does NOT compact (asserted independently).

No ``pytest-asyncio``: each async case runs via ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vella.core import EdgeTypes, Node, UnresolvedRef
from vella.graph import GraphProjection
from vella.runtime import Runtime

from vella.agent import (
    AssemblyPolicy,
    CompactionPolicy,
    GraphContextAssembler,
    Message,
    MessageData,
    ProviderData,
    RunData,
    SummaryData,
    TextBlock,
    agent_registry,
)
from vella.agent._writeback import append_message, create_run

_TENANT = "t-agent"
_ACTOR = UnresolvedRef(identifier="vella:test")


def _text(message: Message) -> str:
    """The first content block's text (the test messages are all single text blocks)."""
    block = message.content[0]
    assert isinstance(block, TextBlock)
    return block.text


async def _provider(rt: Runtime) -> Node[Any, Any]:
    node = Node.from_data(
        ProviderData(model_id="m", cache_capable=True),
        name="provider",
        created_by=_ACTOR,
        tenant_id=_TENANT,
    )
    await rt.create(node)
    return node


async def _run_with_messages(rt: Runtime, count: int, *, text_len: int) -> Node[Any, Any]:
    run = await create_run(rt, RunData(goal="g"), name="run", tenant_id=_TENANT)
    for i in range(count):
        await append_message(
            rt,
            run.id,
            MessageData(role="user", content=(TextBlock(text="x" * text_len),)),
            name=f"m{i}",
            tenant_id=_TENANT,
        )
    return run


def test_soft_watermark_folds_older_turns_into_summary_node() -> None:
    async def _case() -> None:
        agent_registry()
        rt = Runtime()
        prov = await _provider(rt)
        # 8 messages * 10 chars = 80 cumulative; soft=50, hard=1000.
        run = await _run_with_messages(rt, 8, text_len=10)
        ctx = await GraphContextAssembler().assemble(
            rt,
            run.id,
            tenant_id=_TENANT,
            provider_node=prov.id,
            policy=AssemblyPolicy(
                compaction=CompactionPolicy(compaction_threshold=50),
                token_budget=1000,
            ),
        )
        assert ctx.compacted is True
        assert ctx.summary_ref is not None

        # the summary node is a real agent.summary node written via verbs.
        summary = await rt.get(_TENANT, ctx.summary_ref)
        assert isinstance(summary, Node)
        assert summary.type == "agent.summary"
        assert isinstance(summary.data, SummaryData)

        # it is linked PART_OF the run (it is a run's "in" neighbour over PART_OF).
        view = await GraphProjection().fold(rt, _TENANT)
        part_of = await view.neighbors(
            run.id, edge_type=EdgeTypes.PART_OF, direction="in"
        )
        assert ctx.summary_ref in {n.node_id for n in part_of}

        # the run continues with a SHORTER context: system + summary prefix + tail.
        assert any(
            m.role == "system" and "compacted" in _text(m) for m in ctx.messages
        )

    asyncio.run(asyncio.wait_for(_case(), timeout=5.0))


def test_no_compaction_below_the_soft_watermark() -> None:
    async def _case() -> None:
        agent_registry()
        rt = Runtime()
        prov = await _provider(rt)
        # 8 * 10 = 80 cumulative, but soft threshold is 200 -> below watermark.
        run = await _run_with_messages(rt, 8, text_len=10)
        ctx = await GraphContextAssembler().assemble(
            rt,
            run.id,
            tenant_id=_TENANT,
            provider_node=prov.id,
            policy=AssemblyPolicy(
                compaction=CompactionPolicy(compaction_threshold=200),
                token_budget=1000,
            ),
        )
        assert ctx.compacted is False
        assert ctx.summary_ref is None

    asyncio.run(asyncio.wait_for(_case(), timeout=5.0))


def test_compaction_never_runs_past_the_hard_token_budget() -> None:
    """At/над the hard ``token_budget`` the assembler does NOT compact (terminal)."""

    async def _case() -> None:
        agent_registry()
        rt = Runtime()
        prov = await _provider(rt)
        # 8 * 10 = 80 cumulative. soft=50 (would fire), but hard=60 is already
        # breached (80 >= 60) -> the hard halt is terminal; NO compaction occurs.
        run = await _run_with_messages(rt, 8, text_len=10)
        ctx = await GraphContextAssembler().assemble(
            rt,
            run.id,
            tenant_id=_TENANT,
            provider_node=prov.id,
            policy=AssemblyPolicy(
                compaction=CompactionPolicy(compaction_threshold=50),
                token_budget=60,
            ),
        )
        assert ctx.compacted is False
        assert ctx.summary_ref is None

        # and no agent.summary node was written behind the result, either.
        view = await GraphProjection().fold(rt, _TENANT)
        part_of = await view.neighbors(
            run.id, edge_type=EdgeTypes.PART_OF, direction="in"
        )
        for neighbour in part_of:
            node = await rt.get(_TENANT, neighbour.node_id)
            assert not (isinstance(node, Node) and node.type == "agent.summary")

    asyncio.run(asyncio.wait_for(_case(), timeout=5.0))
