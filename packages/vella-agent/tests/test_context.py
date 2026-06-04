"""``GraphContextAssembler`` composition: stable prefix + volatile tail + recall.

Covers the M4 "Context assembly" criterion: the assembler perceives the run through
the graph and composes ``[stable prefix] + [volatile tail]``; the prefix is
byte-stable across turns (the cache-stability property); recall is graph-relationship
only; ordering is deterministic (turn order, never hash order).

No ``pytest-asyncio``: each async case runs via ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vella.core import EdgeTypes, Node, UnresolvedRef
from vella.runtime import Runtime

from vella.agent import (
    AssemblyPolicy,
    CompactionPolicy,
    GraphContextAssembler,
    Message,
    MessageData,
    ProviderData,
    RunData,
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


async def _provider(rt: Runtime, *, cache_capable: bool) -> Node[Any, Any]:
    node = Node.from_data(
        ProviderData(model_id="m", cache_capable=cache_capable),
        name="provider",
        created_by=_ACTOR,
        tenant_id=_TENANT,
    )
    await rt.create(node)
    return node


def test_prefix_plus_tail_composition_in_turn_order() -> None:
    async def _case() -> None:
        agent_registry()
        rt = Runtime()
        run = await create_run(
            rt, RunData(goal="find vella"), name="run", tenant_id=_TENANT
        )
        prov = await _provider(rt, cache_capable=True)
        for i in range(3):
            await append_message(
                rt,
                run.id,
                MessageData(role="user", content=(TextBlock(text=f"msg{i}"),)),
                name=f"m{i}",
                tenant_id=_TENANT,
            )
        ctx = await GraphContextAssembler().assemble(
            rt,
            run.id,
            tenant_id=_TENANT,
            provider_node=prov.id,
            policy=AssemblyPolicy(compaction=CompactionPolicy()),
        )
        roles = [m.role for m in ctx.messages]
        texts = [_text(m) for m in ctx.messages]
        # system prompt prefix, then the three user turns in creation (turn) order.
        assert roles == ["system", "user", "user", "user"]
        assert texts == ["find vella", "msg0", "msg1", "msg2"]
        # the stable prefix is exactly the leading system message.
        assert ctx.cacheable_prefix_len == 1
        assert texts[: ctx.cacheable_prefix_len] == ["find vella"]

    asyncio.run(asyncio.wait_for(_case(), timeout=5.0))


def test_prefix_is_byte_stable_across_turns() -> None:
    """The stable prefix is byte-identical as the conversation grows (cache-stability)."""

    async def _case() -> None:
        agent_registry()
        rt = Runtime()
        run = await create_run(rt, RunData(goal="goal"), name="run", tenant_id=_TENANT)
        prov = await _provider(rt, cache_capable=True)
        asm = GraphContextAssembler()
        policy = AssemblyPolicy(compaction=CompactionPolicy(pin=("system",)))

        prefixes: list[str] = []
        for turn in range(3):
            await append_message(
                rt,
                run.id,
                MessageData(role="user", content=(TextBlock(text=f"turn{turn}"),)),
                name=f"u{turn}",
                tenant_id=_TENANT,
            )
            ctx = await asm.assemble(
                rt, run.id, tenant_id=_TENANT, provider_node=prov.id, policy=policy
            )
            prefix = ctx.messages[: ctx.cacheable_prefix_len]
            prefixes.append(
                str([m.model_dump(mode="json") for m in prefix])
            )

        # The prefix must be IDENTICAL across all turns even though the tail grew —
        # no per-turn value (token count, turn index) leaked into the prefix.
        assert len(set(prefixes)) == 1

    asyncio.run(asyncio.wait_for(_case(), timeout=5.0))


def test_recall_is_graph_relationships_only() -> None:
    """Recall surfaces entities reached by graph edges; depth 0 recalls nothing."""

    async def _case() -> None:
        agent_registry()
        rt = Runtime()
        run = await create_run(rt, RunData(goal="g"), name="run", tenant_id=_TENANT)
        prov = await _provider(rt, cache_capable=True)
        msg = await append_message(
            rt,
            run.id,
            MessageData(role="user", content=(TextBlock(text="see acme"),)),
            name="u0",
            tenant_id=_TENANT,
        )
        # An entity the message MENTIONS — a graph relationship, no vector.
        entity = Node.from_data(
            RunData(goal="acme corp"), name="acme-entity", created_by=_ACTOR,
            tenant_id=_TENANT,
        )
        await rt.create(entity)
        await rt.link(
            _TENANT, msg.id, entity.id, edge_type=EdgeTypes.MENTIONED_IN,
            created_by=_ACTOR,
        )

        asm = GraphContextAssembler()
        # depth 0: no recall — only system + the one user turn.
        ctx0 = await asm.assemble(
            rt, run.id, tenant_id=_TENANT, provider_node=prov.id,
            policy=AssemblyPolicy(compaction=CompactionPolicy(recall_depth=0)),
        )
        assert all("recall:" not in _text(m) for m in ctx0.messages)

        # depth 1: the mentioned entity is recalled into the prefix via the graph.
        ctx1 = await asm.assemble(
            rt, run.id, tenant_id=_TENANT, provider_node=prov.id,
            policy=AssemblyPolicy(compaction=CompactionPolicy(recall_depth=1)),
        )
        recalls = [_text(m) for m in ctx1.messages if "recall:" in _text(m)]
        assert recalls == ["recall:acme-entity"]

    asyncio.run(asyncio.wait_for(_case(), timeout=5.0))


def test_assembly_is_deterministic_across_repeated_calls() -> None:
    """Two assemblies of the same run produce byte-identical message sequences."""

    async def _case() -> None:
        agent_registry()
        rt = Runtime()
        run = await create_run(rt, RunData(goal="g"), name="run", tenant_id=_TENANT)
        prov = await _provider(rt, cache_capable=True)
        for i in range(5):
            await append_message(
                rt,
                run.id,
                MessageData(role="assistant", content=(TextBlock(text=f"a{i}"),)),
                name=f"a{i}",
                tenant_id=_TENANT,
            )
        asm = GraphContextAssembler()
        policy = AssemblyPolicy(compaction=CompactionPolicy())
        a = await asm.assemble(
            rt, run.id, tenant_id=_TENANT, provider_node=prov.id, policy=policy
        )
        b = await asm.assemble(
            rt, run.id, tenant_id=_TENANT, provider_node=prov.id, policy=policy
        )
        assert a.model_dump(mode="json") == b.model_dump(mode="json")

    asyncio.run(asyncio.wait_for(_case(), timeout=5.0))
