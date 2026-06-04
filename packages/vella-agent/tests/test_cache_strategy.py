"""Cache-capability coupling: the provider node's flag drives the strategy.

Covers the M4 "Caching capability honored" criterion (spec §6/§7 — caching ⇔
context-assembly are the same problem): a cache-capable ``provider`` node makes the
assembler mark the stable prefix as a cache breakpoint (and the canonical ``Usage``
carries cache-token fields); a non-capable node makes the assembler switch to
AGGRESSIVE compaction (no breakpoint, a tighter soft watermark). Both paths are
deterministic and green.

No ``pytest-asyncio``: each async case runs via ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vella.core import Node, UnresolvedRef
from vella.runtime import Runtime

from vella.agent import (
    AssemblyPolicy,
    CompactionPolicy,
    GraphContextAssembler,
    MessageData,
    ProviderData,
    RunData,
    TextBlock,
    Usage,
    agent_registry,
)
from vella.agent._writeback import append_message, create_run

_TENANT = "t-agent"
_ACTOR = UnresolvedRef(identifier="vella:test")


async def _provider(rt: Runtime, *, cache_capable: bool) -> Node[Any, Any]:
    node = Node.from_data(
        ProviderData(model_id="m", cache_capable=cache_capable),
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


def test_cache_capable_marks_a_stable_prefix_breakpoint() -> None:
    async def _case() -> None:
        agent_registry()
        rt = Runtime()
        prov = await _provider(rt, cache_capable=True)
        run = await _run_with_messages(rt, 3, text_len=4)
        ctx = await GraphContextAssembler().assemble(
            rt,
            run.id,
            tenant_id=_TENANT,
            provider_node=prov.id,
            policy=AssemblyPolicy(compaction=CompactionPolicy()),
        )
        # a cache breakpoint is marked: the leading prefix is declared cacheable.
        assert ctx.cacheable_prefix_len >= 1
        assert ctx.cacheable_prefix_len <= len(ctx.messages)

    asyncio.run(asyncio.wait_for(_case(), timeout=5.0))


def test_usage_cache_token_fields_are_exercised() -> None:
    """The canonical Usage carries cache-read/write fields a cache-capable run reports."""
    usage = Usage(input_tokens=100, cache_read_tokens=80, cache_write_tokens=20)
    dumped = usage.model_dump(mode="json")
    assert dumped["cache_read_tokens"] == 80
    assert dumped["cache_write_tokens"] == 20
    # cache tokens are accounted separately from the input/output budget fields.
    assert dumped["input_tokens"] == 100


def test_non_capable_switches_to_aggressive_compaction() -> None:
    """Non-capable provider: NO breakpoint AND a tighter (aggressive) watermark."""

    async def _case() -> None:
        agent_registry()
        rt = Runtime()
        # 8 * 10 = 80 cumulative; soft threshold configured at 120.
        run = await _run_with_messages(rt, 8, text_len=10)
        policy = AssemblyPolicy(
            compaction=CompactionPolicy(compaction_threshold=120),
            token_budget=1000,
        )

        cap = await _provider(rt, cache_capable=True)
        non = await _provider(rt, cache_capable=False)
        asm = GraphContextAssembler()

        # cache-capable: 80 < 120 -> does NOT compact; prefix IS a breakpoint.
        cap_ctx = await asm.assemble(
            rt, run.id, tenant_id=_TENANT, provider_node=cap.id, policy=policy
        )
        assert cap_ctx.compacted is False
        assert cap_ctx.cacheable_prefix_len >= 1

        # non-capable: the watermark is tightened to 120 // 2 = 60, so 80 >= 60 ->
        # it compacts AGGRESSIVELY; and there is NO cache breakpoint.
        non_ctx = await asm.assemble(
            rt, run.id, tenant_id=_TENANT, provider_node=non.id, policy=policy
        )
        assert non_ctx.compacted is True
        assert non_ctx.cacheable_prefix_len == 0

    asyncio.run(asyncio.wait_for(_case(), timeout=5.0))


def test_both_paths_are_deterministic() -> None:
    async def _case() -> None:
        agent_registry()
        rt = Runtime()
        run = await _run_with_messages(rt, 6, text_len=8)
        policy = AssemblyPolicy(
            compaction=CompactionPolicy(compaction_threshold=80), token_budget=1000
        )
        cap = await _provider(rt, cache_capable=True)
        non = await _provider(rt, cache_capable=False)
        asm = GraphContextAssembler()

        for prov in (cap, non):
            a = await asm.assemble(
                rt, run.id, tenant_id=_TENANT, provider_node=prov.id, policy=policy
            )
            b = await asm.assemble(
                rt, run.id, tenant_id=_TENANT, provider_node=prov.id, policy=policy
            )
            # message sequence + cache metadata are byte-identical across calls
            # (compare via model_dump — core's registry PrivateAttr breaks ==).
            assert a.model_dump(mode="json") == b.model_dump(mode="json")

    asyncio.run(asyncio.wait_for(_case(), timeout=5.0))
