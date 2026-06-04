"""Invoker-owned retries: capped backoff via ManualClock, deterministic, internal.

Retries are the INVOKER's (R4), never the agent loop's: no matter how many attempts
run, the loop sees a single :class:`~vella.agent.ToolResult`. Each inter-attempt wait
sleeps on the injected :class:`~vella.agent.ManualClock` (off any worker), so the
schedule is deterministic and a test drives the backoff by advancing the clock.
``asyncio.run`` only — no pytest-asyncio.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vella.core import Node, ToolDeclaration, UnresolvedRef

from vella.agent import (
    BuiltinBinding,
    InMemoryToolInvoker,
    ManualClock,
    RetryPolicy,
    ToolData,
    ToolResult,
    agent_registry,
)

_ACTOR = UnresolvedRef(identifier="vella:test")


def _tool_node(retry: RetryPolicy) -> Node[Any, Any]:
    agent_registry()
    return Node.from_data(
        ToolData(
            declaration=ToolDeclaration(name="flaky", description="d"),
            binding=BuiltinBinding(registry_key="flaky"),
            retry=retry,
        ),
        name="flaky",
        created_by=_ACTOR,
    )


def test_retry_drives_off_clock_and_loop_sees_one_result() -> None:
    clock = ManualClock()
    attempts: list[float] = []

    async def _impl(args: dict[str, object]) -> ToolResult:
        # Record the clock time of each attempt; succeed on the 3rd.
        attempts.append(clock.now())
        if len(attempts) < 3:
            return ToolResult(is_error=True, error_kind="Transient")
        return ToolResult(content="ok")

    invoker = InMemoryToolInvoker({"flaky": _impl}, clock=clock)
    node = _tool_node(
        RetryPolicy(max_attempts=3, backoff_base=1.0, backoff_factor=2.0, backoff_cap=10.0)
    )

    async def _drive() -> ToolResult:
        # The invoke coroutine parks on clock.sleep between attempts; a driver task
        # advances the clock so the backoff schedule is fully deterministic.
        task = asyncio.ensure_future(invoker.invoke(node, {}))
        # Let attempt 1 run and park on the first backoff.
        await asyncio.sleep(0)
        await clock.advance(1.0)   # base * factor**0 = 1.0 -> attempt 2
        await clock.advance(2.0)   # base * factor**1 = 2.0 -> attempt 3 (success)
        return await task

    result = asyncio.run(_drive())

    # The loop sees ONE result — the final success, not the intermediate errors.
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.content == "ok"
    # Three attempts, fired at the capped-backoff schedule (0, +1, +1+2).
    assert attempts == [0.0, 1.0, 3.0]


def test_retry_caps_the_backoff() -> None:
    clock = ManualClock()
    attempts: list[float] = []

    async def _impl(args: dict[str, object]) -> ToolResult:
        attempts.append(clock.now())
        return ToolResult(is_error=True, error_kind="Always")

    invoker = InMemoryToolInvoker({"flaky": _impl}, clock=clock)
    # base=5, factor=10, cap=6 -> waits would be 5, 50, ... but cap clamps to 6.
    node = _tool_node(
        RetryPolicy(max_attempts=3, backoff_base=5.0, backoff_factor=10.0, backoff_cap=6.0)
    )

    async def _drive() -> ToolResult:
        task = asyncio.ensure_future(invoker.invoke(node, {}))
        await asyncio.sleep(0)
        await clock.advance(5.0)   # min(5*10**0, 6) = 5 -> attempt 2
        await clock.advance(6.0)   # min(5*10**1, 6) = 6 (capped) -> attempt 3
        return await task

    result = asyncio.run(_drive())
    assert result.is_error is True
    assert result.error_kind == "Always"
    assert attempts == [0.0, 5.0, 11.0]  # final wait capped at 6, not 50


def test_no_retry_policy_is_a_single_attempt() -> None:
    clock = ManualClock()
    attempts: list[int] = []

    async def _impl(args: dict[str, object]) -> ToolResult:
        attempts.append(1)
        return ToolResult(is_error=True, error_kind="Boom")

    invoker = InMemoryToolInvoker({"flaky": _impl}, clock=clock)
    node = Node.from_data(
        ToolData(
            declaration=ToolDeclaration(name="flaky", description="d"),
            binding=BuiltinBinding(registry_key="flaky"),
            retry=None,
        ),
        name="flaky",
        created_by=_ACTOR,
    )
    result = asyncio.run(invoker.invoke(node, {}))
    assert result.is_error is True
    assert len(attempts) == 1  # no retry without a policy
