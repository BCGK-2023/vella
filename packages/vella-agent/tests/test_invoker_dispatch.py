"""Invoker dispatch by ``binding.kind``: builtin runs; the loop sees one result.

Dispatch is strictly on the tool's ``binding.kind`` — a ``builtin`` looks up its
registered callable and runs it; an ``mcp``/``http`` binding raises clearly in-gate
(those adapters are M7 extras). The agent loop sees exactly one
:class:`~vella.agent.ToolResult` per invocation. No pytest-asyncio: ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from vella.core import Node, ToolDeclaration, UnresolvedRef

from vella.agent import (
    Binding,
    BuiltinBinding,
    HTTPBinding,
    InMemoryToolInvoker,
    MCPBinding,
    ToolData,
    ToolDispatchError,
    ToolResult,
    agent_registry,
)

_ACTOR = UnresolvedRef(identifier="vella:test")


def _tool_node(binding: Binding, name: str = "t") -> Node[Any, Any]:
    # Construct against a fresh registry (never the global default) so the tool type
    # validates in isolation.
    agent_registry()
    return Node.from_data(
        ToolData(
            declaration=ToolDeclaration(name=name, description="d"),
            binding=binding,
        ),
        name=name,
        created_by=_ACTOR,
    )


def test_builtin_binding_dispatches_to_registered_callable() -> None:
    calls: list[dict[str, Any]] = []

    async def _impl(args: dict[str, Any]) -> ToolResult:
        calls.append(args)
        return ToolResult(content={"echo": args})

    invoker = InMemoryToolInvoker({"k": _impl})
    node = _tool_node(BuiltinBinding(registry_key="k"))

    result = asyncio.run(invoker.invoke(node, {"a": 1}))
    # The loop sees ONE ToolResult.
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.content == {"echo": {"a": 1}}
    assert calls == [{"a": 1}]  # invoked exactly once


def test_mcp_binding_raises_clearly_in_gate() -> None:
    invoker = InMemoryToolInvoker({})
    node = _tool_node(MCPBinding(server_node_ref=uuid4(), remote_name="r"))
    with pytest.raises(ToolDispatchError, match="mcp"):
        asyncio.run(invoker.invoke(node, {}))


def test_http_binding_raises_clearly_in_gate() -> None:
    invoker = InMemoryToolInvoker({})
    node = _tool_node(HTTPBinding(endpoint="https://example.test"))
    with pytest.raises(ToolDispatchError, match="http"):
        asyncio.run(invoker.invoke(node, {}))


def test_unregistered_builtin_key_raises() -> None:
    invoker = InMemoryToolInvoker({})
    node = _tool_node(BuiltinBinding(registry_key="missing"))
    with pytest.raises(ToolDispatchError, match="missing"):
        asyncio.run(invoker.invoke(node, {}))
