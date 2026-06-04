"""Network-free MCP invoker smoke (``[mcp]``, marked ``extras``).

Drives :class:`MCPToolInvoker` against an in-process stub MCP server/session (no
transport, no network): asserts it dispatches an :class:`MCPBinding` tool-node,
calls the binding's ``remote_name`` with the assembled args, and maps the stub's
``CallToolResult``-shaped response to a canonical :class:`ToolResult`. Also asserts
the dispatch-by-``kind`` guard refuses a non-``mcp`` binding (mutation guard (c):
routing an ``MCPBinding`` through the in-gate invoker, or a builtin binding through
THIS one, must fail by kind).

The ``[mcp]`` extra is only ``importorskip``-ed to keep parity with the OpenRouter
smoke (the invoker's session factory is injected as a stub here, so the test itself
needs no live ``mcp`` session).
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

pytest.importorskip("mcp")

from vella.agent import (  # noqa: E402
    BuiltinBinding,
    MCPBinding,
    MCPServerData,
    ManualClock,
    RetryPolicy,
    ToolData,
    ToolDispatchError,
    agent_registry,
)
from vella.agent.adapters.mcp_invoker import MCPToolInvoker  # noqa: E402
from vella.core import Node, ToolDeclaration, UnresolvedRef  # noqa: E402


class _StubContent:
    """A single MCP text content part (the ``.text`` shape the adapter reads)."""

    def __init__(self, text: str) -> None:
        self.text = text


class _StubCallResult:
    """An ``mcp`` ``CallToolResult``-shaped response (``content`` + ``isError``)."""

    def __init__(self, text: str, *, is_error: bool = False) -> None:
        self.content = [_StubContent(text)]
        self.isError = is_error


class _StubSession:
    """An in-process MCP session (also an async context manager torn down on exit).

    The real ``mcp`` SDK session is an async context manager whose ``call_tool``
    returns a ``CallToolResult``; this stub mirrors that shape exactly so the
    adapter's ``async with await factory(server)`` / ``session.call_tool`` path runs
    unchanged, with no transport and no network.
    """

    def __init__(self, server: MCPServerData, calls: list[tuple[str, dict[str, Any]]]):
        self._server = server
        self._calls = calls
        self.closed = False

    async def __aenter__(self) -> "_StubSession":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.closed = True

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _StubCallResult:
        self._calls.append((name, arguments))
        return _StubCallResult(f"{self._server.endpoint}:{name}:{arguments['q']}")


def _mcp_tool_node(server_id: Any) -> Node[Any, Any]:
    """Build a tool node bound by ``MCPBinding`` to ``server_id``."""
    agent_registry()  # ensure the agent types are registered
    return Node.from_data(
        ToolData(
            declaration=ToolDeclaration(name="search", description="remote search"),
            binding=MCPBinding(server_node_ref=server_id, remote_name="search"),
            retry=RetryPolicy(max_attempts=2),
        ),
        name="mcp-search",
        created_by=UnresolvedRef(identifier="vella:test"),
    )


def test_mcp_invoker_dispatches_binding_and_maps_result() -> None:
    server = MCPServerData(endpoint="stdio://srv", transport="stdio")
    server_id = uuid4()
    calls: list[tuple[str, dict[str, Any]]] = []

    sessions: list[_StubSession] = []

    async def _resolver(node_id: Any) -> MCPServerData:
        assert node_id == server_id
        return server

    async def _factory(srv: MCPServerData) -> _StubSession:
        # The adapter does `async with await factory(server)`: return the session
        # (itself an async context manager) so it is entered + torn down per invoke.
        session = _StubSession(srv, calls)
        sessions.append(session)
        return session

    invoker = MCPToolInvoker(_resolver, _factory, clock=ManualClock())

    node = _mcp_tool_node(server_id)
    result = asyncio.run(invoker.invoke(node, {"q": "vella"}))

    assert calls == [("search", {"q": "vella"})]
    assert result.is_error is False
    assert result.error_kind is None
    assert result.content == "stdio://srv:search:vella"
    # The session was opened and torn down (the adapter leaks no async context).
    assert sessions and sessions[0].closed is True


def test_mcp_invoker_refuses_non_mcp_binding_by_kind() -> None:
    invoker = MCPToolInvoker(
        _unused_resolver, _unused_factory, clock=ManualClock()
    )
    agent_registry()
    builtin_node = Node.from_data(
        ToolData(
            declaration=ToolDeclaration(name="echo", description="echo"),
            binding=BuiltinBinding(registry_key="echo"),
        ),
        name="builtin-echo",
        created_by=UnresolvedRef(identifier="vella:test"),
    )
    with pytest.raises(ToolDispatchError):
        asyncio.run(invoker.invoke(builtin_node, {}))


async def _unused_resolver(node_id: Any) -> MCPServerData:  # pragma: no cover
    raise AssertionError("resolver must not run for a refused binding")


async def _unused_factory(server: MCPServerData) -> Any:  # pragma: no cover
    raise AssertionError("factory must not run for a refused binding")
