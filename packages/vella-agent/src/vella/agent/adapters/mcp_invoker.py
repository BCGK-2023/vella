"""``MCPToolInvoker`` — a real :class:`~vella.agent.ToolInvoker` (``[mcp]``).

An out-of-gate adapter that dispatches an :class:`~vella.agent.MCPBinding` against a
live Model Context Protocol server: it resolves the binding's ``server_node_ref`` to
the referenced ``mcp_server`` node (:class:`~vella.agent.MCPServerData`), connects
over that node's transport, calls the binding's ``remote_name`` with the assembled
arguments, and maps the MCP response to exactly one canonical
:class:`~vella.agent.ToolResult` — after any internal retries, so the agent loop sees
a single outcome (R4, the same invoker-owned retry contract the in-gate
:class:`~vella.agent.InMemoryToolInvoker` honors, sleeping its capped backoff on the
injected :class:`~vella.agent.Clock`).

``mcp`` (and any transport client) is imported **lazily** (inside the connect path,
never at module top), so importing ``vella.agent`` — or even this module — never
pulls ``mcp``: the cognition core stays five-dep and the import-boundary invariant
holds whether or not the ``[mcp]`` extra is installed.

Dispatch is strictly **by ``binding.kind``** (the discriminator IS the routing key):
this adapter handles ``mcp`` and refuses anything else with a clear
:class:`~vella.agent.ToolDispatchError` — the exact mirror of how the in-gate invoker
refuses ``mcp``/``http``. A connection/session is opened per :meth:`invoke` and torn
down before returning (no leaked async generator or task — the gate's
``filterwarnings=error::UserWarning`` would catch a leak).
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from vella.core import Node

from ..clock import Clock, SystemClock
from ..invoker import ToolDispatchError
from ..tool import MCPBinding, MCPServerData, RetryPolicy, ToolData, ToolResult

MCPSessionFactory = Callable[[MCPServerData], "Awaitable[Any]"]
"""An async factory that opens an MCP client session for an ``mcp_server`` node.

It receives the resolved :class:`~vella.agent.MCPServerData` and returns an async
context manager yielding an object with an async ``call_tool(name, arguments)``
method (the ``mcp`` SDK ``ClientSession`` shape). Injecting this is the seam a
network-free smoke test feeds an in-process stub through; the production default
builds a real ``mcp`` SDK session from the node's transport.
"""

ServerResolver = Callable[[Any], "Awaitable[MCPServerData]"]
"""Resolve an ``mcp_server`` node id (a ``UUID``) to its :class:`MCPServerData`.

The invoker is handed only the ``tool`` node; this seam reads the referenced
``mcp_server`` node (through the runtime/graph the caller owns) so the adapter takes
no privileged storage path of its own.
"""


def _result_to_canonical(call_result: Any) -> ToolResult:
    """Map an ``mcp`` ``CallToolResult`` to a canonical :class:`ToolResult`.

    The MCP response carries a ``content`` list of content parts and an ``isError``
    flag. Text parts are joined (the common single-text-part case stays a plain
    string); a failure carries the same content with ``error_kind="MCPToolError"``
    so the hint resolver keys off it like any other failure.

    Args:
        call_result: The ``mcp`` SDK ``CallToolResult`` (duck-typed: ``.content``
            iterable of parts with optional ``.text``, ``.isError`` bool).

    Returns:
        The canonical :class:`ToolResult`.
    """
    parts = getattr(call_result, "content", None) or ()
    texts: list[str] = []
    payload: list[Any] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text is not None:
            texts.append(text)
        else:
            payload.append(part)
    # Prefer the joined text (the usual MCP text-tool shape); fall back to the raw
    # non-text parts when a tool returned structured/binary content only.
    content: Any = "".join(texts) if texts else (payload or None)
    is_error = bool(getattr(call_result, "isError", False))
    return ToolResult(
        content=content,
        is_error=is_error,
        error_kind="MCPToolError" if is_error else None,
    )


class MCPToolInvoker:
    """A real :class:`~vella.agent.ToolInvoker` over MCP servers (``[mcp]``).

    Resolves a tool node's :class:`~vella.agent.MCPBinding` to its ``mcp_server``
    node via the injected :data:`ServerResolver`, opens a session via the injected
    :data:`MCPSessionFactory`, calls ``remote_name``, and maps the response to one
    canonical :class:`~vella.agent.ToolResult`. Retries follow the tool's
    :class:`~vella.agent.RetryPolicy` with capped backoff, sleeping on the injected
    :class:`~vella.agent.Clock` between attempts — the same invoker-owned retry
    contract as the in-gate reference. It satisfies the structural
    :class:`~vella.agent.ToolInvoker` Protocol by shape.

    ``mcp`` is imported lazily by the production :data:`MCPSessionFactory`, so
    importing this module does not require the ``[mcp]`` extra; a test injects an
    in-process stub factory and never touches the network.
    """

    def __init__(
        self,
        resolver: ServerResolver,
        session_factory: MCPSessionFactory,
        *,
        clock: Optional[Clock] = None,
    ) -> None:
        """Build an MCP invoker over a server resolver + session factory.

        Args:
            resolver: Resolves a binding's ``server_node_ref`` to its
                :class:`~vella.agent.MCPServerData` (the caller owns the read path).
            session_factory: Opens an MCP client session (an async context manager)
                for a resolved server node; the production factory imports ``mcp``
                lazily and builds a real SDK session, a test injects a stub.
            clock: The :class:`~vella.agent.Clock` backoff waits sleep on; defaults
                to a real :class:`SystemClock`. Tests pass a
                :class:`~vella.agent.ManualClock` for determinism.
        """
        self._resolver = resolver
        self._session_factory = session_factory
        self._clock: Clock = clock if clock is not None else SystemClock()

    async def invoke(
        self, tool_node: Node[Any, Any], args: dict[str, Any]
    ) -> ToolResult:
        """Dispatch ``tool_node`` by ``binding.kind`` and return one result.

        MCP dispatch only: a non-``mcp`` binding (or a non-tool node) raises
        :class:`~vella.agent.ToolDispatchError`, mirroring how the in-gate invoker
        refuses bindings it does not own. Retries follow the tool's
        :class:`~vella.agent.RetryPolicy` via the injected clock; the loop sees a
        single :class:`~vella.agent.ToolResult`.

        Args:
            tool_node: The ``tool`` node to invoke (its ``binding`` must be an
                :class:`~vella.agent.MCPBinding`).
            args: The assembled call arguments.

        Returns:
            The single :class:`~vella.agent.ToolResult` (after internal retries).

        Raises:
            ToolDispatchError: If the node is not a tool node, or its binding kind is
                not ``mcp``.
        """
        data = tool_node.data
        if not isinstance(data, ToolData):
            raise ToolDispatchError(
                f"node {tool_node.id} is not a tool node (data is "
                f"{type(data).__name__}, expected ToolData)."
            )
        binding = data.binding
        # Dispatch strictly by binding.kind — the discriminator IS the routing key.
        if not isinstance(binding, MCPBinding):
            raise ToolDispatchError(
                f"binding kind {binding.kind!r} is not dispatchable by the MCP "
                f"invoker (it dispatches 'mcp' bindings only)."
            )
        return await self._invoke_with_retry(binding, args, data.retry)

    async def _invoke_with_retry(
        self,
        binding: MCPBinding,
        args: dict[str, Any],
        retry: Optional[RetryPolicy],
    ) -> ToolResult:
        """Run the MCP call with capped-backoff retries (the invoker owns this).

        A raised exception is caught and classified (``error_kind =
        type(exc).__name__``); a returned ``is_error`` result is a soft failure. On
        either, if attempts remain, the invoker sleeps the capped backoff on the
        injected clock and retries. The agent loop only ever sees the final result.

        Args:
            binding: The tool's :class:`~vella.agent.MCPBinding`.
            args: The call arguments.
            retry: The tool's policy, or ``None`` for a single attempt.

        Returns:
            The final :class:`~vella.agent.ToolResult` (last failure if all attempts
            fail, or the first success).
        """
        policy = retry or RetryPolicy(max_attempts=1)
        last: ToolResult = ToolResult(is_error=True, error_kind="NotInvoked")
        for attempt in range(policy.max_attempts):
            last = await self._attempt(binding, args)
            if not last.is_error:
                return last
            if attempt < policy.max_attempts - 1:
                delay = min(
                    policy.backoff_base * (policy.backoff_factor**attempt),
                    policy.backoff_cap,
                )
                await self._clock.sleep(delay)
        return last

    async def _attempt(
        self, binding: MCPBinding, args: dict[str, Any]
    ) -> ToolResult:
        """Run one MCP call, turning a raised exception into an error result.

        Resolves the ``mcp_server`` node, opens a session via the factory (an async
        context manager torn down before returning), calls ``remote_name``, and maps
        the response. Any exception is classified by its type name.

        Args:
            binding: The tool's :class:`~vella.agent.MCPBinding`.
            args: The call arguments.

        Returns:
            The mapped :class:`~vella.agent.ToolResult`, or an ``is_error=True``
            result classified by the exception type name.
        """
        try:
            server = await self._resolver(binding.server_node_ref)
            async with await self._session_factory(server) as session:
                call_result = await session.call_tool(binding.remote_name, args)
            return _result_to_canonical(call_result)
        except Exception as exc:  # noqa: BLE001 — the invoker classifies any failure
            return ToolResult(
                content=str(exc), is_error=True, error_kind=type(exc).__name__
            )
