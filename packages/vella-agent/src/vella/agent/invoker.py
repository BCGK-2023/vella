"""The ``ToolInvoker`` seam + the in-gate reference :class:`InMemoryToolInvoker`.

This is the behavior seam of the three (``ModelProvider`` / ``ToolInvoker`` /
``ContextAssembler``). A :class:`ToolInvoker` turns a tool-node + assembled args into
exactly one :class:`~vella.agent.ToolResult` — dispatching on the tool's
``binding.kind`` and owning any retries internally (R4). The agent loop never sees a
retry: no matter how many attempts run, it gets a single :class:`ToolResult`.

The :class:`ToolInvoker` Protocol is **structural** (like runtime's ``Store`` and the
``ModelProvider``): an adapter satisfies it by shape, so the in-gate
:class:`InMemoryToolInvoker` and a future out-of-gate MCP adapter are interchangeable
to the interpreter without a common base class.

:class:`InMemoryToolInvoker` is the deterministic reference impl: a registry of
in-process async callables keyed by a :class:`~vella.agent.BuiltinBinding`
``registry_key``. It dispatches **only** ``builtin`` bindings; an ``mcp``/``http``
binding raises a clear :class:`ToolDispatchError` in-gate (those adapters are M7).
Retries follow the tool's :class:`~vella.agent.RetryPolicy` with capped backoff, and
**every inter-attempt wait sleeps on the injected** :class:`~vella.agent.Clock` —
OFF any worker — so a :class:`~vella.agent.ManualClock` makes the schedule fully
deterministic and a test drives the backoff by advancing the clock.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional, Protocol, runtime_checkable

from vella.core import Node

from .clock import Clock, SystemClock
from .tool import BuiltinBinding, RetryPolicy, ToolData, ToolResult

ToolCallable = Callable[[dict[str, Any]], Awaitable[ToolResult]]
"""A builtin tool implementation: ``async (args) -> ToolResult``.

Registered under a :class:`~vella.agent.BuiltinBinding` ``registry_key``. It MAY
raise — the invoker catches an exception, classifies it (``error_kind =
type(exc).__name__``), and retries per the tool's policy; it MAY also return a
:class:`~vella.agent.ToolResult` with ``is_error=True`` to signal a soft failure
that the invoker likewise retries.
"""


class ToolDispatchError(Exception):
    """Raised when an invoker cannot dispatch a tool's ``binding.kind``.

    The in-gate :class:`InMemoryToolInvoker` raises this for an ``mcp``/``http``
    binding (those adapters ship as out-of-gate extras at M7) and for a ``builtin``
    binding whose ``registry_key`` is not registered.
    """


@runtime_checkable
class ToolInvoker(Protocol):
    """The behavior seam — structurally typed, retries internal.

    An adapter satisfies this by shape alone (no inheritance), exactly like
    runtime's ``Store`` and the ``ModelProvider``. :meth:`invoke` dispatches a
    tool-node by its ``binding.kind`` and returns the single
    :class:`~vella.agent.ToolResult` the agent loop reasons about; any retry/backoff
    is the invoker's own concern (R4), never the loop's.
    """

    async def invoke(self, tool_node: Node[Any, Any], args: dict[str, Any]) -> ToolResult:
        """Invoke ``tool_node`` with ``args``; return one canonical result.

        Args:
            tool_node: The ``tool`` node whose ``data`` is a
                :class:`~vella.agent.ToolData` (its ``binding`` selects the adapter).
            args: The assembled call arguments (a
                :class:`~vella.agent.ToolUseBlock` ``input``).

        Returns:
            Exactly one :class:`~vella.agent.ToolResult` — after any internal
            retries, so the agent loop sees a single outcome.
        """
        ...


class InMemoryToolInvoker:
    """The in-gate reference :class:`ToolInvoker`: builtin registry + Clock backoff.

    Holds a registry mapping a :class:`~vella.agent.BuiltinBinding` ``registry_key``
    to an async :data:`ToolCallable`. Dispatch is **by ``binding.kind``**: a
    ``builtin`` binding looks its callable up and runs it; an ``mcp``/``http``
    binding raises :class:`ToolDispatchError` (those are M7 extras). Retries follow
    the tool's :class:`~vella.agent.RetryPolicy` with capped backoff, sleeping on the
    injected :class:`~vella.agent.Clock` between attempts so the schedule is
    deterministic under a :class:`~vella.agent.ManualClock`.

    Examples:
        >>> import asyncio
        >>> from vella.agent import (
        ...     BuiltinBinding, InMemoryToolInvoker, ToolData, ToolResult, agent_registry,
        ... )
        >>> from vella.core import Node, ToolDeclaration, UnresolvedRef
        >>> async def _echo(args):
        ...     return ToolResult(content=args)
        >>> inv = InMemoryToolInvoker({"echo": _echo})
        >>> reg = agent_registry()
        >>> node = Node.from_data(
        ...     ToolData(
        ...         declaration=ToolDeclaration(name="echo", description="echo args"),
        ...         binding=BuiltinBinding(registry_key="echo"),
        ...     ),
        ...     name="echo-tool",
        ...     created_by=UnresolvedRef(identifier="vella:test"),
        ... )
        >>> asyncio.run(inv.invoke(node, {"x": 1})).content
        {'x': 1}
    """

    def __init__(
        self,
        registry: Optional[dict[str, ToolCallable]] = None,
        *,
        clock: Optional[Clock] = None,
    ) -> None:
        """Build an invoker over a builtin-callable registry and a backoff clock.

        Args:
            registry: Mapping of ``registry_key -> async (args) -> ToolResult``. A
                copy is taken so later mutation of the caller's dict does not leak in.
            clock: The :class:`~vella.agent.Clock` backoff waits sleep on; defaults
                to a real :class:`SystemClock`. Tests pass a
                :class:`~vella.agent.ManualClock` for determinism.
        """
        self._registry: dict[str, ToolCallable] = dict(registry or {})
        self._clock: Clock = clock if clock is not None else SystemClock()

    def register(self, registry_key: str, fn: ToolCallable) -> None:
        """Bind ``registry_key`` to a builtin callable (overwriting any prior).

        Args:
            registry_key: The key a :class:`~vella.agent.BuiltinBinding` names.
            fn: The async tool implementation ``(args) -> ToolResult``.
        """
        self._registry[registry_key] = fn

    async def invoke(self, tool_node: Node[Any, Any], args: dict[str, Any]) -> ToolResult:
        """Dispatch ``tool_node`` by ``binding.kind`` and return one result.

        Builtin dispatch only: an ``mcp``/``http`` binding raises
        :class:`ToolDispatchError` (M7 adapters). Retries follow the tool's
        :class:`~vella.agent.RetryPolicy` with capped backoff via the injected clock;
        the loop sees a single :class:`~vella.agent.ToolResult`.

        Args:
            tool_node: The ``tool`` node to invoke.
            args: The assembled call arguments.

        Returns:
            The single :class:`~vella.agent.ToolResult` (after internal retries).

        Raises:
            ToolDispatchError: For a non-builtin binding, or a ``builtin`` whose
                ``registry_key`` is unregistered.
        """
        data = tool_node.data
        if not isinstance(data, ToolData):
            raise ToolDispatchError(
                f"node {tool_node.id} is not a tool node (data is "
                f"{type(data).__name__}, expected ToolData)."
            )
        binding = data.binding
        # Dispatch strictly by binding.kind — the discriminator IS the routing key.
        if isinstance(binding, BuiltinBinding):
            fn = self._registry.get(binding.registry_key)
            if fn is None:
                raise ToolDispatchError(
                    f"no builtin registered for registry_key "
                    f"{binding.registry_key!r}."
                )
            return await self._invoke_with_retry(fn, args, data.retry)
        # mcp / http bindings are real adapters shipped as out-of-gate extras (M7);
        # the in-gate invoker refuses them clearly rather than silently degrading.
        raise ToolDispatchError(
            f"binding kind {binding.kind!r} is not dispatchable by the in-gate "
            f"InMemoryToolInvoker (mcp/http adapters are optional extras, M7)."
        )

    async def _invoke_with_retry(
        self,
        fn: ToolCallable,
        args: dict[str, Any],
        retry: Optional[RetryPolicy],
    ) -> ToolResult:
        """Run ``fn(args)`` with capped-backoff retries (the invoker owns this).

        A raised exception is caught and classified (``error_kind =
        type(exc).__name__``); a returned ``is_error`` result is a soft failure. On
        either, if attempts remain, the invoker sleeps the capped backoff on the
        injected clock and retries. The agent loop only ever sees the final result.

        Args:
            fn: The builtin callable to run.
            args: The call arguments.
            retry: The tool's policy, or ``None`` for a single attempt.

        Returns:
            The final :class:`~vella.agent.ToolResult` (last failure if all attempts
            fail, or the first success).
        """
        policy = retry or RetryPolicy(max_attempts=1)
        last: ToolResult = ToolResult(is_error=True, error_kind="NotInvoked")
        for attempt in range(policy.max_attempts):
            last = await self._attempt(fn, args)
            if not last.is_error:
                return last
            # Failure: wait the capped backoff (on the Clock, off any worker) before
            # the next attempt — never after the final attempt.
            if attempt < policy.max_attempts - 1:
                delay = min(
                    policy.backoff_base * (policy.backoff_factor ** attempt),
                    policy.backoff_cap,
                )
                await self._clock.sleep(delay)
        return last

    async def _attempt(self, fn: ToolCallable, args: dict[str, Any]) -> ToolResult:
        """Run one attempt, turning a raised exception into an error result.

        Args:
            fn: The builtin callable.
            args: The call arguments.

        Returns:
            The callable's :class:`~vella.agent.ToolResult`, or an
            ``is_error=True`` result classified by the exception type name.
        """
        try:
            return await fn(args)
        except Exception as exc:  # noqa: BLE001 — the invoker classifies any failure
            return ToolResult(content=str(exc), is_error=True, error_kind=type(exc).__name__)
