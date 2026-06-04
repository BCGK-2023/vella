"""The frozen tool-node + invoker contract (M3, locked R13) — tools-are-nodes.

A tool is a plain ``core.Node`` whose ``data`` is a :class:`ToolData`; the invoker
(:mod:`vella.agent.invoker`) is the behavior registry that turns a tool-node + args
into a :class:`ToolResult`. There is **no new envelope** — a tool, an MCP server,
and a recorded tool call are all ordinary registered core node types.

Shapes (locked, from the spec's "Tool-node + Invoker Contract"):

* :class:`ToolData` — ``{declaration, binding, hints, retry}``, registered
  ``@node_type("tool")``. ``declaration`` is a :class:`~vella.core.ToolDeclaration`
  (name/description/parameters/returns); ``binding`` is the discriminated
  :data:`Binding` union; ``hints`` is the :class:`ToolHints` legibility block;
  ``retry`` is an optional :class:`RetryPolicy` the **invoker** owns (R4 — retries
  never live in the agent loop).
* :data:`Binding` — discriminated by ``kind``: :class:`BuiltinBinding` |
  :class:`MCPBinding` | :class:`HTTPBinding`. The invoker dispatches on ``kind``.
* :class:`ToolHints` — ``{result_hint, error_hints, default_error_hint}``.
  ``error_hints`` order is **semantic** (first match wins) and is therefore a
  ``tuple`` that is NEVER sorted — matching is order-sensitive.
* :class:`ErrorHint` — ``{match, hint}``; ``match`` is matched against a
  :class:`ToolResult` ``error_kind``.
* :class:`RetryPolicy` — capped-backoff params (``max_attempts`` + base/factor/cap).
* :class:`ToolResult` — the invoker's output ``{content, is_error, error_kind}``;
  ``error_kind`` keys the hint lookup.

Two more node-type data classes complete self-hosting:

* :class:`MCPServerData` — registered ``@node_type("mcp_server")``
  (``{endpoint, transport, config_ref}``), the node an :class:`MCPBinding` refers to.
* :class:`ToolCallData` — registered ``@node_type("agent.tool_call")``
  (``{tool_ref, args, intent, result, error_kind, hint}``), the durable record the
  harness writes through runtime verbs for every invocation.

All models are frozen :class:`~vella.core.VellaModel` (frozen + ``extra='forbid'``),
so equality/round-trip is via ``model_dump(mode="json")``. ``error_hints`` is the
one tuple here whose order is the contract; no other field is set-derived.

Test isolation (pre-mortem #2): registration writes into the shared
``default_registry`` once at import; tests construct nodes against a fresh
``Registry()`` from :func:`vella.agent.agent_registry`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union
from uuid import UUID

from pydantic import Field
from vella.core import Registry, ToolDeclaration, VellaModel, default_registry, node_type

# The stable type names frozen by the surface tripwire from M3 onward. ``tool`` and
# ``mcp_server`` are substrate-level (bare names, matching the plan's reservation);
# ``agent.tool_call`` is a cognition record (agent.* prefix).
TOOL_TYPE = "tool"
"""Registered type name for a tool node (``data`` is a :class:`ToolData`)."""

MCP_SERVER_TYPE = "mcp_server"
"""Registered type name for an MCP-server node (``data`` is a :class:`MCPServerData`)."""

TOOL_CALL_TYPE = "agent.tool_call"
"""Registered type name for a recorded tool-call node (``data`` is :class:`ToolCallData`)."""


class BuiltinBinding(VellaModel):
    """Binds a tool to an in-process callable in the invoker's registry.

    Attributes:
        kind: The discriminator literal (always ``"builtin"``).
        registry_key: The key the :class:`~vella.agent.InMemoryToolInvoker` looks up
            to find the async callable that implements the tool.
    """

    kind: Literal["builtin"] = "builtin"
    registry_key: str


class MCPBinding(VellaModel):
    """Binds a tool to a remote tool exposed by an ``mcp_server`` node.

    Attributes:
        kind: The discriminator literal (always ``"mcp"``).
        server_node_ref: Id of the :class:`MCPServerData` node providing the
            transport.
        remote_name: The tool's name as the MCP server exposes it.
    """

    kind: Literal["mcp"] = "mcp"
    server_node_ref: UUID
    remote_name: str


class HTTPBinding(VellaModel):
    """Binds a tool to an HTTP endpoint (adapter reserved for a later milestone).

    Attributes:
        kind: The discriminator literal (always ``"http"``).
        endpoint: The fully-qualified URL the tool invokes.
    """

    kind: Literal["http"] = "http"
    endpoint: str


Binding = Annotated[
    Union[BuiltinBinding, MCPBinding, HTTPBinding],
    Field(discriminator="kind"),
]
"""A tool's transport binding — discriminated by ``kind``.

The same idiom as core ``Overlay``/``Actuator``: a closed union tagged by a
``Literal`` ``kind`` field, resolved by pydantic via ``Field(discriminator="kind")``.
The invoker dispatches on ``kind`` to the registered adapter.
"""


class ErrorHint(VellaModel):
    """One error-interpretation rule: match an ``error_kind``, supply a hint.

    Attributes:
        match: The ``error_kind`` (or pattern) this rule applies to.
        hint: The interpretation guidance surfaced to the model when ``match`` is
            the result's ``error_kind``.
    """

    match: str
    hint: str


class ToolHints(VellaModel):
    """Legibility hints a tool-node carries for result/error interpretation.

    ``error_hints`` order is **semantic** — resolution takes the FIRST matching
    entry — so it is a ``tuple`` that is NEVER sorted. (This is the deliberate
    exception to the package's sort-set-derived-values rule, like a message's
    content sequence: the order is the contract, not a set-derived artifact.)

    Attributes:
        result_hint: Guidance attached to a successful result, or ``None``.
        error_hints: Ordered rules tried first-match-wins against a result's
            ``error_kind`` (order is semantic; never sorted).
        default_error_hint: Fallback guidance when no ``error_hints`` entry matches,
            or ``None``.
    """

    result_hint: Optional[str] = None
    error_hints: tuple[ErrorHint, ...] = ()
    default_error_hint: Optional[str] = None


class RetryPolicy(VellaModel):
    """Capped-backoff retry parameters the **invoker** owns (R4).

    The invoker, not the agent loop, runs the retry schedule: it sleeps on the
    injected :class:`~vella.agent.Clock` between attempts so a
    :class:`~vella.agent.ManualClock` makes the schedule deterministic. The agent
    loop sees a single :class:`ToolResult` regardless of how many attempts ran.

    The per-attempt wait is ``min(backoff_base * backoff_factor ** attempt,
    backoff_cap)`` (capped) for attempts ``0..max_attempts-2`` (the final attempt is
    not followed by a wait).

    Attributes:
        max_attempts: Total attempts including the first (``>= 1``).
        backoff_base: The first inter-attempt wait in seconds of clock time.
        backoff_factor: The multiplier applied per subsequent attempt.
        backoff_cap: The hard ceiling on any single inter-attempt wait.
    """

    max_attempts: int = Field(default=1, ge=1)
    backoff_base: float = Field(default=0.0, ge=0.0)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    backoff_cap: float = Field(default=60.0, ge=0.0)


class ToolData(VellaModel):
    """Frozen data payload of a ``tool`` node (``@node_type("tool")``).

    Attributes:
        declaration: The capability schema the model is offered (a
            :class:`~vella.core.ToolDeclaration`; ``declaration.name`` is what a
            :class:`~vella.agent.ToolUseBlock` names).
        binding: The discriminated :data:`Binding` the invoker dispatches on.
        hints: The :class:`ToolHints` the harness resolves a result/error against.
        retry: An optional :class:`RetryPolicy` the invoker owns, or ``None`` for a
            single attempt.
    """

    declaration: ToolDeclaration
    binding: Binding
    hints: ToolHints = Field(default_factory=ToolHints)
    retry: Optional[RetryPolicy] = None


class ToolResult(VellaModel):
    """The invoker's output for one tool invocation (after any internal retries).

    The agent loop sees exactly one of these per :class:`~vella.agent.ToolUseBlock`,
    no matter how many attempts the invoker ran. ``error_kind`` is the key the hint
    resolver (:mod:`vella.agent._hints`) looks up in the tool-node's ``error_hints``.

    Attributes:
        content: The tool's output payload (free-form; serialized as-is).
        is_error: Whether the invocation ultimately failed.
        error_kind: A short stable classifier of the failure (keys hint lookup), or
            ``None`` on success.
    """

    content: Any = None
    is_error: bool = False
    error_kind: Optional[str] = None


# --- node-type data that completes self-hosting (mcp_server + agent.tool_call) ---


class MCPServerData(VellaModel):
    """Frozen data payload of an ``mcp_server`` node (``@node_type("mcp_server")``).

    The node an :class:`MCPBinding` refers to; the real MCP adapter (an out-of-gate
    extra, M7) reads its transport details from here.

    Attributes:
        endpoint: The server's transport address (URL / command, per ``transport``).
        transport: The transport kind the server speaks.
        config_ref: An optional opaque reference to out-of-band config/credentials.
    """

    endpoint: str
    transport: Literal["stdio", "sse", "http"] = "stdio"
    config_ref: Optional[str] = None


class ToolCallData(VellaModel):
    """Frozen data payload of an ``agent.tool_call`` node (the durable call record).

    Written through runtime verbs for every invocation and linked ``PART_OF`` the
    step, this is what makes a tool call replayable/observable: the resolved
    ``hint`` and ``error_kind`` live here, not only on the in-flight
    :class:`~vella.agent.ToolResultBlock`. ``args`` is the assembled call input;
    ``result`` is the :class:`ToolResult` content the loop saw.

    Attributes:
        tool_ref: Id of the ``tool`` node that was invoked.
        args: The assembled call arguments (the :class:`~vella.agent.ToolUseBlock`
            ``input``).
        intent: The model's one-sentence UX narration for the call.
        result: The invoker's :class:`ToolResult` content (free-form), or ``None``.
        error_kind: The failure classifier, or ``None`` on success.
        hint: The hint the resolver produced for this result, or ``None``.
    """

    tool_ref: UUID
    args: dict[str, Any] = Field(default_factory=dict)
    intent: str = ""
    result: Any = None
    error_kind: Optional[str] = None
    hint: Optional[str] = None


def register_tool_types(registry: Registry) -> Registry:
    """Register the M3 tool/server/call node type-specs into ``registry``; return it.

    Binds :class:`ToolData`, :class:`MCPServerData`, and :class:`ToolCallData` under
    their stable names. Tests pass a fresh ``Registry()`` here for isolation rather
    than touching the global ``default_registry`` (pre-mortem #2).

    Args:
        registry: The registry to populate (mutated in place).

    Returns:
        The same ``registry``, now populated, for call-site chaining.
    """
    node_type(TOOL_TYPE, registry=registry)(ToolData)
    node_type(MCP_SERVER_TYPE, registry=registry)(MCPServerData)
    node_type(TOOL_CALL_TYPE, registry=registry)(ToolCallData)
    return registry


# Register once, at import, into core's process-wide default registry — the single
# idempotent module-import side effect, mirroring :mod:`vella.agent.types`. This also
# stamps each class's ``__vella_type__`` so ``Node.from_data`` resolves the type name.
register_tool_types(default_registry)
