"""Materialize a run's cognition through runtime verbs ONLY (no privileged path).

Every write here goes through the published ``Runtime`` action contract — the run,
step, and message nodes via ``create``; their ``PART_OF`` links via ``link``; the
token-level reasoning trace via ``emit_telemetry`` (an ``observe_only`` entry that
reaches the log and observers but never bumps the entity's state-table version,
locked decision #1). There is NO direct store access, NO private import, and the
authoritative state is always read back via ``runtime.get()`` / ``history()`` — a
``LogEntry.payload`` is adapter-dependent and is never reconstructed from (TRAP-1).

The node-construction door is ``Node.from_data``: the ``*Data`` classes carry their
``__vella_type__`` (stamped by :mod:`vella.agent.types`'s registration), so the run
projection is built entirely from registered type-specs — no new ``Node`` subclass.
"""

from __future__ import annotations

from typing import Any, Optional, Union
from uuid import UUID

from vella.core import EdgeTypes, Node, UnresolvedRef
from vella.runtime import LogEntry, Runtime

from .tool import ToolCallData
from .types import MessageData, RunData, StepData

_CreatedBy = Union[UUID, UnresolvedRef]


def _agent_actor() -> UnresolvedRef:
    """The default authorship ref stamped on agent-written nodes."""
    return UnresolvedRef(identifier="vella:agent")


async def create_run(
    runtime: Runtime,
    data: RunData,
    *,
    name: str,
    tenant_id: str,
    created_by: Optional[_CreatedBy] = None,
) -> Node[Any, Any]:
    """Create the ``agent.run`` node through ``runtime.create`` and return it.

    Args:
        runtime: The runtime to write through.
        data: The frozen run payload.
        name: Human-facing node name.
        tenant_id: The tenant the run belongs to.
        created_by: Authorship ref; defaults to the agent actor.

    Returns:
        The constructed run node (its id is the parent for steps/messages).
    """
    node = Node.from_data(
        data, name=name, created_by=created_by or _agent_actor(), tenant_id=tenant_id
    )
    await runtime.create(node)
    return node


async def append_step(
    runtime: Runtime,
    run_id: UUID,
    data: StepData,
    *,
    name: str,
    tenant_id: str,
    created_by: Optional[_CreatedBy] = None,
) -> Node[Any, Any]:
    """Create an ``agent.step`` node and link it ``PART_OF`` the run.

    Both writes go through runtime verbs (``create`` then ``link`` with
    ``edge_type=EdgeTypes.PART_OF``).

    Args:
        runtime: The runtime to write through.
        run_id: The owning run's node id.
        data: The frozen step payload.
        name: Human-facing node name.
        tenant_id: The tenant the step belongs to.
        created_by: Authorship ref; defaults to the agent actor.

    Returns:
        The constructed step node.
    """
    actor = created_by or _agent_actor()
    node = Node.from_data(data, name=name, created_by=actor, tenant_id=tenant_id)
    await runtime.create(node)
    await runtime.link(
        tenant_id, node.id, run_id, edge_type=EdgeTypes.PART_OF, created_by=actor
    )
    return node


async def append_message(
    runtime: Runtime,
    run_id: UUID,
    data: MessageData,
    *,
    name: str,
    tenant_id: str,
    created_by: Optional[_CreatedBy] = None,
) -> Node[Any, Any]:
    """Create an ``agent.message`` node and link it ``PART_OF`` the run.

    Both writes go through runtime verbs (``create`` then ``link`` with
    ``edge_type=EdgeTypes.PART_OF``).

    Args:
        runtime: The runtime to write through.
        run_id: The owning run's node id.
        data: The frozen message payload.
        name: Human-facing node name.
        tenant_id: The tenant the message belongs to.
        created_by: Authorship ref; defaults to the agent actor.

    Returns:
        The constructed message node.
    """
    actor = created_by or _agent_actor()
    node = Node.from_data(data, name=name, created_by=actor, tenant_id=tenant_id)
    await runtime.create(node)
    await runtime.link(
        tenant_id, node.id, run_id, edge_type=EdgeTypes.PART_OF, created_by=actor
    )
    return node


async def append_tool_call(
    runtime: Runtime,
    step_id: UUID,
    data: ToolCallData,
    *,
    name: str,
    tenant_id: str,
    created_by: Optional[_CreatedBy] = None,
) -> Node[Any, Any]:
    """Create an ``agent.tool_call`` node and link it ``PART_OF`` the step.

    This is the write that completes self-hosting: every invocation lands a durable
    record (``tool_ref``, ``args``, ``intent``, ``result``, ``error_kind``, resolved
    ``hint``) through runtime verbs, linked ``PART_OF`` its step — so a tool call is
    replayable/observable from the graph, not only live on a
    :class:`~vella.agent.ToolResultBlock`. Both writes go through runtime verbs
    (``create`` then ``link`` with ``edge_type=EdgeTypes.PART_OF``).

    Args:
        runtime: The runtime to write through.
        step_id: The owning step's node id.
        data: The frozen tool-call payload.
        name: Human-facing node name.
        tenant_id: The tenant the call belongs to.
        created_by: Authorship ref; defaults to the agent actor.

    Returns:
        The constructed tool-call node.
    """
    actor = created_by or _agent_actor()
    node = Node.from_data(data, name=name, created_by=actor, tenant_id=tenant_id)
    await runtime.create(node)
    await runtime.link(
        tenant_id, node.id, step_id, edge_type=EdgeTypes.PART_OF, created_by=actor
    )
    return node


async def emit_reasoning_trace(
    runtime: Runtime,
    run_id: UUID,
    *,
    tenant_id: str,
    payload: dict[str, Any],
) -> LogEntry:
    """Emit the run's reasoning trace as an ``observe_only`` telemetry entry.

    The trace reaches the log and live observers but never touches the run's
    state-table version (locked decision #1) — that is what makes the token-level
    trace non-bloating yet replayable. Routes through ``runtime.emit_telemetry``;
    the returned entry's ``transition`` is ``observe_only``.

    Args:
        runtime: The runtime to write through.
        run_id: The run node the trace is about.
        tenant_id: The run's tenant.
        payload: The trace payload (free-form telemetry, not state).

    Returns:
        The appended ``observe_only`` log entry.
    """
    return await runtime.emit_telemetry(tenant_id, run_id, payload)
