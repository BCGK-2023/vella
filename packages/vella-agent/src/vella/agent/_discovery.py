"""Graph-driven, edge-tied tool discovery + idempotent baseline seeding (M3).

An agent run's available toolset is **nodes the run can see**, never a privileged
internal table:

* **seeded baseline "system" tool-nodes** — the universal toolset the agent layer
  ships, created as ``tool`` nodes at startup. Seeding is **idempotent** via
  ``runtime.upsert`` keyed by a stable ``IntegrationBinding(plugin, external_id)``:
  seeding twice yields the same node ids, never duplicates.
* **per-run tool-nodes linked via a ``HAS_TOOL`` edge** — any ``tool`` node the run
  is linked to by an edge of type :data:`HAS_TOOL_EDGE` (a custom edge string; core
  emits no warning for it — see ``tests/test_has_tool_no_warning.py``).

Discovery reads the graph, never a runtime scan: it folds the runtime into a
:class:`~vella.graph.GraphView` and queries
``GraphView.neighbors(run, edge_type=HAS_TOOL_EDGE, direction="out")`` with an
**explicit direction** (a ``HAS_TOOL`` edge points run -> tool, so the run's tools are
its ``"out"`` neighbours; never ``"both"`` — that would be nondeterministic/wrong).
The baseline nodes are returned alongside; the union is returned as a deterministic
**sorted** (by ``str(node_id)``) tuple so the assembled toolset is byte-stable.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
from uuid import UUID

from vella.core import IntegrationBinding, Node, UnresolvedRef
from vella.graph import GraphProjection
from vella.runtime import Runtime

from .tool import ToolData

HAS_TOOL_EDGE = "has_tool"
"""The custom edge string tying a run to a per-run tool node (run -> tool).

NOT a core ``EdgeTypes`` constant (adding one is forbidden — no edits pushed into
core). Core's unknown-edge-type validator stays silent on it (``"has_tool"`` is too
dissimilar from any ``EdgeTypes`` constant to clear the ``difflib`` cutoff), so no
suppression is needed; ``tests/test_has_tool_no_warning.py`` guards that property.
"""

SYSTEM_TOOL_PLUGIN = "vella.agent.system"
"""The idempotency ``plugin`` namespacing baseline "system" tool bindings."""


def _agent_actor() -> UnresolvedRef:
    """The default authorship ref stamped on agent-written tool nodes."""
    return UnresolvedRef(identifier="vella:agent")


def _system_binding(tool_name: str) -> IntegrationBinding:
    """The stable idempotency binding for a baseline system tool.

    Args:
        tool_name: The tool's ``declaration.name`` (its ``external_id``).

    Returns:
        The :class:`~vella.core.IntegrationBinding` keying the ``upsert`` so reseeding
        the same baseline tool resolves the existing node rather than creating a new
        one.
    """
    return IntegrationBinding(plugin=SYSTEM_TOOL_PLUGIN, external_id=tool_name)


async def seed_system_tools(
    runtime: Runtime,
    tools: Sequence[ToolData],
    *,
    tenant_id: str,
    created_by: Optional[UnresolvedRef] = None,
) -> list[Node[Any, Any]]:
    """Idempotently seed the baseline "system" tool-nodes; return them in seed order.

    Each tool is upserted under ``(tenant_id, SYSTEM_TOOL_PLUGIN,
    declaration.name)`` — so calling this twice with the same tools resolves the
    existing nodes (same ids) rather than creating duplicates (idempotency invariant;
    a plain ``create`` per call would violate it). The node carries the
    :class:`~vella.core.IntegrationBinding` so the runtime's ``find_by_binding`` can
    resolve it on a later seed.

    Args:
        runtime: The runtime to write through (``upsert``).
        tools: The baseline tool payloads to seed (order is preserved in the result).
        tenant_id: The tenant the tools belong to.
        created_by: Authorship ref; defaults to the agent actor.

    Returns:
        The seeded tool nodes in ``tools`` order (each id stable across reseeds).
    """
    actor = created_by or _agent_actor()
    out: list[Node[Any, Any]] = []
    for data in tools:
        binding = _system_binding(data.declaration.name)
        node = Node.from_data(
            data,
            name=data.declaration.name,
            created_by=actor,
            tenant_id=tenant_id,
            integrations=[binding],
        )
        entry = await runtime.upsert(
            tenant_id, binding.plugin, binding.external_id, node
        )
        # upsert returns the (possibly pre-existing) entity's id; re-read the
        # authoritative node so the result reflects the durable record, not the
        # candidate we may not have inserted (TRAP-1: never trust .payload shape).
        resolved = await runtime.get(tenant_id, entry.entity_id)
        out.append(resolved if isinstance(resolved, Node) else node)
    return out


async def link_run_tool(
    runtime: Runtime,
    run_id: UUID,
    tool_id: UUID,
    *,
    tenant_id: str,
    created_by: Optional[UnresolvedRef] = None,
) -> None:
    """Link ``tool_id`` to ``run_id`` with a ``HAS_TOOL`` edge (run -> tool).

    The edge direction is the discovery contract: a run's per-run tools are its
    ``HAS_TOOL`` ``"out"`` neighbours.

    Args:
        runtime: The runtime to write through (``link``).
        run_id: The owning run's node id (the edge's ``from`` node).
        tool_id: The tool node's id (the edge's ``to`` node).
        tenant_id: The tenant both nodes belong to.
        created_by: Authorship ref; defaults to the agent actor.
    """
    await runtime.link(
        tenant_id,
        run_id,
        tool_id,
        edge_type=HAS_TOOL_EDGE,
        created_by=created_by or _agent_actor(),
    )


async def discover_tools(
    runtime: Runtime,
    run_id: UUID,
    *,
    tenant_id: str,
    baseline: Sequence[Node[Any, Any]] = (),
) -> tuple[UUID, ...]:
    """Discover a run's toolset = baseline system tools + ``HAS_TOOL`` neighbours.

    Folds the runtime into a :class:`~vella.graph.GraphView` for ``tenant_id`` and
    queries the run's ``HAS_TOOL`` ``"out"`` neighbours (explicit direction — a
    ``HAS_TOOL`` edge points run -> tool; never ``"both"``). The discovered
    neighbour ids are unioned with the ``baseline`` node ids and returned as a
    deterministic **sorted** tuple (by ``str(id)``), so the assembled toolset is
    byte-stable across hash seeds and fold order.

    Args:
        runtime: The runtime whose log the graph projects (read-only).
        run_id: The run whose toolset to discover.
        tenant_id: The tenant to project.
        baseline: The seeded baseline tool nodes (from :func:`seed_system_tools`).

    Returns:
        The discovered tool-node ids, sorted by ``str(id)`` (deterministic; no
        privileged/internal entries — just the baseline union the linked neighbours).
    """
    view = await GraphProjection().fold(runtime, tenant_id)
    neighbours = await view.neighbors(
        run_id, edge_type=HAS_TOOL_EDGE, direction="out"
    )
    ids: set[UUID] = {n.id for n in baseline}
    ids.update(neighbour.node_id for neighbour in neighbours)
    # Set-derived serialized value -> sorted() for deterministic, hash-stable bytes.
    return tuple(sorted(ids, key=str))
