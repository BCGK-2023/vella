"""Self-hosting cognition node type-specs (the M1 subset).

The agent dogfoods the substrate: a run, its steps, and its messages are ordinary
``vella.core`` nodes registered through ``@node_type`` — there is no new ``Node``
subclass and no privileged envelope (locked decision #3). This module declares the
**frozen** data payloads for the cognition-record types whose shapes are stable at
M1 and registers them **once at import** into core's process-wide
``default_registry`` under stable names.

Scope (M1, per the plan's acceptance-criteria coverage map): only the
cognition-record types are registered here —

* ``agent.run`` — the run envelope (goal + status + policy/provider refs);
* ``agent.step`` — one interpreter turn within a run;
* ``agent.message`` — one conversation message (MINIMAL for the M1 smoke);
* ``agent.summary`` — a compaction record over a turn range.

The tool/provider/policy types (``agent.tool_call``, ``provider``, ``tool``,
``mcp_server``, ``loop_policy``) are **deliberately deferred** to their owning
milestones (M2/M3/M5). Registering minimal stubs now would mean retyping a shipped
field once the real contract freezes; deferring avoids that. ``loop_policy_ref`` and
``provider_ref`` on ``agent.run`` are therefore plain node-id references (``UUID``)
to nodes created in those later milestones — never embedded data.

Test isolation (pre-mortem #2): ``@node_type`` writes into the shared
``default_registry`` and *silently overwrites* on re-registration, so every test
that constructs nodes uses a FRESH ``Registry()`` from :func:`agent_registry` (or
populated via :func:`register_agent_types`), never the global default. The
module-level registration here is the single idempotent write into the default.
"""

from __future__ import annotations

from typing import Literal, Optional
from uuid import UUID

from vella.core import Registry, VellaModel, default_registry, node_type

from .turn import ContentBlock, MessageRole

RunStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]
"""Lifecycle status of an ``agent.run`` node."""

StepKind = Literal["turn", "planning"]
"""The kind of interpreter step an ``agent.step`` records."""

# ``MessageRole`` is the canonical-turn role vocabulary, defined once in
# :mod:`vella.agent.turn` (the message author role of a canonical ``Message``) and
# re-exported here so an ``agent.message`` node and a canonical ``Message`` share one
# role type — there is no second, divergent role enum.

# The stable type names, frozen by the surface tripwire from M1 onward. The
# agent.* prefix namespaces cognition records; bare names are reserved for the
# substrate-level types (provider/tool/...) that land in later milestones.
RUN_TYPE = "agent.run"
"""Registered type name for the run-envelope node."""

STEP_TYPE = "agent.step"
"""Registered type name for an interpreter-step node."""

MESSAGE_TYPE = "agent.message"
"""Registered type name for a conversation-message node."""

SUMMARY_TYPE = "agent.summary"
"""Registered type name for a compaction-summary node."""


class RunData(VellaModel):
    """Frozen data payload of an ``agent.run`` node.

    Attributes:
        goal: The natural-language objective the run pursues.
        status: The run's lifecycle status.
        loop_policy_ref: Id of the ``loop_policy`` node configuring the run, or
            ``None`` until one is attached (the policy type lands in M5; this is a
            reference, never embedded data).
        provider_ref: Id of the ``provider`` node the run infers through, or
            ``None`` until one is attached (the provider type lands in M2; this is a
            reference, never embedded data).
    """

    goal: str
    status: RunStatus = "pending"
    loop_policy_ref: Optional[UUID] = None
    provider_ref: Optional[UUID] = None


class StepData(VellaModel):
    """Frozen data payload of an ``agent.step`` node.

    Attributes:
        turn_index: The zero-based ordinal of this step within its run.
        kind: Whether the step is a normal model turn or a planning step.
    """

    turn_index: int
    kind: StepKind = "turn"


class MessageData(VellaModel):
    """Frozen data payload of an ``agent.message`` node (canonical content, M2).

    M2 owns the message content contract, so this type now carries the canonical
    block-union ``content`` (the same :data:`~vella.agent.turn.ContentBlock` tuple a
    :class:`~vella.agent.Message` holds): a stored ``agent.message`` node and an
    in-flight canonical message share ONE content shape, so a recorded message
    round-trips back to the exact blocks the model emitted. This supersedes the M1
    ``{role, text}`` shape — the sanctioned monotonic growth at the milestone that
    owns the contract (the surface tripwire is re-baselined for the field-type
    change). ``content`` order is **semantic** and is never sorted.

    Attributes:
        role: Who authored the message.
        content: The ordered tuple of canonical content blocks (order is meaning).
    """

    role: MessageRole
    content: tuple[ContentBlock, ...] = ()


class SummaryData(VellaModel):
    """Frozen data payload of an ``agent.summary`` node.

    Attributes:
        compacted_range: The inclusive ``(start_turn, end_turn)`` index range the
            summary compacts.
        text: The compacted summary text.
    """

    compacted_range: tuple[int, int]
    text: str


def register_agent_types(registry: Registry) -> Registry:
    """Register the M1 cognition node type-specs into ``registry``; return it.

    Binds :class:`RunData`, :class:`StepData`, :class:`MessageData`, and
    :class:`SummaryData` under their stable ``agent.*`` names. Tests pass a fresh
    ``Registry()`` here for isolation rather than touching the global
    ``default_registry`` (pre-mortem #2).

    Args:
        registry: The registry to populate (mutated in place).

    Returns:
        The same ``registry``, now populated, for call-site chaining.
    """
    node_type(RUN_TYPE, registry=registry)(RunData)
    node_type(STEP_TYPE, registry=registry)(StepData)
    node_type(MESSAGE_TYPE, registry=registry)(MessageData)
    node_type(SUMMARY_TYPE, registry=registry)(SummaryData)
    return registry


def agent_registry() -> Registry:
    """Return a fresh ``Registry`` containing exactly the M1 agent types.

    The isolated registry tests inject via ``Node(...)`` /
    ``model_validate(context={"registry": ...})`` so node construction validates
    against the agent's types without depending on (or polluting) the shared
    ``default_registry`` (pre-mortem #2).

    Returns:
        A new ``Registry`` populated with the four M1 cognition type-specs.
    """
    return register_agent_types(Registry())


# Register once, at import, into core's process-wide default registry — the single
# idempotent module-import side effect. This also stamps each class's
# ``__vella_type__`` (so ``Node.from_data`` resolves the type name). Tests still
# inject a fresh ``agent_registry()`` for construction/validation isolation.
register_agent_types(default_registry)
