"""The frozen ``loop_policy`` typed FSM schema (M5, §2.1) — loop-as-data.

This is the configuration surface the M5 interpreter (:mod:`vella.agent.interpreter`)
reads to drive every behaviour: budgets, stop conditions, planning, tool gating,
intent enforcement, sub-agent spawning, and compaction. The interpreter is a pure
FSM *over* this data — there is no behaviour hard-coded into control flow that the
policy does not name (the loop-as-data invariant). A :class:`LoopPolicy` is itself an
ordinary registered core node type (``@node_type("loop_policy")``): a run references
one by id (``RunData.loop_policy_ref``), so the policy is durable, replayable, and
perceivable through the graph like every other piece of cognition.

Shapes (frozen, from the plan §2.1):

* :class:`LoopPolicy` — the top-level FSM config. ``stop_conditions`` is a set-derived
  serialized value, so it is ``sorted()`` (by the field validator) for byte-stable
  bytes; the ``compaction_threshold < token_budget`` cross-field invariant is enforced
  by a ``model_validator`` (the validator the M4 verifier flagged as belonging here —
  the soft watermark must always fire before the hard halt).
* :data:`StopCondition` — the closed stop-condition vocabulary.
* :data:`ToolChoice` — discriminated by ``mode``: :class:`ToolChoiceModel` |
  :class:`ToolChoiceForced` | :class:`ToolChoiceRestricted` (whose ``types`` is
  ``sorted()`` — a set-derived serialized value).
* :data:`SubAgentSpawn` — discriminated by ``mode``: :class:`SubAgentDeny` |
  :class:`SubAgentAllow` (``max_depth``/``max_fanout`` ``>= 1``). The FIELD is frozen
  here; the spawn MECHANISM is M6 — at M5 the interpreter does NOT spawn, and the
  default is :class:`SubAgentDeny`.

The compaction knobs reuse M4's :class:`~vella.agent.CompactionPolicy` verbatim — the
policy does NOT redefine a divergent shape. The interpreter derives the
:class:`~vella.agent.AssemblyPolicy` it hands the assembler FROM this policy (the
compaction block + the hard ``token_budget``), so there is a single source of truth.

All models are frozen :class:`~vella.core.VellaModel` (frozen + ``extra='forbid'``),
compared via ``model_dump(mode="json")`` (never ``==`` — core attaches a private
registry attr). Every set-derived serialized value is ``sorted()``; the discriminated
unions use a ``Literal`` ``mode`` tag + ``Field(discriminator="mode")`` (the same idiom
as core ``Overlay``/``Actuator`` and the canonical content blocks).
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import Field, model_validator
from vella.core import Registry, VellaModel, default_registry, node_type

from .context import AssemblyPolicy, CompactionPolicy

LOOP_POLICY_TYPE = "loop_policy"
"""Registered type name for a loop-policy node (``data`` is a :class:`LoopPolicy`)."""

StopCondition = Literal[
    "no_tool_calls",
    "max_steps",
    "max_tokens",
    "refusal",
    "explicit_stop_node",
]
"""The closed stop-condition vocabulary evaluated at each turn boundary.

* ``no_tool_calls`` — the assistant turn ended with ``stop_reason=="end_turn"`` and
  zero ``tool_use`` blocks (the model has nothing more to do).
* ``max_steps`` — the mirror of ``step_budget`` firing (recorded as the halt reason).
* ``max_tokens`` — the mirror of ``token_budget`` firing (the HARD halt).
* ``refusal`` — the assistant turn ended with ``stop_reason=="refusal"``.
* ``explicit_stop_node`` — a goal-completion sentinel tool/marker was observed.

The :class:`LoopPolicy.stop_conditions` tuple is ``sorted()`` before serialization
(a set-derived serialized value), and the interpreter evaluates the conditions in
that sorted order so the FIRST firing one is the deterministically-recorded reason.
"""

# The sentinel tool name whose invocation fires ``explicit_stop_node``. A goal-
# completion marker the model emits as a tool_use; observing it ends the run. Kept a
# module constant (not a knob) so the FSM transition is fixed and replayable.
EXPLICIT_STOP_TOOL = "stop"
"""The tool name that, when called, fires the ``explicit_stop_node`` stop condition."""


class ToolChoiceModel(VellaModel):
    """Tool gating: the model decides freely whether to call a tool.

    Attributes:
        mode: The discriminator literal (always ``"model"``).
    """

    mode: Literal["model"] = "model"


class ToolChoiceForced(VellaModel):
    """Tool gating: the model MUST emit at least one tool call this turn.

    A turn that emits no ``tool_use`` block under this mode is a policy violation
    the interpreter rejects (the provider param requires a tool; the in-gate
    :class:`~vella.agent.MockProvider` honours it).

    Attributes:
        mode: The discriminator literal (always ``"forced"``).
    """

    mode: Literal["forced"] = "forced"


class ToolChoiceRestricted(VellaModel):
    """Tool gating: only tools whose ``declaration.name`` is in ``types`` are offered.

    The interpreter FILTERS the discovered tool-nodes to those whose declaration name
    is in ``types`` BEFORE assembling the request's ``tools``, and REJECTS a
    ``tool_use`` naming a tool outside the set.

    Attributes:
        mode: The discriminator literal (always ``"restricted"``).
        types: The allowed tool names (a set of names — order is not semantic — so it
            is ``sorted()`` before serialization for byte-stable bytes).
    """

    mode: Literal["restricted"] = "restricted"
    types: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _sort_types(self) -> "ToolChoiceRestricted":
        """Sort ``types`` — a set-derived serialized value — for deterministic bytes."""
        ordered = tuple(sorted(self.types))
        if ordered != self.types:
            object.__setattr__(self, "types", ordered)
        return self


ToolChoice = Annotated[
    Union[ToolChoiceModel, ToolChoiceForced, ToolChoiceRestricted],
    Field(discriminator="mode"),
]
"""The tool-gating choice — discriminated by ``mode``.

The same idiom as core ``Overlay``/``Actuator``: a closed union tagged by a
``Literal`` ``mode`` field, resolved by pydantic via ``Field(discriminator="mode")``.
"""


class SubAgentDeny(VellaModel):
    """Sub-agent gating: spawning a child run is a policy violation (the default).

    Attributes:
        mode: The discriminator literal (always ``"deny"``).
    """

    mode: Literal["deny"] = "deny"


class SubAgentAllow(VellaModel):
    """Sub-agent gating: spawning is allowed, bounded by depth and fanout (M6).

    The FIELD is frozen here at M5; the spawn MECHANISM (the pre-spawn graph gate +
    child run creation + result propagation) lands in M6. At M5 the interpreter never
    spawns, so these bounds are carried but not yet enforced.

    Attributes:
        mode: The discriminator literal (always ``"allow"``).
        max_depth: Maximum ``PART_OF`` chain depth from a child up to the root run
            (``>= 1``).
        max_fanout: Maximum direct children a single run may spawn (``>= 1``).
    """

    mode: Literal["allow"] = "allow"
    max_depth: int = Field(default=1, ge=1)
    max_fanout: int = Field(default=1, ge=1)


SubAgentSpawn = Annotated[
    Union[SubAgentDeny, SubAgentAllow],
    Field(discriminator="mode"),
]
"""The sub-agent spawn policy — discriminated by ``mode`` (deny | allow)."""


class LoopPolicy(VellaModel):
    """The frozen ``loop_policy`` FSM configuration (``@node_type("loop_policy")``).

    The interpreter is a pure FSM over this data; every behaviour it exhibits is named
    here (loop-as-data). A run references one of these by id
    (``RunData.loop_policy_ref``), so the policy is durable/replayable like every other
    node.

    The cross-field invariant ``compaction_threshold < token_budget`` (when BOTH are
    set) is enforced by :meth:`_check_compaction_below_budget`: the soft watermark must
    always fire strictly before the hard halt, so they can never breach on the same
    turn (§2.1). ``stop_conditions`` is ``sorted()`` by
    :meth:`_sort_stop_conditions` — a set-derived serialized value.

    Attributes:
        step_budget: Maximum ``agent.step`` nodes before a forced ``max_steps`` halt,
            or ``None`` for unbounded (within the token budget and the driver's
            ``max_steps`` backstop). Enforced at the turn boundary BEFORE requesting
            the next provider turn.
        token_budget: The HARD cumulative-token halt (``input + output + reasoning``;
            cache tokens are NOT counted). When cumulative usage reaches it the run
            halts ``max_tokens`` (terminal, NO compaction). ``None`` = unbounded.
        stop_conditions: The stop-condition vocabulary evaluated each turn boundary
            (``sorted()`` for deterministic first-firing-wins ordering).
        planning: The planning FSM mode — ``"off"`` (straight loop), ``"single"`` (one
            planning turn first, then loop), or ``"replan_on_failure"`` (a non-retryable
            tool error transitions back to a planning turn before continuing).
        tool_choice: The discriminated tool-gating choice.
        require_tool_intent: When ``True``, a ``tool_use`` block missing a non-empty
            ``<= 1``-sentence ``intent`` is a policy violation.
        sub_agent_spawn: The discriminated sub-agent spawn policy (default deny; the
            mechanism is M6).
        compaction: The M4 :class:`~vella.agent.CompactionPolicy` (soft watermark / pin
            / recall) — reused verbatim, never redefined.
    """

    step_budget: Optional[int] = Field(default=None, ge=1)
    token_budget: Optional[int] = Field(default=None, ge=1)
    stop_conditions: tuple[StopCondition, ...] = ()
    planning: Literal["off", "single", "replan_on_failure"] = "off"
    tool_choice: ToolChoice = Field(default_factory=ToolChoiceModel)
    require_tool_intent: bool = True
    sub_agent_spawn: SubAgentSpawn = Field(default_factory=SubAgentDeny)
    compaction: CompactionPolicy = Field(default_factory=CompactionPolicy)

    @model_validator(mode="after")
    def _sort_stop_conditions(self) -> "LoopPolicy":
        """Sort ``stop_conditions`` — a set-derived serialized value — deterministically."""
        ordered = tuple(sorted(self.stop_conditions))
        if ordered != self.stop_conditions:
            object.__setattr__(self, "stop_conditions", ordered)
        return self

    @model_validator(mode="after")
    def _check_compaction_below_budget(self) -> "LoopPolicy":
        """Assert ``compaction_threshold < token_budget`` whenever both are set.

        The soft compaction watermark must fire strictly before the hard token halt,
        so the two never breach on the same turn (§2.1). A non-strict or inverted
        configuration is a frozen-policy error caught at construction.

        Raises:
            ValueError: If both are set and ``compaction_threshold >= token_budget``.
        """
        threshold = self.compaction.compaction_threshold
        budget = self.token_budget
        if threshold is not None and budget is not None and threshold >= budget:
            raise ValueError(
                f"compaction_threshold ({threshold}) must be < token_budget "
                f"({budget}) so the soft watermark fires before the hard halt."
            )
        return self

    def assembly_policy(self) -> AssemblyPolicy:
        """Derive the :class:`~vella.agent.AssemblyPolicy` the assembler reads.

        The interpreter hands the assembler exactly this derived view — the M4
        compaction knobs plus the HARD ``token_budget`` (which the assembler needs to
        honour the terminal "never compact past the hard halt" rule, §2.1) — so the
        compaction contract has a single source of truth in this policy rather than a
        divergent duplicate.

        Returns:
            The frozen assembly policy (``compaction`` + ``token_budget``).
        """
        return AssemblyPolicy(compaction=self.compaction, token_budget=self.token_budget)


def register_policy_types(registry: Registry) -> Registry:
    """Register the :class:`LoopPolicy` node type-spec into ``registry``; return it.

    Binds :class:`LoopPolicy` under :data:`LOOP_POLICY_TYPE`. Tests pass a fresh
    ``Registry()`` here for isolation rather than touching the global
    ``default_registry`` (pre-mortem #2).

    Args:
        registry: The registry to populate (mutated in place).

    Returns:
        The same ``registry``, now populated, for call-site chaining.
    """
    node_type(LOOP_POLICY_TYPE, registry=registry)(LoopPolicy)
    return registry


# Register once, at import, into core's process-wide default registry — the single
# idempotent module-import side effect, mirroring :mod:`vella.agent.types` and
# :mod:`vella.agent.tool`. This also stamps ``LoopPolicy.__vella_type__`` so
# ``Node.from_data`` resolves the type name. Tests still inject a fresh registry.
register_policy_types(default_registry)
