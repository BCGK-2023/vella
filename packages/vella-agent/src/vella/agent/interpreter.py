"""The ``loop_policy`` FSM interpreter — the first end-to-end self-hosting loop (M5).

This is the cognition core's control surface: a pure finite-state machine *over* a
frozen :class:`~vella.agent.LoopPolicy` (loop-as-data). It owns no behaviour the
policy does not name; it acts ONLY through the runtime's published verbs (every
run/step/message/tool_call node is materialized via :mod:`vella.agent._writeback`,
the reasoning trace via ``emit_telemetry``) and perceives ONLY through the graph
projection and ``runtime.get`` authority. There is no privileged path and no new
``Node`` subclass.

The turn loop (§2.1, implemented EXACTLY):

    open a new agent.step  ->  assemble (ContextAssembler)
      ->  provider.turn(request)
      ->  record agent.message + an observe_only reasoning trace (no version bump)
      ->  if the assistant turn has tool_use blocks:
            for each: invoke (ToolInvoker) -> resolve hint
                      -> write agent.tool_call + a tool_result Message (via verbs)
      ->  evaluate stop conditions / budgets at the turn boundary
      ->  next turn or halt

Budget enforcement (the off-by-one matters, mutation (a)): ``step_budget`` is checked
**at the turn boundary, BEFORE requesting the next provider turn** — so N steps
produce AT MOST N ``agent.step`` nodes and the halt reason is ``max_steps``. The HARD
``token_budget`` and the SOFT ``compaction_threshold`` are enforced in the §2.1 order
at each turn boundary: (i) record usage; (ii) cumulative ``>= token_budget`` ⇒ HALT
``max_tokens`` (terminal, NO compaction); (iii) else cumulative ``>=
compaction_threshold`` ⇒ the assembler compacts on the next assemble and the loop
CONTINUES.

The non-retryable predicate (pinned): a :class:`~vella.agent.ToolResult` is
non-retryable *at the loop level* exactly when ``result.is_error`` is ``True`` AFTER
:meth:`ToolInvoker.invoke` returns. The invoker owns retries (R4) — it surfaces a
final ``is_error`` result only once its own capped retries are exhausted — so the loop
NEVER re-invokes; it only branches the FSM (``replan_on_failure`` ⇒ a planning turn)
on that final error. ``replan_on_failure`` is the only mode that reads this predicate.

Bounded driver (no ``pytest-asyncio``): ``max_steps`` is the hard backstop so a
fully-unbounded policy (both budgets ``None``) still terminates — the loop body runs
at most ``max_steps`` times regardless of policy. Backoff/clock waits go through the
injected :class:`~vella.agent.Clock` (the invoker's concern); the interpreter itself
never sleeps on a real timer.

Replay/resume (TRAP-1): a run's progress is DURABLE — its ``agent.step`` nodes are the
authority for how many turns ran (read from the graph + ``runtime.get``, never from an
in-memory counter), and its cumulative token usage is folded from the run's
``observe_only`` reasoning-trace telemetry (the trace's designed home — telemetry
payload, not entity state). An interrupted run resumes by folding those recorded steps
and continuing from the last recorded turn index, reaching the SAME terminal
projection as an uninterrupted run.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from uuid import UUID

from vella.core import EdgeTypes, Node, UnresolvedRef
from vella.graph import GraphProjection
from vella.runtime import Runtime

from ._discovery import discover_tools
from ._hints import resolve_hint
from ._subagent import (
    SPAWN_TOOL,
    child_terminal_messages,
    gate_allows_spawn,
    spawn_child,
)
from ._writeback import (
    append_message,
    append_step,
    append_tool_call,
    emit_reasoning_trace,
)
from .clock import Clock
from .context import AssembledContext, ContextAssembler
from .invoker import ToolInvoker
from .policy import EXPLICIT_STOP_TOOL, LoopPolicy, SubAgentAllow
from .provider import ModelProvider, ToolSchema, TurnParams, TurnRequest
from .tool import ToolCallData, ToolData, ToolResult
from .turn import AssistantTurn, Message, ToolResultBlock, ToolUseBlock
from .types import MessageData, RunData, StepData


class RunResult:
    """The terminal state of an interpreter :func:`run`.

    A small frozen-by-convention value object (not a node — it is the function's
    return, the durable record IS the run/step nodes). Carries the run id, the
    terminal status, the halt reason, the number of ``agent.step`` nodes the run
    produced (including any from a prior, resumed segment), and the cumulative token
    total the run accounted.

    Attributes:
        run_id: The ``agent.run`` node id.
        status: The terminal :data:`~vella.agent.RunStatus` (``"succeeded"`` /
            ``"failed"``).
        halt_reason: The :data:`~vella.agent.StopCondition`-shaped reason the loop
            stopped, or ``None`` if it ran to a natural ``no_tool_calls`` end without a
            configured stop condition recording one.
        steps: The total ``agent.step`` node count for the run (durable authority).
        tokens: The cumulative ``input + output + reasoning`` token total.
    """

    __slots__ = ("run_id", "status", "halt_reason", "steps", "tokens")

    def __init__(
        self,
        run_id: UUID,
        *,
        status: str,
        halt_reason: Optional[str],
        steps: int,
        tokens: int,
    ) -> None:
        """Build the terminal run result (see the class docstring for fields)."""
        self.run_id = run_id
        self.status = status
        self.halt_reason = halt_reason
        self.steps = steps
        self.tokens = tokens


def _turn_tokens(turn: AssistantTurn) -> int:
    """The tokens a turn counts against the budget: ``input + output + reasoning``.

    Cache read/write tokens are deliberately NOT counted (§2.1) — they are a caching
    artefact, not budgeted spend.

    Args:
        turn: The assistant turn whose usage to total.

    Returns:
        The budgeted token total for the turn.
    """
    u = turn.usage
    return u.input_tokens + u.output_tokens + u.reasoning_tokens


def _tool_uses(turn: AssistantTurn) -> list[ToolUseBlock]:
    """The ``tool_use`` blocks of an assistant turn, in content (semantic) order."""
    return [b for b in turn.content if isinstance(b, ToolUseBlock)]


async def _resumed_progress(
    runtime: Runtime, run_node: UUID, *, tenant_id: str
) -> tuple[int, int]:
    """Fold a run's DURABLE progress: (step count, cumulative tokens) — TRAP-1.

    Step count is the authority for how many turns already ran — read from the graph
    (the run's ``PART_OF`` ``"in"`` ``agent.step`` neighbours) with bodies confirmed
    via ``runtime.get`` (entity authority), never an in-memory counter. Cumulative
    tokens are folded from the run's ``observe_only`` reasoning-trace telemetry entries
    (the trace's designed home — free-form telemetry payload, NOT entity state), each
    of which carries its turn's budgeted token total under ``"tokens"``.

    An uninterrupted run enters with zero steps and zero tokens; a resumed run enters
    with exactly what it durably recorded, so it continues from the last step index and
    the correct cumulative usage.

    Args:
        runtime: The runtime to perceive through (graph fold + get + history).
        run_node: The run whose progress to fold.
        tenant_id: The tenant the run belongs to.

    Returns:
        ``(step_count, cumulative_tokens)`` reconstructed from the durable record.
    """
    view = await GraphProjection().fold(runtime, tenant_id)
    neighbours = await view.neighbors(
        run_node, edge_type=EdgeTypes.PART_OF, direction="in"
    )
    step_count = 0
    for neighbour in neighbours:
        node = await runtime.get(tenant_id, neighbour.node_id)
        if isinstance(node, Node) and isinstance(node.data, StepData):
            step_count += 1
    # Cumulative tokens fold from the durable observe_only reasoning trace. Telemetry
    # payload is the trace's home (not entity state), so reading it here is the
    # sanctioned resume path, not a TRAP-1 payload-reconstruction of an entity.
    tokens = 0
    for entry in await runtime.history(tenant_id, run_node):
        if entry.transition == "observe_only":
            value = entry.payload.get("tokens")
            if isinstance(value, int):
                tokens += value
    return step_count, tokens


async def run(
    runtime: Runtime,
    run_node: UUID,
    *,
    tenant_id: str,
    provider: ModelProvider,
    invoker: ToolInvoker,
    assembler: ContextAssembler,
    clock: Clock,
    max_steps: int,
) -> RunResult:
    """Drive ``run_node``'s configurable FSM loop to a terminal state (§2.1).

    Reads the run's :class:`~vella.agent.LoopPolicy` (via ``RunData.loop_policy_ref``,
    or an all-default policy when none is attached), folds any DURABLE prior progress
    (so an interrupted run resumes from its last recorded step — replay/resume), then
    runs the turn loop EXACTLY as the module docstring specifies, bounded by
    ``max_steps`` as the hard backstop. Every write goes through runtime verbs; the
    reasoning trace is ``observe_only`` (no version bump). The ``clock`` is the seam the
    invoker's backoff waits sleep on (the interpreter never sleeps on a real timer).

    Args:
        runtime: The runtime to act through and perceive (verbs + ``get`` authority).
        run_node: The ``agent.run`` node id to drive.
        tenant_id: The tenant the run (and every node it references) belongs to — the
            runtime keys ``get`` / ``edit`` / ``observe`` by tenant.
        provider: The :class:`~vella.agent.ModelProvider` inference seam.
        invoker: The :class:`~vella.agent.ToolInvoker` behaviour seam (owns retries).
        assembler: The :class:`~vella.agent.ContextAssembler` perception seam.
        clock: The :class:`~vella.agent.Clock` backoff waits sleep on (injected;
            deterministic under a :class:`~vella.agent.ManualClock`).
        max_steps: The hard driver backstop — the loop body runs at most this many
            times regardless of policy, so a fully-unbounded policy still terminates.

    Returns:
        The terminal :class:`RunResult` (status + halt reason + durable step/token
        totals).
    """
    run_obj = await runtime.get(tenant_id, run_node)
    # The run node body is authority (TRAP-1: never reconstruct from a payload).
    assert isinstance(run_obj, Node)
    run_data = run_obj.data
    assert isinstance(run_data, RunData)
    run_version = run_obj.version

    policy = await _load_policy(runtime, tenant_id, run_data)
    assembly_policy = policy.assembly_policy()

    # Discover the run's toolset once (graph + HAS_TOOL) — a tool-id tuple, resolved to
    # nodes on demand. Restricted tool_choice filters this set BEFORE the request.
    tool_ids = await discover_tools(runtime, run_node, tenant_id=tenant_id)
    tool_nodes: dict[str, Node[Any, Any]] = {}
    for tid in tool_ids:
        node = await runtime.get(tenant_id, tid)
        if isinstance(node, Node) and isinstance(node.data, ToolData):
            tool_nodes[node.data.declaration.name] = node

    # --- replay/resume: fold the durable record, never an in-memory counter (TRAP-1) ---
    step_count, cumulative_tokens = await _resumed_progress(
        runtime, run_node, tenant_id=tenant_id
    )

    # Mark the run running (idempotent: only edit when not already running). The run's
    # status is durable entity state nested in the node's ``data`` payload, so the edit
    # replaces the whole ``data`` with an evolved :class:`~vella.agent.RunData` (a
    # version bump) — unlike the observe_only reasoning trace, which never bumps.
    if run_data.status != "running":
        entry = await runtime.edit(
            tenant_id,
            run_node,
            expected_version=run_version,
            data=run_data.model_copy(update={"status": "running"}),
        )
        run_version = entry.version
        run_data = run_data.model_copy(update={"status": "running"})

    # planning FSM: distinct transition tables (mutation (b) — never collapse single/off).
    # `single` schedules ONE leading planning turn (its own step, counts against
    # step_budget); `replan_on_failure` schedules a planning turn after a non-retryable
    # tool error; `off` never plans. A resumed run that already produced steps does NOT
    # re-run the leading plan (the durable step count proves it ran).
    plan_next = policy.planning == "single" and step_count == 0

    halt_reason: Optional[str] = None
    status = "succeeded"

    # The hard backstop: the loop body runs at most `max_steps` times THIS segment.
    for _ in range(max_steps):
        # --- step_budget: enforced at the turn boundary BEFORE the next turn (mutation
        # (a) — checking after would allow an (N+1)th step). N steps => <= N step nodes.
        if policy.step_budget is not None and step_count >= policy.step_budget:
            halt_reason = "max_steps"
            break

        is_planning_turn = plan_next
        plan_next = False

        # --- open a NEW agent.step (planning turns ARE their own step; §2.1) ---
        step_kind = "planning" if is_planning_turn else "turn"
        step = await append_step(
            runtime,
            run_node,
            StepData(turn_index=step_count, kind=step_kind),
            name=f"step-{step_count}",
            tenant_id=tenant_id,
        )
        step_count += 1

        # --- assemble: perceive the run into canonical messages + cache metadata ---
        ctx: AssembledContext = await assembler.assemble(
            runtime,
            run_node,
            tenant_id=tenant_id,
            provider_node=_provider_node(run_data),
            policy=assembly_policy,
        )

        # --- sub-agent result propagation (§2.2): the parent's NEXT turn perceives
        # any TERMINATED child's final output by reading the GRAPH (child PART_OF
        # parent, explicit direction) — never an in-memory handoff (mutation (d)), so
        # it survives replay/resume. Appended AFTER the assembled tail so the model
        # sees the child results as the most recent context.
        propagated = await child_terminal_messages(
            runtime, run_node, tenant_id=tenant_id
        )
        request_messages: tuple[Message, ...] = ctx.messages + propagated

        # --- tool gating: filter offered tools per tool_choice BEFORE the request ---
        offered, mode = _offered_tools(policy, tool_nodes, is_planning_turn)
        request = TurnRequest(
            messages=request_messages,
            tools=offered,
            params=TurnParams(
                tool_choice=mode,
                cache=ctx.cacheable_prefix_len > 0,
            ),
        )

        # --- provider.turn -> canonical assistant turn the FSM pattern-matches on ---
        turn = await provider.turn(request)

        # --- record agent.message (durable) + observe_only reasoning trace (no bump) ---
        await append_message(
            runtime,
            run_node,
            MessageData(role="assistant", content=turn.content),
            name=f"assistant-{step.data.turn_index}",
            tenant_id=tenant_id,
        )
        turn_tokens = _turn_tokens(turn)
        await emit_reasoning_trace(
            runtime,
            run_node,
            tenant_id=tenant_id,
            payload={
                "turn_index": step.data.turn_index,
                "kind": step_kind,
                "stop_reason": turn.stop_reason,
                "tokens": turn_tokens,
            },
        )

        uses = _tool_uses(turn)

        # --- forced tool_choice: a turn with no tool_use is a policy violation ---
        if mode == "forced" and not uses:
            status = "failed"
            halt_reason = "refusal"
            break

        # --- a planning turn invokes NO tools (§2.1): record it, then loop ---
        if is_planning_turn:
            # Token-boundary accounting still applies to a planning turn.
            cumulative_tokens += turn_tokens
            stop = _budget_stop(policy, cumulative_tokens)
            if stop is not None:
                halt_reason = stop
                break
            continue

        # --- refusal stop condition (sorted-first-wins is handled below) ---
        # --- invoke each tool_use, resolve hints, write durable records ---
        saw_explicit_stop = False
        saw_nonretryable = False
        for use in uses:
            # restricted tool_choice: REJECT a tool_use naming an out-of-set tool
            # (mutation (c) — passing it through would breach the restriction).
            if not _tool_allowed(policy, use.name):
                status = "failed"
                halt_reason = "refusal"
                break

            # require_tool_intent: an intent-less block is a policy violation
            # (mutation (d) — accepting it would erase the UX-legibility contract).
            if policy.require_tool_intent and not _valid_intent(use.intent):
                status = "failed"
                halt_reason = "refusal"
                break

            # --- sub-agent spawn: the reserved SPAWN_TOOL routes through the M6
            # pre-spawn graph gate, NOT the ToolInvoker. `deny` makes a spawn request
            # a policy violation; `allow` runs it through the bounded gate (depth +
            # fanout from the durable graph BEFORE any child node exists). ---
            if use.name == SPAWN_TOOL:
                spawn_halt = await _handle_spawn(
                    runtime,
                    run_node,
                    use,
                    step,
                    policy,
                    tenant_id=tenant_id,
                    provider=provider,
                    invoker=invoker,
                    assembler=assembler,
                    clock=clock,
                    max_steps=max_steps,
                    provider_ref=run_data.provider_ref,
                )
                if spawn_halt is not None:
                    status = "failed"
                    halt_reason = spawn_halt
                    break
                continue

            tool_node = tool_nodes.get(use.name)
            if tool_node is None:
                status = "failed"
                halt_reason = "refusal"
                break

            result: ToolResult = await invoker.invoke(tool_node, use.input)
            hint = resolve_hint(tool_node.data.hints, result)

            await append_tool_call(
                runtime,
                step.id,
                ToolCallData(
                    tool_ref=tool_node.id,
                    args=use.input,
                    intent=use.intent,
                    result=result.content,
                    error_kind=result.error_kind,
                    hint=hint,
                ),
                name=f"call-{step.data.turn_index}-{use.id}",
                tenant_id=tenant_id,
            )
            # Feed the tool result back as a `tool`-role message for the next turn.
            await append_message(
                runtime,
                run_node,
                MessageData(
                    role="tool",
                    content=(
                        ToolResultBlock(
                            tool_use_id=use.id,
                            content=result.content,
                            is_error=result.is_error,
                            hint=hint,
                        ),
                    ),
                ),
                name=f"tool-{step.data.turn_index}-{use.id}",
                tenant_id=tenant_id,
            )
            if use.name == EXPLICIT_STOP_TOOL:
                saw_explicit_stop = True
            # Pinned non-retryable predicate: is_error AFTER invoke (the invoker's own
            # capped retries are exhausted — the loop never re-invokes).
            if result.is_error:
                saw_nonretryable = True

        if halt_reason is not None:
            break

        # --- turn-boundary accounting: (i) record usage ---
        cumulative_tokens += turn_tokens

        # --- (ii) HARD token_budget halt (terminal, NO compaction) then SOFT watermark ---
        budget_stop = _budget_stop(policy, cumulative_tokens)
        if budget_stop is not None:
            halt_reason = budget_stop
            break

        # --- stop conditions evaluated in SORTED order; first firing wins ---
        stop = _evaluate_stops(policy, turn, uses, saw_explicit_stop)
        if stop is not None:
            halt_reason = stop
            break

        # --- planning: replan_on_failure transitions to a planning turn next ---
        if policy.planning == "replan_on_failure" and saw_nonretryable:
            plan_next = True
            continue
    else:
        # The for-loop exhausted max_steps without an explicit break — the hard
        # backstop fired. Record it as max_steps so an unbounded policy still halts
        # deterministically rather than silently running forever.
        if halt_reason is None:
            halt_reason = "max_steps"

    # --- finalize the durable run status (an edit — version bump, not telemetry) ---
    terminal_status = "failed" if status == "failed" else "succeeded"
    await runtime.edit(
        tenant_id,
        run_node,
        expected_version=run_version,
        data=run_data.model_copy(update={"status": terminal_status}),
    )

    return RunResult(
        run_node,
        status=terminal_status,
        halt_reason=halt_reason,
        steps=step_count,
        tokens=cumulative_tokens,
    )


async def _load_policy(
    runtime: Runtime, tenant_id: str, run_data: RunData
) -> LoopPolicy:
    """Load the run's :class:`~vella.agent.LoopPolicy`, or an all-default one.

    The policy is a durable node referenced by ``RunData.loop_policy_ref``; it is read
    from ``runtime.get`` authority (TRAP-1). A run with no attached policy gets an
    all-default :class:`LoopPolicy` (the safe baseline: unbounded budgets within the
    driver backstop, no stop conditions, ``planning="off"``, ``tool_choice=model``,
    intent required, spawn denied).

    Args:
        runtime: The runtime to read the policy node through.
        tenant_id: The tenant the policy node belongs to.
        run_data: The run's frozen payload (its ``loop_policy_ref`` is the policy id).

    Returns:
        The run's loop policy, or a default one when none is attached.
    """
    if run_data.loop_policy_ref is None:
        return LoopPolicy()
    node = await runtime.get(tenant_id, run_data.loop_policy_ref)
    if isinstance(node, Node) and isinstance(node.data, LoopPolicy):
        return node.data
    return LoopPolicy()


def _provider_node(run_data: RunData) -> UUID:
    """The run's provider node id for the assembler (a zero UUID when none attached).

    The assembler treats a missing/non-provider node as non-cache-capable (its safe
    default), so an unattached provider degrades to aggressive compaction rather than
    failing — the M5 reference loop runs with a MockProvider whose cache-capability is
    declared on an attached ``provider`` node when caching is under test.

    Args:
        run_data: The run's frozen payload.

    Returns:
        The attached provider node id, or a nil UUID sentinel.
    """
    return run_data.provider_ref if run_data.provider_ref is not None else UUID(int=0)


def _offered_tools(
    policy: LoopPolicy,
    tool_nodes: dict[str, Node[Any, Any]],
    is_planning_turn: bool,
) -> tuple[tuple[ToolSchema, ...], Literal["model", "forced", "none"]]:
    """The tool schemas offered this turn + the provider ``tool_choice`` mode.

    A planning turn offers NO tools (it produces a plan, never a call; §2.1). Otherwise
    the offered set is the discovered tool declarations, FILTERED by a
    ``restricted(types)`` choice to declarations whose name is in ``types`` (mutation
    (c) — the filter is BEFORE the request, not after the turn). The provider mode is
    ``"forced"`` for a forced choice, else ``"model"`` (restricted still lets the model
    choose among the filtered set, so its provider mode is ``"model"``).

    Args:
        policy: The loop policy (its ``tool_choice`` selects the filter + mode).
        tool_nodes: The discovered tool nodes keyed by declaration name.
        is_planning_turn: Whether this is a leading/replan planning turn (no tools).

    Returns:
        ``(offered tool schemas sorted by name, provider tool_choice mode)``.
    """
    if is_planning_turn:
        return (), "none"
    choice = policy.tool_choice
    names = sorted(tool_nodes)
    if choice.mode == "restricted":
        allowed = set(choice.types)
        names = [n for n in names if n in allowed]
    offered = tuple(tool_nodes[n].data.declaration for n in names)
    if choice.mode == "forced":
        return offered, "forced"
    return offered, "model"


def _tool_allowed(policy: LoopPolicy, name: str) -> bool:
    """Whether a ``tool_use`` naming ``name`` is permitted under the tool_choice.

    Only ``restricted(types)`` constrains which tools may be CALLED: a call naming a
    tool outside ``types`` is rejected (mutation (c)). ``model``/``forced`` place no
    name restriction (forced only requires that *some* tool is called).

    Args:
        policy: The loop policy.
        name: The tool name the model's ``tool_use`` block named.

    Returns:
        ``True`` if the call is permitted by the tool-choice policy.
    """
    choice = policy.tool_choice
    if choice.mode == "restricted":
        return name in set(choice.types)
    return True


def _valid_intent(intent: str) -> bool:
    """Whether a ``tool_use`` intent satisfies ``require_tool_intent`` (mutation (d)).

    A valid intent is non-empty (after stripping whitespace) and at most one sentence
    — a single natural-language clause of UX narration. "At most one sentence" is
    enforced as "no more than one sentence-terminator that is followed by more text",
    so a trailing period is fine but two sentences are not.

    Args:
        intent: The block's ``intent`` string.

    Returns:
        ``True`` iff the intent is present and is at most one sentence.
    """
    text = intent.strip()
    if not text:
        return False
    # Count sentence terminators that are followed by further non-space text — a
    # second sentence. A single trailing terminator is allowed.
    body = text.rstrip(".!?")
    return not any(sep in body for sep in (". ", "! ", "? "))


def _budget_stop(policy: LoopPolicy, cumulative_tokens: int) -> Optional[str]:
    """The HARD token-budget halt reason, if cumulative usage reached the budget.

    Step (ii) of the §2.1 turn-boundary order: the HARD ``token_budget`` always wins
    and is terminal — there is NO compaction on the halting turn. (The SOFT
    ``compaction_threshold`` is the assembler's concern on the NEXT assemble; the loop
    continues past it, so it is not a halt.)

    Args:
        policy: The loop policy.
        cumulative_tokens: The run's cumulative budgeted token total.

    Returns:
        ``"max_tokens"`` if the hard budget is reached, else ``None``.
    """
    if policy.token_budget is not None and cumulative_tokens >= policy.token_budget:
        return "max_tokens"
    return None


async def _handle_spawn(
    runtime: Runtime,
    parent: UUID,
    use: ToolUseBlock,
    step: Node[Any, Any],
    policy: LoopPolicy,
    *,
    tenant_id: str,
    provider: ModelProvider,
    invoker: ToolInvoker,
    assembler: ContextAssembler,
    clock: Clock,
    max_steps: int,
    provider_ref: Optional[UUID],
) -> Optional[str]:
    """Handle a ``SPAWN_TOOL`` request through the M6 bounded pre-spawn gate (§2.2).

    The control flow, exactly:

    * ``deny`` ⇒ a spawn request is a POLICY VIOLATION — return ``"refusal"`` (the
      caller halts the parent ``failed``).
    * ``allow`` ⇒ evaluate :func:`~vella.agent._subagent.gate_allows_spawn` against the
      DURABLE graph BEFORE any child node exists (TRAP-1). A breach of either bound is
      a BOUNDED REFUSAL: record a ``tool_result`` (``is_error=True``) so the model
      learns the spawn was refused, and CONTINUE (return ``None``) — the bound held on
      the durable record, the parent is not failed. Within bounds: create the child via
      verbs (its OWN budget — no pooling), run it to terminal via the recursive
      :func:`run` (the depth bound makes the recursion finite), then record a
      ``tool_result`` referencing the child. The child's terminal output propagates
      into the parent's NEXT turn via the graph (handled at assembly).

    Args:
        runtime: The runtime to gate against + write the child/link through.
        parent: The spawning (parent) run id.
        use: The ``SPAWN_TOOL`` ``tool_use`` block (its ``input`` carries the child's
            ``goal`` and OPTIONAL own ``step_budget`` / ``token_budget``).
        step: The parent's current ``agent.step`` node (the tool-call record's parent).
        policy: The parent's loop policy (its ``sub_agent_spawn`` gates the request).
        tenant_id: The tenant the run-tree belongs to.
        provider: The inference seam the child run reuses (recursive).
        invoker: The behaviour seam the child run reuses.
        assembler: The perception seam the child run reuses.
        clock: The backoff clock the child run reuses.
        max_steps: The driver backstop the child run reuses (its own bound).
        provider_ref: The child's ``provider`` node id (inherited from the parent).

    Returns:
        ``"refusal"`` when the spawn is a policy violation (``deny``) — the caller
        halts the parent; ``None`` otherwise (within bounds OR a bounded refusal —
        the parent continues).
    """
    spawn = policy.sub_agent_spawn
    if not isinstance(spawn, SubAgentAllow):
        # deny: a spawn request is a policy violation.
        return "refusal"

    goal = use.input.get("goal")
    goal_text = goal if isinstance(goal, str) and goal else "subagent"

    allowed = await gate_allows_spawn(
        runtime, parent, tenant_id=tenant_id, allow=spawn
    )
    if not allowed:
        # BOUNDED REFUSAL: NO child node/edge was created (the gate ran on the graph
        # before any write). Record the refusal as a tool_result and CONTINUE — the
        # bound held on the durable record (mutation (b)/(c) would let a phantom child
        # land here). The parent is not failed; runaway spawning is simply prevented.
        await append_tool_call(
            runtime,
            step.id,
            ToolCallData(
                tool_ref=parent,
                args=use.input,
                intent=use.intent,
                result="spawn refused: depth/fanout bound reached",
                error_kind="SubAgentBoundExceeded",
                hint=None,
            ),
            name=f"spawn-refused-{step.data.turn_index}-{use.id}",
            tenant_id=tenant_id,
        )
        await append_message(
            runtime,
            parent,
            MessageData(
                role="tool",
                content=(
                    ToolResultBlock(
                        tool_use_id=use.id,
                        content="spawn refused: depth/fanout bound reached",
                        is_error=True,
                    ),
                ),
            ),
            name=f"spawn-refused-msg-{step.data.turn_index}-{use.id}",
            tenant_id=tenant_id,
        )
        return None

    # --- within bounds: build the child's OWN loop_policy (its OWN budget — NO pooling
    # of a shared mutable counter, mutation (f)). The child inherits the SAME bounded
    # `allow` so it may itself spawn grandchildren, still gated by the same depth/fanout
    # against the durable graph — the depth bound is what makes the recursion finite. ---
    child_step_budget = _opt_int(use.input.get("step_budget"))
    child_token_budget = _opt_int(use.input.get("token_budget"))
    child_policy = LoopPolicy(
        step_budget=child_step_budget,
        token_budget=child_token_budget,
        sub_agent_spawn=spawn,
    )
    child_policy_node = Node.from_data(
        child_policy,
        name="subagent-policy",
        created_by=UnresolvedRef(identifier="vella:agent"),
        tenant_id=tenant_id,
    )
    await runtime.create(child_policy_node)

    child_id = await spawn_child(
        runtime,
        parent,
        tenant_id=tenant_id,
        goal=goal_text,
        child_policy_ref=child_policy_node.id,
        provider_ref=provider_ref,
    )
    assert child_id is not None

    # --- run the child to terminal via the recursive M5 driver. The child is bounded
    # by its OWN budget + the same max_steps backstop + ManualClock. The depth gate
    # (checked on every grandchild spawn) guarantees the recursion terminates. ---
    await run(
        runtime,
        child_id,
        tenant_id=tenant_id,
        provider=provider,
        invoker=invoker,
        assembler=assembler,
        clock=clock,
        max_steps=max_steps,
    )

    # Record the spawn as a durable tool_call on the parent's step. The child's TERMINAL
    # output is propagated into the parent's NEXT turn via the graph (not from here).
    await append_tool_call(
        runtime,
        step.id,
        ToolCallData(
            tool_ref=child_id,
            args=use.input,
            intent=use.intent,
            result="spawned",
            error_kind=None,
            hint=None,
        ),
        name=f"spawn-{step.data.turn_index}-{use.id}",
        tenant_id=tenant_id,
    )
    return None


def _opt_int(value: Any) -> Optional[int]:
    """Coerce a spawn-input budget value to a positive ``int``, or ``None``.

    A child's own ``step_budget`` / ``token_budget`` arrives as free-form tool input;
    only a positive integer sets a bound (anything else — absent, wrong type,
    non-positive — leaves the child's budget unbounded within the driver backstop).

    Args:
        value: The raw spawn-input value.

    Returns:
        The positive integer budget, or ``None``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 1:
        return value
    return None


def _evaluate_stops(
    policy: LoopPolicy,
    turn: AssistantTurn,
    uses: list[ToolUseBlock],
    saw_explicit_stop: bool,
) -> Optional[str]:
    """The first firing stop condition in SORTED order (deterministic; §2.1).

    The configured conditions are already ``sorted()`` on the policy; the first whose
    predicate fires is the recorded halt reason. ``max_steps``/``max_tokens`` are
    handled by the budget gates (their mirrors here would never out-fire the gates), so
    this evaluates the model-shaped conditions:

    * ``no_tool_calls`` — ``stop_reason=="end_turn"`` and zero ``tool_use`` blocks;
    * ``refusal`` — ``stop_reason=="refusal"``;
    * ``explicit_stop_node`` — the goal-completion sentinel tool was observed.

    Args:
        policy: The loop policy (its sorted ``stop_conditions``).
        turn: The assistant turn just produced.
        uses: The turn's ``tool_use`` blocks.
        saw_explicit_stop: Whether the sentinel stop tool was invoked this turn.

    Returns:
        The first firing condition's name, or ``None`` if none fired.
    """
    for condition in policy.stop_conditions:
        if condition == "no_tool_calls" and turn.stop_reason == "end_turn" and not uses:
            return "no_tool_calls"
        if condition == "refusal" and turn.stop_reason == "refusal":
            return "refusal"
        if condition == "explicit_stop_node" and saw_explicit_stop:
            return "explicit_stop_node"
    return None


__all__ = ["RunResult", "run"]
