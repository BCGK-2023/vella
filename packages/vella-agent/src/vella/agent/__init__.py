"""Vella agent (a self-hosted cognition core over the runtime + graph).

Where ``vella.runtime`` is *physics* — the append-only log, the
optimistic-concurrency store, and the write verbs that move world state forward —
and ``vella.graph`` is a *read-only projection* that answers traversal queries from
memory, ``vella.agent`` is the *cognition core*: a data-configured interpreter that
acts ONLY through the runtime's published verbs and perceives ONLY through the
graph's published projection. It owns no storage and takes no privileged path; an
agent run, its steps, tool calls, messages, and policy are all ordinary registered
core node types.

Design principles
-----------------
* **Self-hosting is the substrate, not a feature.** The agent acts only through
  ``vella.runtime``'s public verbs; cognition is nodes/edges and ``observe_only``
  telemetry. There is no new ``Node`` subclass and no edit pushed down into
  core/runtime/graph.
* **Three Protocol seams.** ``ModelProvider`` / ``ToolInvoker`` / ``ContextAssembler``
  each have a deterministic in-gate reference impl and optional out-of-gate real
  adapters (``[openrouter]`` / ``[mcp]``), so heavy I/O never enters the gate.
* **Determinism is a property, not a hope.** The interpreter is network-free under
  its reference impls; any set-derived serialized value is ``sorted()``; the gated
  determinism artifact is byte-identical across hash seeds.
* **Depend downward only.** The agent imports only the published ``vella.core``,
  ``vella.runtime``, and ``vella.graph`` surfaces — NEVER ``vella.reconciler`` (a
  sibling, not a dependency); all three lower layers are unaware of it.

The public surface grows milestone by milestone; everything in ``__all__`` is
importable, documented, and snapshotted by the surface tripwire from M0 onward. The
node type-specs, canonical-turn models, the three Protocols, and the FSM interpreter
land in later milestones; the surface is baselined now (empty) so the tripwire
guards it from the start.
"""

from __future__ import annotations

from ._discovery import (
    HAS_TOOL_EDGE,
    discover_tools,
    link_run_tool,
    seed_system_tools,
)
from ._hints import resolve_hint
from ._subagent import SPAWN_TOOL, max_run_tree_size
from .clock import Clock, ManualClock
from .context import (
    AssembledContext,
    AssemblyPolicy,
    CompactionPolicy,
    ContextAssembler,
)
from .graph_assembler import GraphContextAssembler
from .interpreter import RunResult, run
from .invoker import InMemoryToolInvoker, ToolDispatchError, ToolInvoker
from .policy import (
    EXPLICIT_STOP_TOOL,
    LoopPolicy,
    StopCondition,
    SubAgentAllow,
    SubAgentDeny,
    SubAgentSpawn,
    ToolChoice,
    ToolChoiceForced,
    ToolChoiceModel,
    ToolChoiceRestricted,
    register_policy_types,
)
from .mock_provider import (
    MockProvider,
    ScriptedText,
    ScriptedThinking,
    ScriptedToolUse,
    ScriptedTurn,
    assistant_turn_from_blocks,
)
from .provider import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    ContentDelta,
    InputJsonDelta,
    MessageDelta,
    MessageStart,
    MessageStop,
    ModelProvider,
    TextDelta,
    ThinkingDelta,
    ToolSchema,
    TurnEvent,
    TurnParams,
    TurnRequest,
)
from .tool import (
    Binding,
    BuiltinBinding,
    ErrorHint,
    HTTPBinding,
    MCPBinding,
    MCPServerData,
    RetryPolicy,
    ToolCallData,
    ToolData,
    ToolHints,
    ToolResult,
    register_tool_types,
)
from .turn import (
    AssistantTurn,
    ContentBlock,
    Message,
    MessageRole,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from .types import (
    MessageData,
    ProviderData,
    ProviderLimits,
    ProviderTransport,
    RunData,
    RunStatus,
    StepData,
    StepKind,
    SummaryData,
    agent_registry,
    register_agent_types,
)

__all__: list[str] = [
    # --- canonical turn (M2, frozen) ---
    "AssistantTurn",
    "ContentBlock",
    "Message",
    "MessageRole",
    "StopReason",
    "TextBlock",
    "ThinkingBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "Usage",
    # --- ModelProvider seam + streaming events (M2, frozen) ---
    "ContentBlockDelta",
    "ContentBlockStart",
    "ContentBlockStop",
    "ContentDelta",
    "InputJsonDelta",
    "MessageDelta",
    "MessageStart",
    "MessageStop",
    "ModelProvider",
    "TextDelta",
    "ThinkingDelta",
    "ToolSchema",
    "TurnEvent",
    "TurnParams",
    "TurnRequest",
    # --- MockProvider reference impl (M2) ---
    "MockProvider",
    "ScriptedText",
    "ScriptedThinking",
    "ScriptedToolUse",
    "ScriptedTurn",
    "assistant_turn_from_blocks",
    # --- self-hosting node type-specs (M1, content upgraded at M2) ---
    "MessageData",
    "RunData",
    "RunStatus",
    "StepData",
    "StepKind",
    "SummaryData",
    "agent_registry",
    "register_agent_types",
    # --- provider node type-spec (M4, owns cache-strategy capability) ---
    "ProviderData",
    "ProviderLimits",
    "ProviderTransport",
    # --- ContextAssembler seam + in-gate reference impl (M4, frozen) ---
    "AssembledContext",
    "AssemblyPolicy",
    "CompactionPolicy",
    "ContextAssembler",
    "GraphContextAssembler",
    # --- tool-node + invoker contract (M3, frozen) ---
    "Binding",
    "BuiltinBinding",
    "ErrorHint",
    "HTTPBinding",
    "MCPBinding",
    "MCPServerData",
    "RetryPolicy",
    "ToolCallData",
    "ToolData",
    "ToolHints",
    "ToolResult",
    "register_tool_types",
    # --- ToolInvoker seam + in-gate reference impl (M3) ---
    "InMemoryToolInvoker",
    "ToolDispatchError",
    "ToolInvoker",
    # --- loop_policy FSM schema (M5, frozen) ---
    "EXPLICIT_STOP_TOOL",
    "LoopPolicy",
    "StopCondition",
    "SubAgentAllow",
    "SubAgentDeny",
    "SubAgentSpawn",
    "ToolChoice",
    "ToolChoiceForced",
    "ToolChoiceModel",
    "ToolChoiceRestricted",
    "register_policy_types",
    # --- FSM interpreter entry point (M5, frozen) ---
    "RunResult",
    "run",
    # --- bounded sub-agents (M6, frozen) ---
    "SPAWN_TOOL",
    "max_run_tree_size",
    # --- Clock seam (M3; SystemClock is the unexported production default) ---
    "Clock",
    "ManualClock",
    # --- discovery + hint entry points (M3) ---
    "HAS_TOOL_EDGE",
    "discover_tools",
    "link_run_tool",
    "resolve_hint",
    "seed_system_tools",
]
