"""The ``ModelProvider`` seam: streaming lifecycle events + the inference Protocol.

This is the inference seam of the three (``ModelProvider`` / ``ToolInvoker`` /
``ContextAssembler``). It is **streaming-first**: a provider yields a typed sequence
of lifecycle events (:data:`TurnEvent`) that the deterministic accumulator
(:mod:`vella.agent._assembler`) folds into a canonical
:class:`~vella.agent.AssistantTurn`; the non-streaming :meth:`ModelProvider.turn`
convenience is exactly that drain.

Event lifecycle (locked, mirrors the live SSE dialects every adapter normalizes):

    MessageStart(usage)
      ContentBlockStart(index, block_stub)              # text | thinking | tool_use{id,name}
      ContentBlockDelta(index, delta)                   # TextDelta | ThinkingDelta | InputJsonDelta
      ContentBlockStop(index)
    MessageDelta(stop_reason, usage)
    MessageStop

Both the event union and the delta union are discriminated by a ``type`` literal
(the same idiom as core ``Overlay``/``Actuator`` and the canonical content blocks),
so a serialized event round-trips back to its exact class and a consumer can
exhaustively match on ``type``.

The :class:`ModelProvider` Protocol is **structural** (like runtime's ``Store``): an
adapter satisfies it by shape, never by inheritance — so the in-gate
:class:`~vella.agent.MockProvider` and a future ``OpenRouterProvider`` are
interchangeable to the interpreter without a common base class.
"""

from __future__ import annotations

from typing import (
    Annotated,
    AsyncIterator,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from pydantic import Field
from vella.core import ToolDeclaration, VellaModel

from .turn import AssistantTurn, Message, StopReason, Usage

# --- content-block stubs (the identity a block opens with, before its deltas) ---


class TextBlockStub(VellaModel):
    """The opening stub of a streamed text block (no text yet).

    Attributes:
        type: The discriminator literal (always ``"text"``).
    """

    type: Literal["text"] = "text"


class ThinkingBlockStub(VellaModel):
    """The opening stub of a streamed thinking block (no text yet).

    Attributes:
        type: The discriminator literal (always ``"thinking"``).
    """

    type: Literal["thinking"] = "thinking"


class ToolUseBlockStub(VellaModel):
    """The opening stub of a streamed tool-use block: identity known up front.

    The arguments arrive afterward as :class:`InputJsonDelta` fragments; the stub
    carries the call identity (id + name) and the model's ``intent`` narration —
    both known when the block opens, before any argument JSON streams — so the
    accumulator can key and complete the eventual :class:`~vella.agent.ToolUseBlock`.

    Attributes:
        type: The discriminator literal (always ``"tool_use"``).
        id: The provider-assigned tool-call id.
        name: The tool's declared name.
        intent: The one-sentence UX narration the model emits for the call (carried
            onto the assembled block); empty when the dialect omits it.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    intent: str = ""


BlockStub = Annotated[
    Union[TextBlockStub, ThinkingBlockStub, ToolUseBlockStub],
    Field(discriminator="type"),
]
"""The opening identity of a streamed content block, discriminated by ``type``."""

# --- content-block deltas (incremental fragments within an open block) ---


class TextDelta(VellaModel):
    """An incremental fragment of text appended to an open text block.

    Attributes:
        type: The discriminator literal (always ``"text_delta"``).
        text: The text fragment to append in stream order.
    """

    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(VellaModel):
    """An incremental fragment appended to an open thinking block.

    Attributes:
        type: The discriminator literal (always ``"thinking_delta"``).
        text: The reasoning fragment to append in stream order.
    """

    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


class InputJsonDelta(VellaModel):
    """A fragment of a tool-call's argument JSON (assembled, parsed only at stop).

    The provider streams a tool call's ``input`` as a sequence of these
    partial-JSON fragments split at arbitrary byte boundaries. The accumulator
    concatenates them in stream order and ``json.loads`` the result **once**, at
    ``content_block_stop`` — never a partial parse (the determinism invariant).

    Attributes:
        type: The discriminator literal (always ``"input_json_delta"``).
        partial_json: A raw JSON fragment (NOT independently valid JSON).
    """

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


ContentDelta = Annotated[
    Union[TextDelta, ThinkingDelta, InputJsonDelta],
    Field(discriminator="type"),
]
"""An incremental content fragment, discriminated by ``type``."""

# --- top-level lifecycle events ---


class MessageStart(VellaModel):
    """Opens a streamed turn, carrying the initial usage snapshot.

    Attributes:
        type: The discriminator literal (always ``"message_start"``).
        usage: The initial token accounting (often only ``input_tokens``).
    """

    type: Literal["message_start"] = "message_start"
    usage: Usage = Field(default_factory=Usage)


class ContentBlockStart(VellaModel):
    """Opens a content block at a given integer index with its identifying stub.

    Attributes:
        type: The discriminator literal (always ``"content_block_start"``).
        index: The block's integer index — the SOLE key the accumulator orders by.
        block: The opening :data:`BlockStub` (text | thinking | tool_use identity).
    """

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    block: BlockStub


class ContentBlockDelta(VellaModel):
    """An incremental fragment for the block open at ``index``.

    Attributes:
        type: The discriminator literal (always ``"content_block_delta"``).
        index: The integer index of the block this delta extends.
        delta: The :data:`ContentDelta` fragment to append in stream order.
    """

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: ContentDelta


class ContentBlockStop(VellaModel):
    """Closes the block open at ``index`` (the sole parse point for tool JSON).

    Attributes:
        type: The discriminator literal (always ``"content_block_stop"``).
        index: The integer index of the block being closed.
    """

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDelta(VellaModel):
    """Carries the turn's terminal ``stop_reason`` and final usage delta.

    Attributes:
        type: The discriminator literal (always ``"message_delta"``).
        stop_reason: Why the turn ended.
        usage: The final usage snapshot/delta for the turn.
    """

    type: Literal["message_delta"] = "message_delta"
    stop_reason: StopReason = "end_turn"
    usage: Usage = Field(default_factory=Usage)


class MessageStop(VellaModel):
    """Terminates a streamed turn (no payload).

    Attributes:
        type: The discriminator literal (always ``"message_stop"``).
    """

    type: Literal["message_stop"] = "message_stop"


TurnEvent = Annotated[
    Union[
        MessageStart,
        ContentBlockStart,
        ContentBlockDelta,
        ContentBlockStop,
        MessageDelta,
        MessageStop,
    ],
    Field(discriminator="type"),
]
"""A single streaming lifecycle event, discriminated by ``type``."""

# --- the request the interpreter hands the provider ---

ToolSchema = ToolDeclaration
"""The tool schema a provider is offered — reuses core ``ToolDeclaration``.

A tool-node's ``declaration`` IS a :class:`~vella.core.ToolDeclaration`
(``name``/``description``/``parameters``/``returns``); the request carries those
declarations directly rather than a parallel agent-local shape.
"""


class TurnParams(VellaModel):
    """Per-turn inference knobs, sourced from the loop policy + provider node.

    Kept minimal but sufficient for M2: the streaming/non-streaming paths and
    ``tool_choice`` handling exercise these; richer policy lives in the M5
    ``loop_policy`` schema and is threaded in via the interpreter.

    Attributes:
        max_tokens: Output-token ceiling for the turn, or ``None`` for the
            provider default.
        temperature: Sampling temperature, or ``None`` for the provider default.
        tool_choice: ``"model"`` (model decides), ``"forced"`` (model MUST emit
            at least one tool_use), or ``"none"`` (no tools this turn).
        stop: Stop sequences (order is semantic — passed through as given).
        cache: Whether the request should insert cache breakpoints (honored only
            when the provider node declares cache capability).
    """

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    tool_choice: Literal["model", "forced", "none"] = "model"
    stop: tuple[str, ...] = ()
    cache: bool = False


class TurnRequest(VellaModel):
    """The full request the interpreter hands a :class:`ModelProvider`.

    Attributes:
        messages: The assembled conversation (order is semantic; never sorted).
        tools: The tool schemas the model may call (a tuple of
            :class:`~vella.core.ToolDeclaration`).
        params: The per-turn inference knobs.
    """

    messages: tuple[Message, ...] = ()
    tools: tuple[ToolSchema, ...] = ()
    params: TurnParams = Field(default_factory=TurnParams)


@runtime_checkable
class ModelProvider(Protocol):
    """The inference seam — streaming-first, structurally typed.

    An adapter satisfies this by shape alone (no inheritance), exactly like
    runtime's ``Store``: :meth:`stream` yields the typed :data:`TurnEvent`
    lifecycle, and :meth:`turn` is the non-streaming convenience that drains
    :meth:`stream` through the deterministic accumulator into one canonical
    :class:`~vella.agent.AssistantTurn`. The interpreter depends only on this
    Protocol and the canonical turn types — never on a concrete adapter.
    """

    def stream(self, request: TurnRequest) -> AsyncIterator[TurnEvent]:
        """Yield the turn's lifecycle events for ``request`` in stream order.

        Args:
            request: The assembled messages + tool schemas + per-turn params.

        Returns:
            An async iterator over the typed :data:`TurnEvent` lifecycle.
        """
        ...

    async def turn(self, request: TurnRequest) -> AssistantTurn:
        """Drain :meth:`stream` into one canonical :class:`AssistantTurn`.

        The wrapper folds the event stream through the deterministic accumulator,
        so its result is byte-identical (under ``model_dump(mode="json")``) to
        assembling the same events by hand.

        Args:
            request: The assembled messages + tool schemas + per-turn params.

        Returns:
            The assembled canonical assistant turn.
        """
        ...
