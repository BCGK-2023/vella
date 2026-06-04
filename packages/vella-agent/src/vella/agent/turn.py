"""The canonical turn contract — the frozen ``ModelProvider`` surface (M2).

This is the most-depended-on frozen surface in ``vella.agent``: every provider
adapter (the in-gate :class:`~vella.agent.MockProvider`, the out-of-gate
``OpenRouterProvider`` at M7) translates its dialect *into* these types, and the M5
interpreter pattern-matches *only* on them — it never sees a provider specific.
Freezing it here (M2, the milestone that owns the content contract) is what lets the
interpreter be written against one shape regardless of the underlying model dialect.

Shapes (locked, from the spec's "Canonical Turn Contract"):

* :class:`Message` — ``{role, content}``. ``content`` is a ``tuple`` of
  :data:`ContentBlock`; its order is **semantic** (the order the model emitted /
  the conversation reads in) and is therefore NEVER sorted. This is the one place
  the package keeps ordering by insertion rather than by ``sorted()``: the content
  sequence is meaning, not a set-derived artifact.
* :data:`ContentBlock` — a discriminated union over the ``type`` literal, the same
  idiom as core ``Overlay``/``Actuator`` (a ``Literal`` ``kind``/``type`` tag +
  ``Field(discriminator=...)``): :class:`TextBlock` | :class:`ThinkingBlock` |
  :class:`ToolUseBlock` | :class:`ToolResultBlock`.
* :class:`Usage` — the canonical token accounting (``input``/``output``/
  ``cache_read``/``cache_write``/``reasoning``); cost stays provider-specific and is
  deliberately NOT canonical.
* :data:`StopReason` — the closed vocabulary of why a turn ended.
* :class:`AssistantTurn` — what a :class:`~vella.agent.ModelProvider` returns: an
  assistant :class:`Message` plus its :class:`Usage` and :data:`StopReason`.

All models are frozen :class:`~vella.core.VellaModel` (frozen + ``extra='forbid'``),
so equality/round-trip is via ``model_dump(mode="json")`` (never ``==`` — core
attaches a private registry attr). No field here is set-derived, so nothing is
``sorted()``: a content tuple's order is the contract.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import Field
from vella.core import VellaModel

StopReason = Literal[
    "end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal"
]
"""Why an :class:`AssistantTurn` ended — the closed canonical vocabulary.

* ``end_turn`` — the model finished its reply normally.
* ``tool_use`` — the model emitted one or more :class:`ToolUseBlock` items and is
  waiting for their results.
* ``max_tokens`` — the provider hit its output-token ceiling mid-turn.
* ``stop_sequence`` — a configured stop string was produced.
* ``refusal`` — the model declined to answer.
"""

MessageRole = Literal["system", "user", "assistant", "tool"]
"""The author role of a canonical :class:`Message`."""


class TextBlock(VellaModel):
    """Plain assistant/user text content.

    Attributes:
        type: The discriminator literal (always ``"text"``).
        text: The textual content of the block.
    """

    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(VellaModel):
    """Model reasoning content — re-feedable across turns and mirrored to telemetry.

    The token-level reasoning trace is also emitted as ``observe_only`` telemetry
    (locked decision #1); as a content block it can be fed back into a later turn so
    the model retains its chain of thought.

    Attributes:
        type: The discriminator literal (always ``"thinking"``).
        text: The reasoning text.
    """

    type: Literal["thinking"] = "thinking"
    text: str


class ToolUseBlock(VellaModel):
    """A model request to invoke a tool, with its assembled arguments.

    The ``input`` dict is assembled by the deterministic streaming accumulator
    (:mod:`vella.agent._assembler`) from fragmented ``input_json_delta`` events and
    parsed exactly once, at ``content_block_stop`` — never from partial JSON.

    Attributes:
        type: The discriminator literal (always ``"tool_use"``).
        id: The provider-assigned id correlating this call to its
            :class:`ToolResultBlock` (``tool_use_id``).
        name: The tool's declared name (matches a tool-node ``declaration.name``).
        input: The fully assembled, parsed call arguments.
        intent: A short single-sentence natural-language statement of what the call
            is for (UX narration); enforced/elicited by the ``require_tool_intent``
            loop-policy knob at M5 and recorded on the ``agent.tool_call`` node.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    intent: str = ""


class ToolResultBlock(VellaModel):
    """The result of an invoked tool, fed back to the model on the next turn.

    Attributes:
        type: The discriminator literal (always ``"tool_result"``).
        tool_use_id: The :class:`ToolUseBlock` ``id`` this result answers.
        content: The tool's output payload (free-form; serialized as-is).
        is_error: Whether the invocation failed.
        hint: Interpretation guidance resolved by the harness from the tool-node's
            ``hints`` (the result hint on success; the matching error hint, else the
            default error hint, on failure). The field exists at M2; hint
            *resolution* lands at M3. ``None`` until resolved.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Any = None
    is_error: bool = False
    hint: Optional[str] = None


ContentBlock = Annotated[
    Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock],
    Field(discriminator="type"),
]
"""A single content block — discriminated by its ``type`` literal.

The same idiom as core ``Overlay``/``Actuator``: a closed union tagged by a
``Literal`` field, resolved by pydantic via ``Field(discriminator="type")`` so a
serialized block round-trips back to its exact concrete class.
"""


class Usage(VellaModel):
    """Canonical token accounting for one turn (cost stays provider-specific).

    Attributes:
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens produced.
        cache_read_tokens: Prompt tokens served from the provider's cache.
        cache_write_tokens: Prompt tokens written into the provider's cache.
        reasoning_tokens: Tokens spent on hidden reasoning (thinking).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


class Message(VellaModel):
    """One canonical conversation message: a role plus an ordered block sequence.

    ``content`` order is **semantic** — it is the order the blocks read in, not a
    set-derived artifact — so it is NEVER ``sorted()`` (the one deliberate exception
    to the package's sort-set-derived-values rule).

    Attributes:
        role: Who authored the message.
        content: The ordered tuple of content blocks (order is meaning).
    """

    role: MessageRole
    content: tuple[ContentBlock, ...] = ()


class AssistantTurn(VellaModel):
    """The canonical assistant turn a :class:`~vella.agent.ModelProvider` returns.

    The streaming path (drained by :mod:`vella.agent._assembler`) and the
    non-streaming wrapper both produce this exact shape — byte-identical under
    ``model_dump(mode="json")`` — so the interpreter is indifferent to which path
    produced it.

    Attributes:
        role: Always ``"assistant"``.
        content: The ordered assembled blocks (order is meaning; never sorted).
        stop_reason: Why the turn ended.
        usage: The turn's token accounting.
    """

    role: Literal["assistant"] = "assistant"
    content: tuple[ContentBlock, ...] = ()
    stop_reason: StopReason = "end_turn"
    usage: Usage = Field(default_factory=Usage)
