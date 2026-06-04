"""``MockProvider`` — the in-gate reference ``ModelProvider`` (deterministic, scriptable).

This is the deterministic reference implementation of the :class:`ModelProvider`
seam used throughout the gate. It is **scriptable**: a caller hands it an ordered
list of :class:`ScriptedTurn` items, and each call to :meth:`stream`/:meth:`turn`
emits the next one as the full streaming lifecycle. It deliberately **emulates the
OpenAI/OpenRouter function-calling dialect**: a tool call's ``input`` is streamed as
*fragmented* :class:`~vella.agent.provider.InputJsonDelta` events — multiple deltas
per ``tool_use`` index, split at the byte boundaries the script chooses — interleaved
with text/thinking deltas, supporting multi-tool-call turns. Because the same
deterministic accumulator (:mod:`vella.agent._assembler`) drains these events as it
will drain a live provider's, the mock and live paths prove the *identical* canonical
shape.

Determinism: the emitted event sequence is a pure function of the script (no clocks,
no randomness, no hash-order-dependent iteration). ``tool_choice:"forced"`` is
honored — a forced turn that scripts no tool call is a programming error (the gate's
interpreter is what would re-prompt a live model; the mock asserts the contract).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Literal, Optional, Union

from pydantic import Field
from vella.core import VellaModel

from ._assembler import drain
from .provider import (
    BlockStub,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    InputJsonDelta,
    MessageDelta,
    MessageStart,
    MessageStop,
    TextBlockStub,
    TextDelta,
    ThinkingBlockStub,
    ThinkingDelta,
    ToolUseBlockStub,
    TurnEvent,
    TurnRequest,
)
from .turn import AssistantTurn, StopReason, Usage


def _fragment(text: str, n: int) -> list[str]:
    """Split ``text`` into ``n`` near-even contiguous fragments (deterministic).

    The fragments concatenate back to ``text`` exactly; this is how the mock splits
    a tool call's argument JSON at arbitrary-but-reproducible byte boundaries to
    emulate streamed ``input_json_delta`` chunks. ``n`` is clamped to ``[1, len]``.

    Args:
        text: The string to fragment.
        n: The desired fragment count.

    Returns:
        The contiguous fragments in order (their concatenation equals ``text``).
    """
    if not text:
        return [""]
    n = max(1, min(n, len(text)))
    size = len(text) // n
    rem = len(text) % n
    out: list[str] = []
    pos = 0
    for i in range(n):
        extra = 1 if i < rem else 0
        out.append(text[pos : pos + size + extra])
        pos += size + extra
    return out


class ScriptedText(VellaModel):
    """A scripted text block, streamed as ``fragments`` text deltas.

    Attributes:
        kind: The discriminator literal (always ``"text"``).
        text: The full block text.
        fragments: How many text deltas to split ``text`` into (>= 1).
    """

    kind: Literal["text"] = "text"
    text: str
    fragments: int = 1


class ScriptedThinking(VellaModel):
    """A scripted thinking block, streamed as ``fragments`` thinking deltas.

    Attributes:
        kind: The discriminator literal (always ``"thinking"``).
        text: The full reasoning text.
        fragments: How many thinking deltas to split ``text`` into (>= 1).
    """

    kind: Literal["thinking"] = "thinking"
    text: str
    fragments: int = 1


class ScriptedToolUse(VellaModel):
    """A scripted tool call, streamed as ``fragments`` fragmented JSON deltas.

    The argument dict is serialized to canonical JSON (``sort_keys=True`` so the
    emitted bytes are deterministic) and split into ``fragments`` ``input_json_delta``
    chunks at byte boundaries — emulating the OpenAI/OpenRouter streamed-tool-call
    dialect.

    Attributes:
        kind: The discriminator literal (always ``"tool_use"``).
        id: The tool-call id.
        name: The tool's declared name.
        input: The call arguments (serialized + fragmented as deltas).
        intent: The one-sentence UX narration carried on the assembled block.
        fragments: How many JSON fragments to split the serialized ``input`` into.
    """

    kind: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    intent: str = ""
    fragments: int = 1


ScriptedBlock = Union[ScriptedText, ScriptedThinking, ScriptedToolUse]
"""One scripted content block in a :class:`ScriptedTurn`."""


class ScriptedTurn(VellaModel):
    """One canned turn: its ordered blocks plus the terminal stop reason + usage.

    The block order is the content order the canonical turn will read in (semantic;
    never sorted). When any block is a tool call, ``stop_reason`` is typically
    ``"tool_use"``.

    Attributes:
        blocks: The ordered scripted blocks emitted this turn.
        stop_reason: The turn's terminal stop reason.
        usage: The turn's token accounting (emitted on ``message_delta``).
        interleave: When ``True`` and the turn has multiple blocks, their deltas are
            emitted round-robin (proving index-keyed — not arrival-keyed — assembly);
            when ``False`` each block is streamed start-to-stop before the next.
    """

    blocks: tuple[ScriptedBlock, ...] = ()
    stop_reason: StopReason = "end_turn"
    usage: Usage = Field(default_factory=Usage)
    interleave: bool = False


def _block_stub(block: ScriptedBlock) -> BlockStub:
    """Build the opening :data:`BlockStub` for a scripted block."""
    if isinstance(block, ScriptedText):
        return TextBlockStub()
    if isinstance(block, ScriptedThinking):
        return ThinkingBlockStub()
    return ToolUseBlockStub(id=block.id, name=block.name, intent=block.intent)


def _block_deltas(index: int, block: ScriptedBlock) -> list[ContentBlockDelta]:
    """Build the ordered delta events for a scripted block at ``index``.

    Text/thinking are split into ``fragments`` text/thinking deltas; a tool call's
    ``input`` is serialized with ``sort_keys=True`` (deterministic bytes) and split
    into ``fragments`` ``input_json_delta`` chunks.
    """
    if isinstance(block, (ScriptedText, ScriptedThinking)):
        delta_cls = TextDelta if isinstance(block, ScriptedText) else ThinkingDelta
        return [
            ContentBlockDelta(index=index, delta=delta_cls(text=part))
            for part in _fragment(block.text, block.fragments)
        ]
    raw = json.dumps(block.input, sort_keys=True, separators=(",", ":"))
    return [
        ContentBlockDelta(index=index, delta=InputJsonDelta(partial_json=part))
        for part in _fragment(raw, block.fragments)
    ]


def _turn_events(turn: ScriptedTurn) -> list[TurnEvent]:
    """Lower a :class:`ScriptedTurn` into its full ordered lifecycle event list.

    Honors :attr:`ScriptedTurn.interleave`: when set, the per-block deltas are
    emitted round-robin between ``content_block_start`` and ``content_block_stop``
    of all blocks — exercising the accumulator's index-keyed (not arrival-keyed)
    assembly; otherwise each block runs start → deltas → stop before the next.
    """
    events: list[TurnEvent] = [MessageStart(usage=Usage())]
    n = len(turn.blocks)
    if turn.interleave and n > 1:
        for i, block in enumerate(turn.blocks):
            events.append(ContentBlockStart(index=i, block=_block_stub(block)))
        delta_lists = [_block_deltas(i, b) for i, b in enumerate(turn.blocks)]
        cursors = [0] * n
        remaining = sum(len(dl) for dl in delta_lists)
        while remaining > 0:
            for i in range(n):
                if cursors[i] < len(delta_lists[i]):
                    events.append(delta_lists[i][cursors[i]])
                    cursors[i] += 1
                    remaining -= 1
        for i in range(n):
            events.append(ContentBlockStop(index=i))
    else:
        for i, block in enumerate(turn.blocks):
            events.append(ContentBlockStart(index=i, block=_block_stub(block)))
            events.extend(_block_deltas(i, block))
            events.append(ContentBlockStop(index=i))
    events.append(MessageDelta(stop_reason=turn.stop_reason, usage=turn.usage))
    events.append(MessageStop())
    return events


def _has_tool_use(turn: ScriptedTurn) -> bool:
    """Whether a scripted turn contains at least one tool call."""
    return any(isinstance(b, ScriptedToolUse) for b in turn.blocks)


class MockProvider:
    """A deterministic, scriptable :class:`~vella.agent.ModelProvider` reference impl.

    Construct it with an ordered ``script`` of :class:`ScriptedTurn` items; each
    :meth:`stream`/:meth:`turn` call consumes the next turn and emits it as the full
    streaming lifecycle (fragmented tool-call JSON included). It satisfies the
    structural :class:`~vella.agent.ModelProvider` Protocol by shape.

    Examples:
        >>> import asyncio
        >>> from vella.agent import MockProvider, ScriptedTurn, ScriptedText
        >>> from vella.agent import TurnRequest
        >>> p = MockProvider([ScriptedTurn(blocks=(ScriptedText(text="hi"),))])
        >>> turn = asyncio.run(p.turn(TurnRequest()))
        >>> turn.content[0].text
        'hi'
    """

    def __init__(self, script: list[ScriptedTurn]) -> None:
        """Create a mock provider over an ordered script of canned turns.

        Args:
            script: The turns to emit, one per :meth:`stream`/:meth:`turn` call.
        """
        self._script: list[ScriptedTurn] = list(script)
        self._cursor: int = 0

    def _next_turn(self, request: TurnRequest) -> ScriptedTurn:
        """Pop the next scripted turn, enforcing the ``forced`` tool-choice contract.

        Args:
            request: The turn request (its ``params.tool_choice`` is checked).

        Returns:
            The next :class:`ScriptedTurn` in the script.

        Raises:
            IndexError: If the script is exhausted.
            ValueError: If ``tool_choice`` is ``"forced"`` but the scripted turn
                emits no tool call.
        """
        if self._cursor >= len(self._script):
            raise IndexError("MockProvider script exhausted")
        turn = self._script[self._cursor]
        self._cursor += 1
        if request.params.tool_choice == "forced" and not _has_tool_use(turn):
            raise ValueError(
                "tool_choice='forced' requires the scripted turn to emit >=1 tool_use"
            )
        return turn

    async def stream(self, request: TurnRequest) -> AsyncIterator[TurnEvent]:
        """Emit the next scripted turn as its full lifecycle event stream.

        Args:
            request: The assembled request (drives the ``forced`` contract check).

        Yields:
            The turn's :data:`~vella.agent.provider.TurnEvent` lifecycle in order.
        """
        turn = self._next_turn(request)
        for event in _turn_events(turn):
            yield event

    async def turn(self, request: TurnRequest) -> AssistantTurn:
        """Drain the next scripted turn's stream into a canonical assistant turn.

        Routes through the same deterministic accumulator the live path uses, so the
        result is byte-identical to assembling :meth:`stream`'s events by hand.

        Args:
            request: The assembled request.

        Returns:
            The assembled canonical :class:`~vella.agent.AssistantTurn`.
        """
        return await drain(self.stream(request))


def assistant_turn_from_blocks(
    blocks: tuple[ScriptedBlock, ...],
    *,
    stop_reason: StopReason = "end_turn",
    usage: Optional[Usage] = None,
) -> AssistantTurn:
    """Build the canonical turn a script's blocks assemble to (a test reference).

    Lowers the blocks through the SAME accumulator the streaming path uses (no
    interleaving needed for a single-pass reference), so a provider-agnostic test can
    assert two different dialect framings converge on this one canonical turn.

    Args:
        blocks: The scripted content blocks in semantic order.
        stop_reason: The turn's stop reason.
        usage: The turn's usage (defaults to a zeroed :class:`~vella.agent.Usage`).

    Returns:
        The canonical :class:`~vella.agent.AssistantTurn` the blocks assemble to.
    """
    turn = ScriptedTurn(
        blocks=blocks, stop_reason=stop_reason, usage=usage or Usage()
    )
    asm_events = _turn_events(turn)
    from ._assembler import assemble

    return assemble(asm_events)
