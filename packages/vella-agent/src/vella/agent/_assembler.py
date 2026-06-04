"""The deterministic streaming accumulator — the load-bearing determinism surface.

This is pre-mortem #1's mitigation in code: a streaming turn is reconstructed into a
canonical :class:`~vella.agent.AssistantTurn` by a strict, order-stable fold over the
:data:`~vella.agent.provider.TurnEvent` lifecycle, with three non-negotiable rules:

1. **Key strictly by the integer ``content_block`` index.** Block state lives in a
   dict keyed by ``index``; the assembled turn iterates indices in ``sorted()``
   order. Arrival order of *events* never determines block order — only the index
   does — so two providers that interleave their blocks differently (e.g. a
   multi-tool-call turn whose deltas arrive round-robin vs. block-at-a-time) assemble
   to the *same* turn.
2. **Append fragments in stream order.** Within one block the text / thinking /
   ``partial_json`` fragments are concatenated in the order they arrive (that order
   IS the content), never reordered.
3. **Parse tool JSON exactly once, at ``content_block_stop``.** ``partial_json``
   fragments are accumulated as a raw string and ``json.loads``-ed a single time when
   the block closes — never on a partial buffer (a partial parse is both wrong and
   nondeterministic).

There is no dict/set iteration whose order leaks into the result: the only ordering
is ``sorted()`` over integer indices and append-order within a block. This module is
private (underscore) — it is the mechanism behind the public ``ModelProvider`` seam,
not itself part of the frozen surface.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from .provider import (
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageDelta,
    MessageStart,
    TextDelta,
    ThinkingDelta,
    ToolUseBlockStub,
    TurnEvent,
)
from .turn import (
    AssistantTurn,
    ContentBlock,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)


class _BlockBuf:
    """Mutable per-index accumulation buffer for one streamed content block.

    Keyed by the block's integer index in :class:`TurnAssembler`; carries the
    opening stub's identity plus the fragments appended in stream order. Tool
    ``partial_json`` is held as a raw string and parsed once at block stop.
    """

    __slots__ = ("kind", "text_parts", "json_parts", "tool_id", "tool_name", "intent")

    def __init__(self, kind: str) -> None:
        """Open a buffer for a block of the given stub ``kind``.

        Args:
            kind: The stub discriminator (``"text"`` | ``"thinking"`` |
                ``"tool_use"``).
        """
        self.kind = kind
        self.text_parts: list[str] = []
        self.json_parts: list[str] = []
        self.tool_id: str = ""
        self.tool_name: str = ""
        self.intent: str = ""

    def finish(self) -> ContentBlock:
        """Materialize this buffer into its frozen canonical content block.

        For a tool-use block the accumulated ``partial_json`` is concatenated in
        stream order and ``json.loads``-ed exactly once here (the sole parse point);
        an empty buffer parses to ``{}``.

        Returns:
            The assembled :class:`~vella.agent.TextBlock` /
            :class:`~vella.agent.ThinkingBlock` /
            :class:`~vella.agent.ToolUseBlock`.

        Raises:
            ValueError: If the buffer's ``kind`` is not a known block kind.
        """
        if self.kind == "text":
            return TextBlock(text="".join(self.text_parts))
        if self.kind == "thinking":
            return ThinkingBlock(text="".join(self.text_parts))
        if self.kind == "tool_use":
            raw = "".join(self.json_parts)
            parsed: dict[str, Any] = json.loads(raw) if raw.strip() else {}
            return ToolUseBlock(
                id=self.tool_id,
                name=self.tool_name,
                input=parsed,
                intent=self.intent,
            )
        raise ValueError(f"unknown content-block kind: {self.kind!r}")


class TurnAssembler:
    """A stateful, deterministic fold from streaming events to one canonical turn.

    Feed each :data:`~vella.agent.provider.TurnEvent` in stream order via
    :meth:`feed`, then call :meth:`finish` to obtain the
    :class:`~vella.agent.AssistantTurn`. Block ordering in the result is by
    ``sorted()`` integer index; intra-block order is append order; tool JSON is
    parsed once at block stop. The assembler is single-use (one turn per instance).
    """

    def __init__(self) -> None:
        """Create an empty assembler positioned before ``message_start``."""
        self._blocks: dict[int, _BlockBuf] = {}
        self._usage: Usage = Usage()
        self._stop_reason: StopReason = "end_turn"

    def feed(self, event: TurnEvent) -> None:
        """Fold one lifecycle event into the in-progress turn.

        Args:
            event: The next :data:`~vella.agent.provider.TurnEvent` in stream order.
        """
        if isinstance(event, MessageStart):
            self._usage = event.usage
            return
        if isinstance(event, ContentBlockStart):
            stub = event.block
            buf = _BlockBuf(stub.type)
            if isinstance(stub, ToolUseBlockStub):
                # The id/name/intent are known when the block opens; only the
                # argument JSON streams afterward as fragments. Text/thinking stubs
                # carry no opening payload (their text arrives entirely as deltas).
                buf.tool_id = stub.id
                buf.tool_name = stub.name
                buf.intent = stub.intent
            self._blocks[event.index] = buf
            return
        if isinstance(event, ContentBlockDelta):
            buf = self._blocks[event.index]
            delta = event.delta
            if isinstance(delta, (TextDelta, ThinkingDelta)):
                # Text/thinking fragments are appended in stream order (that order
                # IS the content); a tool call's JSON fragments accumulate as raw
                # string parts, parsed once at block stop (never on a partial).
                buf.text_parts.append(delta.text)
            else:
                buf.json_parts.append(delta.partial_json)
            return
        if isinstance(event, ContentBlockStop):
            # Closing is a no-op for buffering; the single parse happens in finish().
            return
        if isinstance(event, MessageDelta):
            self._stop_reason = event.stop_reason
            self._usage = event.usage
            return
        # MessageStop: terminal marker, nothing to fold.

    def finish(self) -> AssistantTurn:
        """Assemble the folded state into a frozen canonical assistant turn.

        Blocks are emitted in ``sorted()`` integer-index order (never event-arrival
        order), each materialized via :meth:`_BlockBuf.finish`.

        Returns:
            The canonical :class:`~vella.agent.AssistantTurn`.
        """
        content = tuple(self._blocks[i].finish() for i in sorted(self._blocks))
        return AssistantTurn(
            content=content, stop_reason=self._stop_reason, usage=self._usage
        )


def assemble(events: list[TurnEvent]) -> AssistantTurn:
    """Fold a fully materialized event list into one canonical assistant turn.

    A synchronous convenience over :class:`TurnAssembler` for tests/callers that
    already hold the events.

    Args:
        events: The lifecycle events in stream order.

    Returns:
        The assembled canonical :class:`~vella.agent.AssistantTurn`.
    """
    asm = TurnAssembler()
    for event in events:
        asm.feed(event)
    return asm.finish()


async def drain(stream: AsyncIterator[TurnEvent]) -> AssistantTurn:
    """Drain a provider's event stream into one canonical assistant turn.

    This is the non-streaming convenience behind ``ModelProvider.turn``: it folds
    the live event iterator through :class:`TurnAssembler`, so the result is
    byte-identical (under ``model_dump(mode="json")``) to assembling the same
    events synchronously via :func:`assemble`.

    Args:
        stream: The provider's async iterator of lifecycle events.

    Returns:
        The assembled canonical :class:`~vella.agent.AssistantTurn`.
    """
    asm = TurnAssembler()
    async for event in stream:
        asm.feed(event)
    return asm.finish()
