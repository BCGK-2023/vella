"""Network-free OpenRouter adapter smoke (``[openrouter]``, marked ``extras``).

Feeds a RECORDED OpenRouter SSE stream (realistic: fragmented text, fragmented
``tool_calls[].function.arguments``, then a terminal ``finish_reason`` + ``usage``)
through ``httpx.MockTransport`` and asserts the :class:`OpenRouterProvider`
translates it to the canonical :class:`AssistantTurn` VIA THE SAME ``_assembler`` —
byte-identical (under ``model_dump(mode="json")``) to feeding the equivalent
canonical events to that assembler directly. No live key, no network.

This is the milestone's mutation guard (b): if the adapter assembled deltas with
its OWN accumulator instead of the shared one, this equivalence would break.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("httpx")

import httpx  # noqa: E402

from vella.agent import TurnRequest, Usage  # noqa: E402
from vella.agent._assembler import assemble  # noqa: E402
from vella.agent.provider import TurnEvent  # noqa: E402
from vella.agent.adapters.openrouter import (  # noqa: E402
    _TEXT_INDEX,
    _TOOL_INDEX_BASE,
    OpenRouterProvider,
)
from vella.agent.provider import (  # noqa: E402
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    InputJsonDelta,
    MessageDelta,
    MessageStart,
    MessageStop,
    TextBlockStub,
    TextDelta,
    ToolUseBlockStub,
)

# A realistic OpenRouter/OpenAI-dialect streamed chat-completion: a couple of text
# fragments, a tool call whose id+name open first and whose argument JSON arrives in
# THREE fragments split at arbitrary byte boundaries, then a finish_reason chunk and
# a final usage-only chunk (stream_options.include_usage), then [DONE].
_SSE_CHUNKS = [
    '{"choices":[{"delta":{"role":"assistant","content":"Let me "}}]}',
    '{"choices":[{"delta":{"content":"search."}}]}',
    '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
    '"function":{"name":"search","arguments":"{\\"q\\":"}}]}}]}',
    '{"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"\\"vella"}}]}}]}',
    '{"choices":[{"delta":{"tool_calls":[{"index":0,'
    '"function":{"arguments":"\\"}"}}]}}]}',
    '{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
    '{"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":7,'
    '"prompt_tokens_details":{"cached_tokens":40},'
    '"completion_tokens_details":{"reasoning_tokens":0}}}',
]


def _sse_body() -> bytes:
    """Render the recorded chunks as an SSE ``data:`` body terminated by [DONE]."""
    lines = [f"data: {chunk}\n\n" for chunk in _SSE_CHUNKS]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _handler(request: httpx.Request) -> httpx.Response:
    """Serve the recorded SSE body for the chat-completions request."""
    assert request.url.path.endswith("/chat/completions")
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_sse_body(),
    )


def _canonical_reference() -> dict[str, Any]:
    """The turn the SAME assembler yields from the equivalent canonical events.

    These are exactly the events the adapter SHOULD emit for ``_SSE_CHUNKS`` — text
    on index 0, a tool call on ``_TOOL_INDEX_BASE`` with its argument JSON in the
    same three fragments — fed directly to the shared accumulator.
    """
    events: list[TurnEvent] = [
        MessageStart(usage=Usage()),
        ContentBlockStart(index=_TEXT_INDEX, block=TextBlockStub()),
        ContentBlockDelta(index=_TEXT_INDEX, delta=TextDelta(text="Let me ")),
        ContentBlockDelta(index=_TEXT_INDEX, delta=TextDelta(text="search.")),
        ContentBlockStart(
            index=_TOOL_INDEX_BASE,
            block=ToolUseBlockStub(id="call_1", name="search"),
        ),
        ContentBlockDelta(
            index=_TOOL_INDEX_BASE, delta=InputJsonDelta(partial_json='{"q":')
        ),
        ContentBlockDelta(
            index=_TOOL_INDEX_BASE, delta=InputJsonDelta(partial_json='"vella')
        ),
        ContentBlockDelta(
            index=_TOOL_INDEX_BASE, delta=InputJsonDelta(partial_json='"}')
        ),
        ContentBlockStop(index=_TEXT_INDEX),
        ContentBlockStop(index=_TOOL_INDEX_BASE),
        MessageDelta(
            stop_reason="tool_use",
            usage=Usage(
                input_tokens=42, output_tokens=7, cache_read_tokens=40
            ),
        ),
        MessageStop(),
    ]
    return assemble(events).model_dump(mode="json")


def test_openrouter_stream_assembles_to_canonical_via_shared_assembler() -> None:
    provider = OpenRouterProvider(
        model_id="x/y",
        api_key="sk-test",
        transport=httpx.MockTransport(_handler),
    )

    async def _run() -> dict[str, Any]:
        try:
            turn = await provider.turn(TurnRequest())
            return turn.model_dump(mode="json")
        finally:
            await provider.aclose()

    got = asyncio.run(_run())

    # The adapter's drained turn is byte-identical to feeding the equivalent
    # canonical events to the SAME shared accumulator.
    assert got == _canonical_reference()
    # Sanity: the fragmented argument JSON parsed once at block stop into the dict.
    tool_block = next(b for b in got["content"] if b["type"] == "tool_use")
    assert tool_block["input"] == {"q": "vella"}
    assert tool_block["id"] == "call_1"
    assert tool_block["name"] == "search"
    assert got["stop_reason"] == "tool_use"
    assert got["usage"]["cache_read_tokens"] == 40
