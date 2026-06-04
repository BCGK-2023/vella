"""Freeze-timing consumer stub (M2): drive the frozen surface as M5 will.

The freeze-timing gate (plan §3): each pre-M5 contract must be exercised through the
exact call-shape the M5 interpreter's turn loop will use, so a wrong-shaped frozen
surface is caught at freeze time — not at M5. This is NOT the interpreter; it is the
minimal slice proving the ``ModelProvider`` + canonical-turn surface composes under
the interpreter's calling convention:

    build TurnRequest(messages, tools, params)
      -> await provider.turn(req)        # non-streaming wrapper
      -> async for event in provider.stream(req)  # streaming path
      -> assert the returned/assembled types are the canonical AssistantTurn / blocks
         the interpreter will pattern-match on.

No ``pytest-asyncio``: the async sequence runs via ``asyncio.run``.
"""

from __future__ import annotations

import asyncio

from vella.core import ToolDeclaration

from vella.agent import (
    AssistantTurn,
    Message,
    MockProvider,
    ScriptedText,
    ScriptedToolUse,
    ScriptedTurn,
    TextBlock,
    ToolUseBlock,
    TurnEvent,
    TurnParams,
    TurnRequest,
)
from vella.agent._assembler import TurnAssembler


def _interpreter_shaped_request() -> TurnRequest:
    # Exactly how M5 will build a request: assembled canonical messages + tool
    # schemas (core ToolDeclaration) + per-turn params from the loop policy.
    return TurnRequest(
        messages=(
            Message(role="system", content=(TextBlock(text="you are helpful"),)),
            Message(role="user", content=(TextBlock(text="search please"),)),
        ),
        tools=(
            ToolDeclaration(name="search", description="search the corpus"),
        ),
        params=TurnParams(max_tokens=256, tool_choice="model"),
    )


def test_non_streaming_turn_returns_canonical_assistant_turn() -> None:
    provider = MockProvider(
        [
            ScriptedTurn(
                blocks=(
                    ScriptedText(text="on it", fragments=2),
                    ScriptedToolUse(
                        id="c1",
                        name="search",
                        input={"q": "vella"},
                        intent="search vella",
                        fragments=5,
                    ),
                ),
                stop_reason="tool_use",
            )
        ]
    )

    async def _drive() -> AssistantTurn:
        req = _interpreter_shaped_request()
        return await provider.turn(req)

    turn = asyncio.run(_drive())
    # The interpreter pattern-matches on these canonical types.
    assert isinstance(turn, AssistantTurn)
    assert turn.stop_reason == "tool_use"
    tool_calls = [b for b in turn.content if isinstance(b, ToolUseBlock)]
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "search"
    assert tool_calls[0].input == {"q": "vella"}
    assert tool_calls[0].intent == "search vella"


def test_streaming_path_assembles_to_same_turn() -> None:
    script = ScriptedTurn(
        blocks=(
            ScriptedText(text="on it", fragments=2),
            ScriptedToolUse(
                id="c1", name="search", input={"q": "vella"}, intent="search vella", fragments=5
            ),
        ),
        stop_reason="tool_use",
    )

    async def _drive() -> tuple[AssistantTurn, AssistantTurn]:
        # Path 1: the non-streaming wrapper.
        non_streaming = await MockProvider([script]).turn(_interpreter_shaped_request())
        # Path 2: iterate stream() exactly as a streaming interpreter would, folding
        # each typed event through the public-shaped assembler.
        asm = TurnAssembler()
        events: list[TurnEvent] = []
        async for event in MockProvider([script]).stream(_interpreter_shaped_request()):
            events.append(event)
            asm.feed(event)
        streamed = asm.finish()
        return non_streaming, streamed

    non_streaming, streamed = asyncio.run(_drive())
    assert non_streaming.model_dump(mode="json") == streamed.model_dump(mode="json")


def test_forced_tool_choice_is_honored_in_request_shape() -> None:
    # The interpreter sets params.tool_choice="forced"; the provider must emit a
    # tool_use (the MockProvider asserts the contract).
    provider = MockProvider(
        [
            ScriptedTurn(
                blocks=(
                    ScriptedToolUse(id="c1", name="search", input={}, intent="go"),
                ),
                stop_reason="tool_use",
            )
        ]
    )

    async def _drive() -> AssistantTurn:
        req = TurnRequest(params=TurnParams(tool_choice="forced"))
        return await provider.turn(req)

    turn = asyncio.run(_drive())
    assert any(isinstance(b, ToolUseBlock) for b in turn.content)
