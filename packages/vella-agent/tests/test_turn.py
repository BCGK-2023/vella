"""The canonical turn models: frozen-ness, union round-trips, field presence.

The canonical turn is the frozen ``ModelProvider`` surface — every adapter and the
M5 interpreter speak it. These tests pin: (1) the models are frozen
(``VellaModel``); (2) the ``ContentBlock`` discriminated union round-trips each
concrete block by its ``type`` literal; (3) ``ToolUseBlock.intent`` and
``ToolResultBlock.hint`` fields exist; (4) ``Usage`` carries all five token counters
and ``AssistantTurn`` carries ``stop_reason``/``usage``; (5) content order is
preserved (semantic, never sorted). Comparison is via ``model_dump(mode="json")``.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from vella.agent import (
    AssistantTurn,
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

_BLOCK_ADAPTER: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)


def test_blocks_are_frozen() -> None:
    block = TextBlock(text="hi")
    with pytest.raises(ValidationError):
        block.text = "mutated"  # type: ignore[misc]


def test_assistant_turn_is_frozen() -> None:
    turn = AssistantTurn()
    with pytest.raises(ValidationError):
        turn.stop_reason = "refusal"  # type: ignore[misc]


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        TextBlock(text="x", surprise=1)  # type: ignore[call-arg]


def test_content_block_union_round_trips_each_variant() -> None:
    # Every concrete block round-trips back to its exact class via the `type` tag.
    blocks: list[ContentBlock] = [
        TextBlock(text="t"),
        ThinkingBlock(text="reasoning"),
        ToolUseBlock(id="c1", name="search", input={"q": "x"}, intent="search x"),
        ToolResultBlock(tool_use_id="c1", content="result", is_error=False, hint=None),
    ]
    for block in blocks:
        dumped = _BLOCK_ADAPTER.dump_python(block, mode="json")
        revived = _BLOCK_ADAPTER.validate_python(dumped)
        assert type(revived) is type(block)
        assert revived.model_dump(mode="json") == block.model_dump(mode="json")


def test_type_discriminators_are_fixed() -> None:
    assert TextBlock(text="").type == "text"
    assert ThinkingBlock(text="").type == "thinking"
    assert ToolUseBlock(id="i", name="n").type == "tool_use"
    assert ToolResultBlock(tool_use_id="i").type == "tool_result"


def test_tool_use_block_has_intent_field() -> None:
    block = ToolUseBlock(id="c1", name="search", intent="I search for X")
    assert block.intent == "I search for X"
    # default present (empty) when omitted
    assert ToolUseBlock(id="c1", name="search").intent == ""


def test_tool_result_block_has_hint_field() -> None:
    # hint exists now (resolved at M3); default None.
    assert ToolResultBlock(tool_use_id="c1").hint is None
    assert ToolResultBlock(tool_use_id="c1", hint="retry").hint == "retry"


def test_usage_carries_all_five_counters() -> None:
    usage = Usage(
        input_tokens=1,
        output_tokens=2,
        cache_read_tokens=3,
        cache_write_tokens=4,
        reasoning_tokens=5,
    )
    dumped = usage.model_dump(mode="json")
    assert set(dumped) == {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    }
    assert dumped["reasoning_tokens"] == 5


def test_assistant_turn_shape() -> None:
    turn = AssistantTurn(
        content=(TextBlock(text="hi"),),
        stop_reason="tool_use",
        usage=Usage(output_tokens=7),
    )
    assert turn.role == "assistant"
    assert turn.stop_reason == "tool_use"
    assert turn.usage.output_tokens == 7
    assert turn.content[0].model_dump(mode="json") == {"type": "text", "text": "hi"}


def test_content_order_is_preserved_not_sorted() -> None:
    # Content order is semantic — a reversed-alphabetical sequence must NOT be
    # reordered by serialization.
    content: tuple[ContentBlock, ...] = (
        TextBlock(text="zebra"),
        TextBlock(text="apple"),
        TextBlock(text="mango"),
    )
    msg = Message(role="assistant", content=content)
    texts = [b["text"] for b in msg.model_dump(mode="json")["content"]]
    assert texts == ["zebra", "apple", "mango"]


def test_stop_reason_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        AssistantTurn(stop_reason="exploded")
