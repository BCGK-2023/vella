"""Provider-agnostic canonical turn: two dialect framings, one canonical shape.

The interpreter must see ONLY canonical types — never a provider specific. This
proves: (1) two ``MockProvider`` scripts encoding the *same* logical turn under
DIFFERENT dialect framings (different fragmentation, different interleaving,
block-at-a-time vs. round-robin tool-call deltas) assemble to a byte-identical
canonical turn; (2) the assembled turn's serialized keys are EXACTLY the canonical
union's keys — a dialect-specific field cannot leak through. Async drains run via
``asyncio.run`` (no ``pytest-asyncio``).
"""

from __future__ import annotations

import asyncio

from vella.agent import (
    AssistantTurn,
    MockProvider,
    ScriptedText,
    ScriptedThinking,
    ScriptedToolUse,
    ScriptedTurn,
    TurnRequest,
)
from vella.agent.mock_provider import ScriptedBlock

# The canonical per-block JSON key sets — nothing dialect-specific may appear.
_ALLOWED_KEYS = {
    "text": {"type", "text"},
    "thinking": {"type", "text"},
    "tool_use": {"type", "id", "name", "input", "intent"},
    "tool_result": {"type", "tool_use_id", "content", "is_error", "hint"},
}


def _drain(turn: ScriptedTurn) -> AssistantTurn:
    return asyncio.run(MockProvider([turn]).turn(TurnRequest()))


def test_two_dialect_framings_assemble_identically() -> None:
    # Same LOGICAL turn: one thinking + one text + two tool calls with the same
    # inputs. Dialect A: block-at-a-time, coarse fragments. Dialect B: interleaved
    # deltas, fine byte-level fragmentation. The canonical result must match.
    def logical_blocks(frags: int) -> tuple[ScriptedBlock, ...]:
        return (
            ScriptedThinking(text="reason about it", fragments=frags),
            ScriptedText(text="calling tools now", fragments=frags),
            ScriptedToolUse(
                id="c1", name="search", input={"q": "x", "k": 3}, intent="search", fragments=frags
            ),
            ScriptedToolUse(
                id="c2", name="fetch", input={"u": "y"}, intent="fetch", fragments=frags
            ),
        )

    dialect_a = ScriptedTurn(
        blocks=logical_blocks(1), stop_reason="tool_use", interleave=False
    )
    dialect_b = ScriptedTurn(
        blocks=logical_blocks(9), stop_reason="tool_use", interleave=True
    )
    out_a = _drain(dialect_a).model_dump(mode="json")
    out_b = _drain(dialect_b).model_dump(mode="json")
    assert out_a == out_b


def test_consumer_sees_only_canonical_fields() -> None:
    turn = ScriptedTurn(
        blocks=(
            ScriptedThinking(text="t"),
            ScriptedText(text="u"),
            ScriptedToolUse(id="c1", name="n", input={"a": 1}, intent="do a"),
        ),
        stop_reason="tool_use",
    )
    dumped = _drain(turn).model_dump(mode="json")
    # Top-level AssistantTurn keys are exactly the canonical set.
    assert set(dumped) == {"role", "content", "stop_reason", "usage"}
    # Each content block carries ONLY its canonical union keys — no dialect leak.
    for block in dumped["content"]:
        assert set(block) == _ALLOWED_KEYS[block["type"]]


def test_same_consumer_code_handles_both_providers() -> None:
    # The "interpreter" pattern-matches on canonical block `type` alone; both
    # providers feed it the identical control flow.
    def summarize(turn: AssistantTurn) -> list[str]:
        out: list[str] = []
        for block in turn.content:
            if block.type == "tool_use":
                out.append(f"tool:{block.name}")
            elif block.type == "text":
                out.append("text")
            elif block.type == "thinking":
                out.append("thinking")
        return out

    a = _drain(
        ScriptedTurn(
            blocks=(ScriptedToolUse(id="c1", name="alpha", input={}, intent="a"),),
            stop_reason="tool_use",
            interleave=False,
        )
    )
    b = _drain(
        ScriptedTurn(
            blocks=(ScriptedToolUse(id="c1", name="alpha", input={}, intent="a"),),
            stop_reason="tool_use",
            interleave=True,
        )
    )
    assert summarize(a) == summarize(b) == ["tool:alpha"]
