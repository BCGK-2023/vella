"""Streaming ≡ non-streaming: the deterministic accumulator under adversarial fuzz.

Pre-mortem #1 in tests. The drained event stream must assemble byte-identically to
the non-streaming wrapper, including multi-tool-call turns whose fragmented
``input_json_delta`` chunks arrive interleaved. This file proves:

* a hand-built event list assembles to the expected canonical turn;
* the async ``MockProvider.turn`` drain ≡ the synchronous ``assemble`` of the same
  events (the non-streaming wrapper claim);
* multi-tool interleaved assembly ≡ sequential assembly (index-keyed, not
  arrival-keyed) — the load-bearing determinism property;
* a **Hypothesis** sweep over arbitrary ``partial_json`` fragmentations and random
  interleavings always yields the same assembled turn;
* a **subprocess** determinism check across ``PYTHONHASHSEED ∈ {0,1,42}``.

No ``pytest-asyncio``: async paths run via ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from vella.agent import (
    MockProvider,
    ScriptedText,
    ScriptedThinking,
    ScriptedToolUse,
    ScriptedTurn,
    TurnRequest,
)
from vella.agent._assembler import assemble
from vella.agent.mock_provider import _turn_events

_DETERMINISM_HELPER = (
    Path(__file__).resolve().parent / "_assembly_determinism_helper.py"
)


def _drain_turn(turn: ScriptedTurn) -> dict[str, object]:
    provider = MockProvider([turn])
    assembled = asyncio.run(provider.turn(TurnRequest()))
    return assembled.model_dump(mode="json")


def test_drain_equals_sync_assemble() -> None:
    turn = ScriptedTurn(
        blocks=(
            ScriptedThinking(text="thinking hard", fragments=4),
            ScriptedText(text="hello there", fragments=3),
            ScriptedToolUse(
                id="t1",
                name="search",
                input={"q": "a b c", "n": 5},
                intent="search",
                fragments=9,
            ),
        ),
        stop_reason="tool_use",
    )
    sync = assemble(_turn_events(turn)).model_dump(mode="json")
    drained = _drain_turn(turn)
    assert sync == drained


def test_multi_tool_interleave_equals_sequential() -> None:
    blocks = (
        ScriptedToolUse(
            id="t1", name="alpha", input={"x": 1, "y": "two"}, intent="a", fragments=6
        ),
        ScriptedText(text="between the calls", fragments=4),
        ScriptedToolUse(
            id="t2", name="beta", input={"deep": {"nested": [1, 2, 3]}}, intent="b", fragments=8
        ),
    )
    seq = assemble(
        _turn_events(ScriptedTurn(blocks=blocks, stop_reason="tool_use"))
    ).model_dump(mode="json")
    inter = assemble(
        _turn_events(
            ScriptedTurn(blocks=blocks, stop_reason="tool_use", interleave=True)
        )
    ).model_dump(mode="json")
    assert seq == inter
    # The two tool inputs survived assembly intact and in order.
    tool_blocks = [b for b in seq["content"] if b["type"] == "tool_use"]
    assert [b["name"] for b in tool_blocks] == ["alpha", "beta"]
    assert tool_blocks[0]["input"] == {"x": 1, "y": "two"}
    assert tool_blocks[1]["input"] == {"deep": {"nested": [1, 2, 3]}}


# --- Hypothesis: arbitrary fragmentation + interleaving => same assembled turn. ---

_json_scalars = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.booleans(),
    st.text(max_size=8),
)
_tool_inputs = st.dictionaries(
    keys=st.text(min_size=1, max_size=5),
    values=_json_scalars,
    max_size=4,
)


@settings(max_examples=120, deadline=None)
@given(
    inputs=st.lists(_tool_inputs, min_size=1, max_size=3),
    frags=st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=3),
    interleave=st.booleans(),
)
def test_fragmentation_invariance(
    inputs: list[dict[str, object]], frags: list[int], interleave: bool
) -> None:
    # Build a multi-tool turn; vary fragment counts and interleaving. The assembled
    # turn must equal the canonical reference (one fragment, no interleave) regardless.
    blocks = tuple(
        ScriptedToolUse(
            id=f"t{i}",
            name=f"tool{i}",
            input=inp,
            intent=f"call {i}",
            fragments=frags[i % len(frags)],
        )
        for i, inp in enumerate(inputs)
    )
    reference_blocks = tuple(
        ScriptedToolUse(id=f"t{i}", name=f"tool{i}", input=inp, intent=f"call {i}")
        for i, inp in enumerate(inputs)
    )
    fuzzed = assemble(
        _turn_events(
            ScriptedTurn(blocks=blocks, stop_reason="tool_use", interleave=interleave)
        )
    ).model_dump(mode="json")
    reference = assemble(
        _turn_events(ScriptedTurn(blocks=reference_blocks, stop_reason="tool_use"))
    ).model_dump(mode="json")
    assert fuzzed == reference


def test_tool_json_is_parsed_only_at_block_stop() -> None:
    # A single-fragment-per-byte split of valid JSON still parses (proving the parse
    # happens on the FULL concatenation at stop, never on a partial fragment).
    payload = {"alpha": "beta", "count": 42, "nested": {"k": [1, 2]}}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    turn = ScriptedTurn(
        blocks=(
            ScriptedToolUse(
                id="t1", name="x", input=payload, intent="x", fragments=len(raw)
            ),
        ),
        stop_reason="tool_use",
    )
    out = assemble(_turn_events(turn)).model_dump(mode="json")
    assert out["content"][0]["input"] == payload


def _run_helper_under_seed(seed: str) -> bytes:
    result = subprocess.run(
        [sys.executable, str(_DETERMINISM_HELPER)],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_assembly_is_hash_seed_independent() -> None:
    out_0 = _run_helper_under_seed("0")
    out_1 = _run_helper_under_seed("1")
    out_2 = _run_helper_under_seed("42")
    assert out_0 == out_1 == out_2
    assert out_0
