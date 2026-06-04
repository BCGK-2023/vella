"""Subprocess artifact: a multi-tool assembled turn, dumped as canonical JSON.

Run as a subprocess by ``test_streaming_assembly.py`` under several
``PYTHONHASHSEED`` values; its stdout must be byte-identical across seeds. The
artifact is a genuinely set-derived surface — a multi-tool, interleaved-delta turn
whose tool ``input`` dicts and block ordering exercise the accumulator's
index-keyed, sorted assembly — so any hash-order leak would change the bytes.

``PYTHONHASHSEED`` is read once at interpreter startup, so this MUST be a fresh
process per seed (an in-process re-import would not vary it).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vella.agent import (  # noqa: E402
    ScriptedText,
    ScriptedThinking,
    ScriptedToolUse,
    ScriptedTurn,
    Usage,
)
from vella.agent._assembler import assemble  # noqa: E402
from vella.agent.mock_provider import _turn_events  # noqa: E402


def _artifact() -> str:
    turn = ScriptedTurn(
        blocks=(
            ScriptedThinking(text="weighing the options carefully", fragments=5),
            ScriptedText(text="I will call two tools", fragments=3),
            ScriptedToolUse(
                id="call-a",
                name="search",
                input={"query": "vella graph", "limit": 10, "flags": {"x": True}},
                intent="search the corpus",
                fragments=11,
            ),
            ScriptedToolUse(
                id="call-b",
                name="fetch",
                input={"url": "https://example/x", "headers": {"a": "1", "b": "2"}},
                intent="fetch the page",
                fragments=7,
            ),
        ),
        stop_reason="tool_use",
        usage=Usage(input_tokens=11, output_tokens=22, reasoning_tokens=5),
        interleave=True,
    )
    assembled = assemble(_turn_events(turn))
    return json.dumps(assembled.model_dump(mode="json"), sort_keys=True)


if __name__ == "__main__":
    sys.stdout.write(_artifact())
