"""Resolve a :class:`~vella.agent.ToolResult` to a legibility hint (M3).

The harness turns a tool-node's :class:`~vella.agent.ToolHints` + an invocation's
:class:`~vella.agent.ToolResult` into the single hint string the model reads. The
resolution rule (locked, spec "Hint resolution"):

* **success** (``not result.is_error``) -> ``hints.result_hint``;
* **error** -> the FIRST :class:`~vella.agent.ErrorHint` whose ``match`` equals the
  result's ``error_kind`` (order-preserving, first-match-wins — ``error_hints`` is a
  tuple precisely because its order is the contract), else
  ``hints.default_error_hint``.

The resolved hint goes onto the :class:`~vella.agent.ToolResultBlock` fed back to the
model AND is recorded on the durable ``agent.tool_call`` node, so a UI renders it from
replay/observe, not only live.
"""

from __future__ import annotations

from typing import Optional

from .tool import ToolHints, ToolResult


def resolve_hint(hints: ToolHints, result: ToolResult) -> Optional[str]:
    """Resolve the hint for ``result`` against a tool-node's ``hints``.

    Args:
        hints: The tool-node's :class:`~vella.agent.ToolHints`.
        result: The invocation's :class:`~vella.agent.ToolResult` (its ``is_error``
            selects success vs error; its ``error_kind`` keys the error lookup).

    Returns:
        The resolved hint string, or ``None`` when none applies (no ``result_hint``
        on success; no matching ``error_hints`` entry and no ``default_error_hint``
        on error).
    """
    if not result.is_error:
        return hints.result_hint
    # Error path: first matching entry wins (order is semantic — never sorted).
    for entry in hints.error_hints:
        if entry.match == result.error_kind:
            return entry.hint
    return hints.default_error_hint
