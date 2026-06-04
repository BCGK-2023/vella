"""Subprocess fixture for the M5 loop_policy determinism artifact.

Run as a script under a fixed ``PYTHONHASHSEED`` (set by the parent test). It builds a
:class:`~vella.agent.LoopPolicy` carrying the package's two set-derived serialized
fields — ``stop_conditions`` (a stop-condition set) and a ``restricted`` tool choice's
``types`` (a name set) — supplied in DELIBERATELY UNSORTED order, then dumps the
policy via ``model_dump(mode="json")`` as canonical, byte-stable JSON to stdout.

The load-bearing point: those fields are ``sorted()`` at the policy's validation
boundary, so the serialized bytes are identical regardless of the process hash seed.
A removed ``sorted()`` would let construction order (and, for a real set, hash order)
leak into the artifact — this subprocess test across ``PYTHONHASHSEED {0,1,42}`` is
what would catch it. This is a script (invoked via ``subprocess.run``), not a test
module — only a fresh interpreter can vary the hash seed.
"""

from __future__ import annotations

import json

from vella.agent import LoopPolicy, ToolChoiceRestricted


def build_artifact() -> str:
    """Serialize a set-derived loop policy to canonical, byte-stable JSON."""
    policy = LoopPolicy(
        # Supplied unsorted on purpose — the validator sorts these set-derived fields.
        stop_conditions=("refusal", "no_tool_calls", "max_tokens", "explicit_stop_node"),
        tool_choice=ToolChoiceRestricted(types=("delta", "alpha", "charlie", "bravo")),
        step_budget=100,
        token_budget=1000,
        compaction={"compaction_threshold": 500, "pin": ("user", "system"), "recall_depth": 2},
    )
    return json.dumps(policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def main() -> None:
    """Print the loop-policy artifact as canonical, byte-stable JSON to stdout."""
    print(build_artifact(), end="")


if __name__ == "__main__":
    main()
