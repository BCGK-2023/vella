"""Determinism gate (M5): the loop_policy set-derived artifact is hash-seed stable.

The M5 ``loop_policy`` schema carries two set-derived serialized fields —
``stop_conditions`` and a ``restricted`` choice's ``types`` — each ``sorted()`` at the
policy's validation boundary. The :mod:`tests._policy_determinism_helper` script builds
a policy with those fields supplied UNSORTED and dumps its canonical JSON. This test
runs that helper as a SUBPROCESS under three hash seeds and asserts byte-identical
output, so a removed ``sorted()`` (which would let construction/hash order leak into
the artifact) turns the gate red.

The mechanism MUST be a subprocess: ``PYTHONHASHSEED`` is read once at interpreter
startup, so an in-process re-import does NOT reset it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HELPER = Path(__file__).resolve().parent / "_policy_determinism_helper.py"


def _run_under_seed(seed: str) -> bytes:
    result = subprocess.run(
        [sys.executable, str(_HELPER)],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_loop_policy_artifact_is_hash_seed_independent() -> None:
    """The loop_policy artifact is byte-identical across hash seeds {0,1,42}."""
    out_0 = _run_under_seed("0")
    out_1 = _run_under_seed("1")
    out_2 = _run_under_seed("42")
    assert out_0 == out_1 == out_2
    assert out_0  # non-empty


def test_loop_policy_artifact_is_sorted() -> None:
    """The set-derived fields are emitted in sorted order in the artifact bytes."""
    out = _run_under_seed("0").decode()
    # stop_conditions sorted: explicit_stop_node < max_tokens < no_tool_calls < refusal
    assert (
        '["explicit_stop_node","max_tokens","no_tool_calls","refusal"]' in out
    )
    # restricted types sorted: alpha < bravo < charlie < delta
    assert '["alpha","bravo","charlie","delta"]' in out
