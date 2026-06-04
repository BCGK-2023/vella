"""Determinism gate (M6, M8 hardening): the sub-agent run-tree digest is seed-stable.

The M6 verifier flagged a gap: the bounded sub-agent run-tree was proven cardinality-
and cost-bounded, but its STRUCTURAL DIGEST was not asserted hash-seed independent. This
closes it. The :mod:`tests._subagent_determinism_helper` script drives an adversarial
spawn-every-turn run bounded by ``max_depth``/``max_fanout``, folds the materialized
run-tree from the graph, and emits its structural shape (each run's sorted
``[parent_depth, direct_child_count]``) as canonical JSON. This test runs that helper as
a SUBPROCESS under three hash seeds and asserts byte-identical output — so a missing
``sorted()`` in the digest, or a ``direction="both"`` depth walk that miscounts, turns
the gate red.

The mechanism MUST be a subprocess: ``PYTHONHASHSEED`` is read once at interpreter
startup, so an in-process re-import does NOT reset it; only a fresh interpreter can vary
it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HELPER = Path(__file__).resolve().parent / "_subagent_determinism_helper.py"


def _run_under_seed(seed: str) -> bytes:
    result = subprocess.run(
        [sys.executable, str(_HELPER)],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_run_tree_digest_is_hash_seed_independent() -> None:
    """The sub-agent run-tree structural digest is byte-identical across {0,1,42}."""
    out_0 = _run_under_seed("0")
    out_1 = _run_under_seed("1")
    out_2 = _run_under_seed("42")
    assert out_0 == out_1 == out_2
    assert out_0  # non-empty — a real tree materialized


def test_run_tree_digest_respects_the_closed_form_bound() -> None:
    """The materialized tree never exceeds N_max = Σ fⁱ for (max_depth=2, max_fanout=2)."""
    from vella.agent import max_run_tree_size

    out = _run_under_seed("0").decode()
    # The digest is a JSON array of [depth, child_count] pairs, one per run.
    import json

    shape = json.loads(out)
    assert len(shape) <= max_run_tree_size(2, 2)  # 1 + 2 + 4 = 7
    assert len(shape) > 1  # spawning actually happened
    # No run is deeper than max_depth, and no run exceeds max_fanout direct children.
    assert all(depth <= 2 and children <= 2 for depth, children in shape)
