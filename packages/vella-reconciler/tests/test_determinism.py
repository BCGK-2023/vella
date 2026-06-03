"""Determinism gate (M6, must-fix 4 / C3): the named dead-letter artifact.

The reconciler's NAMED determinism artifact is the **sorted dead-letter record
set** serialized byte-identically across ``PYTHONHASHSEED`` values (mirrors core's
"set-derived serialized output is ``sorted()``" discipline). The
:mod:`tests._determinism_helper` script drives a fixed give-up scenario that
dead-letters several entities across DISTINCT tenants/ids, serializes the sorted
record set to canonical JSON, and prints it to stdout. This test runs that helper
as a SUBPROCESS under three hash seeds and asserts byte-identical output.

The mechanism MUST be a subprocess: ``PYTHONHASHSEED`` is read once at interpreter
startup, so an in-process re-import does NOT reset the hash seed. Only a fresh
interpreter (via ``subprocess.run``) can vary it.

Non-vacuity (what the verifier mutates): removing the helper's ``sorted()`` (NOT
``sort_keys``) makes the bytes depend on dict/hash iteration order over the
heterogeneous ``(tenant_id, entity_id)`` keys, so the three seeds DIVERGE — this
test then fails. With the ``sorted()`` in place the three seeds are identical.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HELPER = Path(__file__).resolve().parent / "_determinism_helper.py"


def _run_under_seed(seed: str) -> bytes:
    result = subprocess.run(
        [sys.executable, str(_HELPER)],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_deadletter_artifact_is_hash_seed_independent() -> None:
    """The sorted dead-letter artifact is byte-identical across hash seeds."""
    out_0 = _run_under_seed("0")
    out_1 = _run_under_seed("1")
    out_2 = _run_under_seed("42")
    assert out_0 == out_1 == out_2
    assert out_0  # non-empty — the helper actually produced a fixture

    # Sanity: the artifact really dead-lettered multiple entities across tenants
    # (so the ordering it sorts is genuinely hash-seed sensitive — not a vacuous
    # single-record set). Distinct tenant ids and a give-up reason must appear.
    assert b'"t-alpha"' in out_0
    assert b'"t-beta"' in out_0
    assert b'"t-gamma"' in out_0
    assert b'"reason"' in out_0
    assert b'"attempts"' in out_0
