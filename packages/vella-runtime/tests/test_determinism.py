"""Determinism gate: serialized output is hash-seed-independent.

Runtime serializes set-derived values via ``sorted()`` so reproducible
artifacts never depend on nondeterministic hash iteration order. This locks
that invariant by driving a full fixed verb sequence (create / edit /
set_desired / upsert / delete / telemetry) through a ``Runtime``, serializing
the resulting state-table + log to canonical JSON, and running it under three
different ``PYTHONHASHSEED`` values — asserting byte-identical stdout. The
fixture pins ids/timestamps/created_by to fixed values, so the ONLY variable
across runs is hash-seed-driven iteration order.

The mechanism MUST be a subprocess: ``PYTHONHASHSEED`` is read once at
interpreter startup, so an in-process re-import does NOT reset the hash seed.
Only a fresh interpreter (via ``subprocess.run``) can vary it.
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


def test_serialization_is_hash_seed_independent() -> None:
    out_0 = _run_under_seed("0")
    out_1 = _run_under_seed("1")
    out_2 = _run_under_seed("42")
    assert out_0 == out_1 == out_2
    assert out_0  # non-empty — the helper actually produced a fixture
    # Sanity: the fixture really exercised the full verb set (table + log).
    assert b'"state_table"' in out_0 and b'"log"' in out_0
