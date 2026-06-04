"""Determinism gate (M0 baseline): the named surface artifact.

The agent's NAMED determinism artifact at M0 is the **canonical public-surface
snapshot** — the same structure the surface tripwire freezes — serialized
byte-identically across ``PYTHONHASHSEED`` values (mirrors core's "set-derived
serialized output is ``sorted()``" discipline and the graph's topology artifact).
The :mod:`tests._determinism_helper` script builds that snapshot from the live
package and prints its canonical JSON to stdout. This test runs that helper as a
SUBPROCESS under three hash seeds and asserts byte-identical output.

The mechanism MUST be a subprocess: ``PYTHONHASHSEED`` is read once at interpreter
startup, so an in-process re-import does NOT reset the hash seed. Only a fresh
interpreter (via ``subprocess.run``) can vary it.

This is real-but-trivial at M0 (``__all__`` is empty, so the snapshot is the
empty-surface structure), but the artifact is produced from the live package and
dumped with ``sort_keys=True`` — so as ``__all__`` grows, the sorted export list,
error-MRO lists, and ``Literal`` value sets it carries are genuinely set-derived,
and this same test tightens to guard their hash-seed independence for free.
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


def test_surface_artifact_is_hash_seed_independent() -> None:
    """The canonical surface artifact is byte-identical across hash seeds {0,1,42}."""
    out_0 = _run_under_seed("0")
    out_1 = _run_under_seed("1")
    out_2 = _run_under_seed("42")
    assert out_0 == out_1 == out_2
    assert out_0  # non-empty — the helper actually produced an artifact


def test_surface_artifact_is_the_canonical_snapshot() -> None:
    """Sanity: the artifact is the canonical surface structure (all five sections)."""
    out = _run_under_seed("0")
    # The artifact carries the full surface structure even while empty at M0.
    assert b'"__all__"' in out
    assert b'"errors"' in out
    assert b'"models"' in out
    assert b'"literals"' in out
    assert b'"verbs"' in out
