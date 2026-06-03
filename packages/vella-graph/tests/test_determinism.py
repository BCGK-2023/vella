"""Determinism gate (M3, pre-mortem (d)): the named topology artifact.

The graph's NAMED determinism artifact is the **sorted topology projection** —
the sorted endpoint-id set per ``(tenant, node, direction, edge_type)`` — serialized
byte-identically across ``PYTHONHASHSEED`` values (mirrors core's "set-derived
serialized output is ``sorted()``" discipline). The :mod:`tests._determinism_helper`
script folds a fixed multi-tenant graph, serializes the sorted projection to
canonical JSON, and prints it to stdout. This test runs that helper as a SUBPROCESS
under three hash seeds and asserts byte-identical output.

The mechanism MUST be a subprocess: ``PYTHONHASHSEED`` is read once at interpreter
startup, so an in-process re-import does NOT reset the hash seed. Only a fresh
interpreter (via ``subprocess.run``) can vary it.

Non-vacuity (what the verifier mutates): removing the helper's per-bucket
``sorted()`` (NOT ``sort_keys``) makes the endpoint order follow hash-driven set
iteration over the multi-endpoint fan-out bucket (``t-gamma`` ``d -KNOWS-> {2,e,3}``),
so the three seeds DIVERGE — this test then fails. With the ``sorted()`` in place the
three seeds are identical.
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


def test_topology_artifact_is_hash_seed_independent() -> None:
    """The sorted topology artifact is byte-identical across hash seeds {0,1,42}."""
    out_0 = _run_under_seed("0")
    out_1 = _run_under_seed("1")
    out_2 = _run_under_seed("42")
    assert out_0 == out_1 == out_2
    assert out_0  # non-empty — the helper actually produced a fixture


def test_topology_artifact_is_non_vacuous() -> None:
    """Sanity: the artifact spans multiple tenants and edge types (not a single set)."""
    out = _run_under_seed("0")
    # >= 3 distinct tenants present (so the projection is genuinely multi-tenant).
    assert b'"t-alpha"' in out
    assert b'"t-beta"' in out
    assert b'"t-gamma"' in out
    # Multiple edge types present (multi-bucket topology, not one trivial set).
    assert b'"knows"' in out
    assert b'"owned_by"' in out
    assert b'"part_of"' in out
    # The multi-endpoint fan-out bucket: the load-bearing >= 2-element endpoint set.
    assert b'"endpoints"' in out
