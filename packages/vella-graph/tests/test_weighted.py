"""Weighted shortest path (M4): baked weights, canonical tie-break, override mode-gating.

Covers the six load-bearing weighted-SP claims:

* **Baked weight is honoured (not hop-count).** A graph with a 1-hop high-cost
  direct edge and a 2-hop low-cost path: weighted SP must return the CHEAPER 2-hop
  path, not the shorter 1-hop path. ``mut-m4-ignore-weight`` (Dijkstra treats all
  weights as 1.0) would pick the 1-hop path → RED.
* **The heap tie-break is node-id canonical, opposing insertion order.** A diamond
  where the lex-LARGER branch (``TIE_HIGH``) is linked FIRST — opposite of canonical
  order. Equal-cost paths; the engine must still return the path through
  ``TIE_LOW`` (lex-smaller). ``mut-m4-tiebreak`` (replace ``str(node_id)`` with an
  insertion counter) pops ``TIE_HIGH`` first → returns the wrong path → RED.
* **Baked weighted SP is mode-equivalent.** Byte-identical result full vs lean.
* **A per-query override (full mode) reorders by the override's weights**, producing
  a DIFFERENT path than the baked weights (proving the override is applied, not the
  baked floats — ``mut-m4-override-reads-baked``).
* **A per-query override in lean mode RAISES** ``WeightOverrideRequiresFullMode``
  (no silent fallback — ``mut-m4-lean-override-allowed``).

Pinned UUIDs used throughout so every canonical-path assertion is exact and
deterministic across hash seeds.

Gap-A fixture (``test_baked_weight_beats_hop_count``):
    Nodes: WS (start), WM (mid), WT (target).
    Edges:  WS -CONTAINS-> WT  baked 10.0  (direct, expensive)
            WS -KNOWS->    WM  baked  1.0  }  2-hop cheap path
            WM -KNOWS->    WT  baked  1.0  }  total 2.0
    Weighted SP must choose (WS, WM, WT). Mutation (all weights=1.0): picks (WS, WT).

Gap-B fixture (``test_tiebreak_canonical_high_id_linked_first``):
    Nodes: TS (start), TIE_LOW, TIE_HIGH (middle), TT (target).
    ``str(TIE_LOW) < str(TIE_HIGH)``  — asserted.
    Edges: TS -KNOWS-> TIE_HIGH  linked FIRST  (insertion-order-first = wrong id)
           TS -KNOWS-> TIE_LOW   linked SECOND
           TIE_HIGH -KNOWS-> TT
           TIE_LOW  -KNOWS-> TT
    Both paths cost 2.0; canonical answer is (TS, TIE_LOW, TT).
    Mutation (counter tie-break): TIE_HIGH pops first → returns (TS, TIE_HIGH, TT) → RED.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from vella.core import Edge, EdgeTypes
from vella.runtime import Runtime

from vella.graph import GraphProjection, GraphView, WeightOverrideRequiresFullMode

from _fixtures import drive, make_node, thing_registry

# ---------------------------------------------------------------------------
# Gap-A: baked-weight-beats-hop-count fixture ids
# str order: WS < WM < WT
# ---------------------------------------------------------------------------
WS = UUID("10000000-0000-0000-0000-000000000000")  # start
WM = UUID("20000000-0000-0000-0000-000000000000")  # mid (cheap 2-hop path)
WT = UUID("30000000-0000-0000-0000-000000000000")  # target

# ---------------------------------------------------------------------------
# Gap-B: tie-break fixture ids — TIE_HIGH linked first, but TIE_LOW is canonical
# ---------------------------------------------------------------------------
TS      = UUID("aa000000-0000-0000-0000-000000000000")  # start
TIE_LOW  = UUID("bb000000-0000-0000-0000-000000000000")  # lex-smaller branch (linked 2nd)
TIE_HIGH = UUID("cc000000-0000-0000-0000-000000000000")  # lex-larger  branch (linked 1st)
TT      = UUID("dd000000-0000-0000-0000-000000000000")  # target

# ---------------------------------------------------------------------------
# Original diamond fixture ids (for override / lean tests)
# str order: N1 < N2 < N3 < N4
# ---------------------------------------------------------------------------
N1 = UUID("11111111-1111-1111-1111-111111111111")
N2 = UUID("22222222-2222-2222-2222-222222222222")
N3 = UUID("33333333-3333-3333-3333-333333333333")
N4 = UUID("44444444-4444-4444-4444-444444444444")

_TENANT = "t"


# ---------------------------------------------------------------------------
# Weight callables for the Gap-A fixture
# ---------------------------------------------------------------------------

def _nonuniform_baked_weight(edge: "Edge[Any, Any]") -> float:
    """Non-uniform baked weight: CONTAINS costs 10.0, every other edge costs 1.0.

    Used by the Gap-A fixture so the direct 1-hop CONTAINS edge (WS->WT, cost 10.0)
    loses to the 2-hop KNOWS path (WS->WM->WT, cost 1.0+1.0=2.0). A Dijkstra that
    ignores these weights and treats every edge as cost 1.0 (mut-m4-ignore-weight)
    would pick the 1-hop path instead — proving the baked weight is read, not
    fabricated.
    """
    return 10.0 if edge.type == EdgeTypes.CONTAINS else 1.0


# ---------------------------------------------------------------------------
# Weight callable for the override tests (original diamond)
# ---------------------------------------------------------------------------

def _uniform_baked_weight(edge: "Edge[Any, Any]") -> float:
    """Bake every edge at cost 1.0 — both N1->N4 branches are equal-cost."""
    return 1.0


def _override_weight(edge: "Edge[Any, Any]") -> float:
    """A per-query override making the OWNED_BY branch (via N3) strictly cheaper.

    Baked weights are all 1.0, so the canonical (lex-smallest) baked path is
    N1-N2-N4. This override prices OWNED_BY edges at 0.1 and KNOWS at 1.0, so the
    minimum-weight path becomes N1-N3-N4 (via the two OWNED_BY edges) — a DIFFERENT
    path than the baked answer, proving the override is the weight actually used.
    """
    return 0.1 if edge.type == EdgeTypes.OWNED_BY else 1.0


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

async def _build_nonuniform(mode: str) -> GraphView:
    """Fold WS->{direct CONTAINS 10.0, 2-hop KNOWS 1.0+1.0}->WT.

    The minimum-weight path is the 2-hop (WS, WM, WT) at cost 2.0, not the
    direct (WS, WT) at cost 10.0.  A hop-count Dijkstra (mut-m4-ignore-weight)
    would incorrectly prefer the 1-hop direct path.
    """
    rt = Runtime()
    thing = thing_registry()
    for nid in (WS, WM, WT):
        await rt.create(make_node(thing, tenant_id=_TENANT, node_id=nid))
    # Direct high-cost edge: WS -CONTAINS-> WT  (baked 10.0)
    await rt.link(_TENANT, WS, WT, EdgeTypes.CONTAINS)
    # Cheap 2-hop path: WS -KNOWS-> WM -KNOWS-> WT  (baked 1.0 + 1.0)
    await rt.link(_TENANT, WS, WM, EdgeTypes.KNOWS)
    await rt.link(_TENANT, WM, WT, EdgeTypes.KNOWS)
    return await GraphProjection().fold(
        rt, _TENANT, mode=mode, weight=_nonuniform_baked_weight  # type: ignore[arg-type]
    )


async def _build_tiebreak(mode: str) -> GraphView:
    """Fold a diamond where the lex-LARGER branch is linked first.

    TS -KNOWS-> TIE_HIGH  (linked first — insertion-order-first)
    TS -KNOWS-> TIE_LOW   (linked second — insertion-order-second)
    TIE_HIGH -KNOWS-> TT
    TIE_LOW  -KNOWS-> TT

    All edges cost 1.0 (baked). Both paths cost 2.0. The canonical answer is
    (TS, TIE_LOW, TT) — the lex-smaller branch wins the tie-break.

    A counter-based tie-break (mut-m4-tiebreak) would pop TIE_HIGH first (it was
    pushed with a smaller counter), returning the wrong (TS, TIE_HIGH, TT) path.
    """
    rt = Runtime()
    thing = thing_registry()
    for nid in (TS, TIE_HIGH, TIE_LOW, TT):
        await rt.create(make_node(thing, tenant_id=_TENANT, node_id=nid))
    # Link TIE_HIGH FIRST — so without the str(node_id) tie-break, it pops first.
    await rt.link(_TENANT, TS, TIE_HIGH, EdgeTypes.KNOWS)
    await rt.link(_TENANT, TS, TIE_LOW, EdgeTypes.KNOWS)
    await rt.link(_TENANT, TIE_HIGH, TT, EdgeTypes.KNOWS)
    await rt.link(_TENANT, TIE_LOW, TT, EdgeTypes.KNOWS)
    return await GraphProjection().fold(
        rt, _TENANT, mode=mode, weight=lambda e: 1.0  # type: ignore[arg-type]
    )


async def _build_diamond(mode: str) -> GraphView:
    """Fold the diamond N1->{N2,N3}->N4 with all-uniform baked weights.

    Used for the override and lean-raises tests.
    """
    rt = Runtime()
    thing = thing_registry()
    for nid in (N1, N2, N3, N4):
        await rt.create(make_node(thing, tenant_id=_TENANT, node_id=nid))
    await rt.link(_TENANT, N1, N2, EdgeTypes.KNOWS)
    await rt.link(_TENANT, N2, N4, EdgeTypes.KNOWS)
    await rt.link(_TENANT, N1, N3, EdgeTypes.OWNED_BY)
    await rt.link(_TENANT, N3, N4, EdgeTypes.OWNED_BY)
    return await GraphProjection().fold(
        rt, _TENANT, mode=mode, weight=_uniform_baked_weight  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_baked_weight_beats_hop_count() -> None:
    """Baked weighted SP follows minimum WEIGHT, not minimum hop count.

    The direct 1-hop path WS->WT is baked at cost 10.0; the 2-hop path
    WS->WM->WT is baked at cost 2.0.  Dijkstra must return the 2-hop path.

    Non-vacuity (mut-m4-ignore-weight): if Dijkstra ignores EdgeRecord.weight and
    treats every edge as cost 1.0 it picks the 1-hop path (WS, WT) — RED.  Only
    by reading the baked weights does it choose (WS, WM, WT).

    Run in both modes to confirm baked-weight SP is mode-equivalent.
    """
    drive(_baked_weight_beats_hop_count_case())


async def _baked_weight_beats_hop_count_case() -> None:
    full = await _build_nonuniform("full")
    lean = await _build_nonuniform("lean")

    for view, label in ((full, "full"), (lean, "lean")):
        path = await view.weighted_shortest_path(WS, WT, direction="out")
        assert path is not None, f"{label}: path should exist"
        assert path.nodes == (WS, WM, WT), (
            f"{label}: expected cheap 2-hop (WS, WM, WT) cost=2.0, "
            f"got {path.nodes} — if this returns (WS, WT) the baked weight is "
            "being ignored (mut-m4-ignore-weight)"
        )

    # Mode-equivalence: baked result identical full vs lean.
    fp = await full.weighted_shortest_path(WS, WT, direction="out")
    lp = await lean.weighted_shortest_path(WS, WT, direction="out")
    assert fp is not None and lp is not None
    assert fp.nodes == lp.nodes


def test_tiebreak_canonical_high_id_linked_first() -> None:
    """Heap tie-break is node-id canonical even when the lex-larger branch is linked first.

    The diamond has two equal-cost paths: TS-TIE_HIGH-TT and TS-TIE_LOW-TT.
    TIE_HIGH is linked BEFORE TIE_LOW — so without the ``str(node_id)`` tie-break
    key, heap insertion order would pop TIE_HIGH first (lower counter), returning
    the wrong (TS, TIE_HIGH, TT) path.

    Non-vacuity (mut-m4-tiebreak): replacing ``str(node_id)`` with an insertion
    counter makes TIE_HIGH pop first → this assertion goes RED.

    Constants (all asserted at runtime):
        str(TIE_LOW) < str(TIE_HIGH)  (TIE_LOW is the canonical choice)
        TIE_HIGH linked before TIE_LOW (insertion order opposes canonical order)
    """
    drive(_tiebreak_high_id_first_case())


async def _tiebreak_high_id_first_case() -> None:
    # Self-documenting invariant: TIE_LOW must sort before TIE_HIGH.
    assert str(TIE_LOW) < str(TIE_HIGH), (
        f"TIE_LOW must have lower str() than TIE_HIGH; "
        f"got str(TIE_LOW)={str(TIE_LOW)!r}, str(TIE_HIGH)={str(TIE_HIGH)!r}"
    )

    full = await _build_tiebreak("full")
    lean = await _build_tiebreak("lean")

    for view, label in ((full, "full"), (lean, "lean")):
        path = await view.weighted_shortest_path(TS, TT, direction="out")
        assert path is not None, f"{label}: path should exist"
        assert path.nodes == (TS, TIE_LOW, TT), (
            f"{label}: expected lex-canonical path (TS, TIE_LOW, TT), "
            f"got {path.nodes}. "
            "If this returns (TS, TIE_HIGH, TT), the str(node_id) tie-break was "
            "dropped and insertion order is deciding (mut-m4-tiebreak)."
        )

    # Mode-equivalence.
    fp = await full.weighted_shortest_path(TS, TT, direction="out")
    lp = await lean.weighted_shortest_path(TS, TT, direction="out")
    assert fp is not None and lp is not None
    assert fp.nodes == lp.nodes == (TS, TIE_LOW, TT)


def test_baked_weighted_sp_mode_equivalent() -> None:
    """Baked weighted SP is byte-identical full vs lean (sorted-id mode-equivalence)."""
    drive(_baked_equivalence_case())


async def _baked_equivalence_case() -> None:
    full = await _build_diamond("full")
    lean = await _build_diamond("lean")
    for src, dst in ((N1, N4), (N1, N2), (N2, N4), (N3, N4), (N4, N1)):
        fp = await full.weighted_shortest_path(src, dst, direction="out")
        lp = await lean.weighted_shortest_path(src, dst, direction="out")
        assert (fp.nodes if fp else None) == (lp.nodes if lp else None)
    # The headline diamond: identical canonical path in both modes.
    full_path = await full.weighted_shortest_path(N1, N4, direction="out")
    lean_path = await lean.weighted_shortest_path(N1, N4, direction="out")
    assert full_path is not None and lean_path is not None
    assert full_path.nodes == lean_path.nodes == (N1, N2, N4)


def test_override_full_mode_reorders_to_override_path() -> None:
    """A full-mode override uses the override's weights, flipping the baked path."""
    drive(_override_full_case())


async def _override_full_case() -> None:
    view = await _build_diamond("full")
    # Baked answer is N1-N2-N4 (both branches cost 2.0; lex tie-break).
    baked = await view.weighted_shortest_path(N1, N4, direction="out")
    assert baked is not None and baked.nodes == (N1, N2, N4)
    # Override prices OWNED_BY at 0.1 -> N1-N3-N4 (cost 0.2) beats N1-N2-N4 (cost 2.0).
    overridden = await view.weighted_shortest_path(
        N1, N4, direction="out", weight=_override_weight
    )
    assert overridden is not None
    assert overridden.nodes == (N1, N3, N4), (
        "override must re-read bodies and use the OVERRIDE weights (N1-N3-N4); if "
        "this returns the baked path N1-N2-N4 the override was ignored "
        "(mut-m4-override-reads-baked)."
    )
    # The two answers genuinely differ — the override is non-vacuous.
    assert overridden.nodes != baked.nodes


def test_lean_override_raises() -> None:
    """A per-query override on a lean view raises WeightOverrideRequiresFullMode."""
    drive(_lean_override_case())


async def _lean_override_case() -> None:
    lean = await _build_diamond("lean")
    # Baked weighted SP still works in lean (no override) ...
    baked = await lean.weighted_shortest_path(N1, N4, direction="out")
    assert baked is not None and baked.nodes == (N1, N2, N4)
    # ... but a per-query override must fail closed, never silently fall back.
    with pytest.raises(WeightOverrideRequiresFullMode):
        await lean.weighted_shortest_path(
            N1, N4, direction="out", weight=_override_weight
        )
