"""Query-core correctness + determinism (M3).

Covers neighbours, BFS/DFS to depth-k, reachability, and unweighted shortest path
over a folded :class:`~vella.graph.GraphView`, asserting:

* sorted-id results (and sorted-id traversal visitation);
* the depth-k bound (a node one hop too far is excluded);
* ``edge_type`` and ``direction`` filters;
* the canonical (lexicographically-smallest) shortest path among equal-length ones;
* query mode-equivalence — every sorted-id result is byte-identical full vs lean
  (extends the M2 topology-equivalence to the M3 query methods).

Pinned ids are used so the canonical-path / sorted-order assertions are exact.
"""

from __future__ import annotations

from uuid import UUID

from vella.core import EdgeTypes
from vella.runtime import Runtime

from vella.graph import GraphProjection, GraphView

from _fixtures import drive, make_node, thing_registry

# Pinned node ids — chosen so str() order is N1 < N2 < N3 < N4 < N5.
N1 = UUID("11111111-1111-1111-1111-111111111111")
N2 = UUID("22222222-2222-2222-2222-222222222222")
N3 = UUID("33333333-3333-3333-3333-333333333333")
N4 = UUID("44444444-4444-4444-4444-444444444444")
N5 = UUID("55555555-5555-5555-5555-555555555555")
_TENANT = "t"


async def _build(mode: str) -> GraphView:
    """Fold a fixed 5-node graph in the given mode.

    Topology (all edges KNOWS unless noted):
        N1 -> N2, N1 -> N3 (OWNED_BY), N2 -> N4, N3 -> N4, N4 -> N5
    so N1 reaches N4 via two equal-length paths (N1-N2-N4 and N1-N3-N4); the
    canonical shortest path is the lexicographically smallest (N1-N2-N4).
    """
    rt = Runtime()
    thing = thing_registry()
    for nid in (N1, N2, N3, N4, N5):
        await rt.create(make_node(thing, tenant_id=_TENANT, node_id=nid))
    await rt.link(_TENANT, N1, N2, EdgeTypes.KNOWS)
    await rt.link(_TENANT, N1, N3, EdgeTypes.OWNED_BY)
    await rt.link(_TENANT, N2, N4, EdgeTypes.KNOWS)
    await rt.link(_TENANT, N3, N4, EdgeTypes.KNOWS)
    await rt.link(_TENANT, N4, N5, EdgeTypes.KNOWS)
    return await GraphProjection().fold(rt, _TENANT, mode=mode)  # type: ignore[arg-type]


# Adversarial constants for the unsorted-neighbors mutation test (mut-m3-unsorted-neighbors).
#
# The index stores records in bucket order (edge_type alphabetically, then edge_id).
# "knows" < "owned_by" alphabetically, so the KNOWS bucket comes first in storage.
# We route KNOWS at the HIGHER id and OWNED_BY at the LOWER id so that storage/bucket
# order [HIGH, LOW] is the OPPOSITE of the contract's endpoint-id-sorted order [LOW, HIGH].
# Removing neighbor_records' sorted() returns [HIGH, LOW] and the assertion below goes RED.
#
#   str(NB_LOW) = "aaaaaaaa-..." < str(NB_HIGH) = "ffffffff-..."   (provable from the values)
#   "knows"     < "owned_by"                                        (alphabetical)
#   storage order:  [NB_HIGH (via KNOWS bucket), NB_LOW (via OWNED_BY bucket)]
#   contract order: [NB_LOW, NB_HIGH]                               (endpoint-id sorted)
NB_SRC = UUID("10000000-0000-0000-0000-000000000000")
NB_LOW = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")  # lower str() — reached via OWNED_BY
NB_HIGH = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")  # higher str() — reached via KNOWS


def test_neighbors_sorted_and_filtered() -> None:
    """neighbours are canonically ordered and honour edge_type / direction."""
    drive(_neighbors_case())


async def _neighbors_case() -> None:
    view = await _build("full")
    # out neighbours of N1: N2 (KNOWS) and N3 (OWNED_BY) — sorted by endpoint id.
    out = await view.neighbors(N1, direction="out")
    assert [n.node_id for n in out] == [N2, N3]
    assert [n.edge_type for n in out] == [EdgeTypes.KNOWS, EdgeTypes.OWNED_BY]
    # edge_type filter: only the KNOWS neighbour remains.
    knows = await view.neighbors(N1, direction="out", edge_type=EdgeTypes.KNOWS)
    assert [n.node_id for n in knows] == [N2]
    # direction="in": N4's predecessors are N2 and N3, sorted.
    incoming = await view.neighbors(N4, direction="in")
    assert [n.node_id for n in incoming] == [N2, N3]
    # direction="both" at N4: predecessors {N2,N3} + successor {N5}, sorted.
    both = await view.neighbors(N4, direction="both")
    assert sorted({n.node_id for n in both}, key=str) == [N2, N3, N5]


def test_neighbors_sorted_by_endpoint_id_not_edge_type() -> None:
    """Neighbour order is endpoint-id, not storage/edge_type order (mut-m3-unsorted-neighbors).

    The index stores records in bucket order: ``(edge_type, str(edge_id))``.  Here
    ``KNOWS`` < ``OWNED_BY`` alphabetically so the ``KNOWS`` bucket comes first in
    storage — but the ``KNOWS`` edge points at the HIGHER endpoint id (``NB_HIGH``)
    and the ``OWNED_BY`` edge points at the LOWER one (``NB_LOW``).

    Storage order  → ``[NB_HIGH, NB_LOW]`` (wrong).
    Contract order → ``[NB_LOW,  NB_HIGH]`` (endpoint-id sorted, correct).

    Without the ``sorted()`` in ``_query.neighbor_records`` the storage order is
    returned and this assertion fails — proving the sort is non-vacuous.
    """
    drive(_unsorted_neighbors_adversarial_case())


async def _unsorted_neighbors_adversarial_case() -> None:
    # Self-documenting ordering guarantee: the constants are what we claim.
    assert str(NB_LOW) < str(NB_HIGH), "NB_LOW must have lower str() than NB_HIGH"
    assert EdgeTypes.KNOWS < EdgeTypes.OWNED_BY, "'knows' must sort before 'owned_by'"

    rt = Runtime()
    thing = thing_registry()
    for nid in (NB_SRC, NB_LOW, NB_HIGH):
        await rt.create(make_node(thing, tenant_id=_TENANT, node_id=nid))
    # KNOWS  -> NB_HIGH  (alphabetically-first edge_type, higher endpoint id)
    # OWNED_BY -> NB_LOW (alphabetically-later  edge_type, lower  endpoint id)
    await rt.link(_TENANT, NB_SRC, NB_HIGH, EdgeTypes.KNOWS)
    await rt.link(_TENANT, NB_SRC, NB_LOW, EdgeTypes.OWNED_BY)

    view = await GraphProjection().fold(rt, _TENANT, mode="full")
    result = await view.neighbors(NB_SRC, direction="out")
    node_ids = [n.node_id for n in result]

    # Contract: endpoint-id order → [NB_LOW, NB_HIGH].
    # Without sorted() the index returns storage/edge_type order → [NB_HIGH, NB_LOW].
    assert node_ids == [NB_LOW, NB_HIGH], (
        f"Expected endpoint-id order [NB_LOW, NB_HIGH] but got {node_ids}. "
        "If this fails under the mutation (sorted() removed), that is correct behaviour."
    )


def test_bfs_dfs_depth_bound_and_sorted() -> None:
    """BFS/DFS reach the same sorted set; depth-k excludes the too-far node."""
    drive(_bfs_dfs_case())


async def _bfs_dfs_case() -> None:
    view = await _build("full")
    # out from N1, depth 1: only N2, N3.
    d1 = await view.bfs(N1, depth=1, direction="out")
    assert d1 == [N2, N3]
    # depth 2 adds N4 (via N2/N3); N5 is 3 hops away -> excluded.
    d2 = await view.bfs(N1, depth=2, direction="out")
    assert d2 == [N2, N3, N4]
    assert N5 not in d2
    # depth 3 reaches N5.
    d3 = await view.bfs(N1, depth=3, direction="out")
    assert d3 == [N2, N3, N4, N5]
    # DFS reaches the identical sorted set for the same bound.
    assert await view.dfs(N1, depth=2, direction="out") == d2
    assert await view.dfs(N1, depth=3, direction="out") == d3
    # depth 0 reaches nothing.
    assert await view.bfs(N1, depth=0, direction="out") == []


def test_reachability_directional() -> None:
    """reachability honours direction and the trivial self case."""
    drive(_reach_case())


async def _reach_case() -> None:
    view = await _build("full")
    assert await view.reachable(N1, N5, direction="out") is True
    assert await view.reachable(N5, N1, direction="out") is False  # wrong direction
    assert await view.reachable(N5, N1, direction="in") is True  # reverse reaches it
    assert await view.reachable(N1, N1, direction="out") is True  # trivial self


def test_shortest_path_canonical() -> None:
    """unweighted shortest path is the lexicographically-smallest equal-length one."""
    drive(_sp_case())


async def _sp_case() -> None:
    view = await _build("full")
    path = await view.shortest_path(N1, N4, direction="out")
    assert path is not None
    # Two equal-length paths exist (N1-N2-N4, N1-N3-N4); canonical is the lex-smallest.
    assert path.nodes == (N1, N2, N4)
    # trivial self path.
    self_path = await view.shortest_path(N1, N1, direction="out")
    assert self_path is not None and self_path.nodes == (N1,)
    # unreachable -> None.
    assert await view.shortest_path(N5, N1, direction="out") is None


def test_query_mode_equivalence() -> None:
    """Every M3 query's sorted-id result is byte-identical full vs lean."""
    drive(_equivalence_case())


async def _equivalence_case() -> None:
    full = await _build("full")
    lean = await _build("lean")

    for direction in ("out", "in", "both"):
        for anchor in (N1, N2, N3, N4, N5):
            f_n = [n.node_id for n in await full.neighbors(anchor, direction=direction)]
            l_n = [n.node_id for n in await lean.neighbors(anchor, direction=direction)]
            assert f_n == l_n
            for depth in (0, 1, 2, 3):
                assert await full.bfs(anchor, depth=depth, direction=direction) == \
                    await lean.bfs(anchor, depth=depth, direction=direction)
                assert await full.dfs(anchor, depth=depth, direction=direction) == \
                    await lean.dfs(anchor, depth=depth, direction=direction)
            for target in (N1, N4, N5):
                assert await full.reachable(anchor, target, direction=direction) == \
                    await lean.reachable(anchor, target, direction=direction)
                fp = await full.shortest_path(anchor, target, direction=direction)
                lp = await lean.shortest_path(anchor, target, direction=direction)
                assert (fp.nodes if fp else None) == (lp.nodes if lp else None)
