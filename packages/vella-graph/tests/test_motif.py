"""Bounded motif matching (M4): anchored, type-pruned, canonically sorted.

Covers the load-bearing motif claims:

* **Anchored matches are canonically sorted.** Every match begins at the supplied
  anchor and the match list is sorted by node-id tuple — deterministic regardless
  of fold / hash order.
* **Node-type pruning is honoured and non-vacuous.** A hop with ``to_node_type``
  keeps only endpoints of that folded node type; a node of the WRONG type reachable
  by the same edge type is excluded (dropping the ``to_node_type`` check —
  mut-m4-motif-no-type-prune — would let it through and the assertion goes red).
* **Multi-hop patterns join correctly** across two typed hops.
* **It is a bounded matcher, not a general language**, and is mode-equivalent
  (topology only) — the same matches in ``full`` and ``lean``.

Uses a local two-node-type registry (``person`` / ``company``) so node-type pruning
is exercised. Pinned ids give ``str(P1) < str(P2) < str(C1) < str(X1)``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from vella.core import EdgeTypes, Node, Registry as CoreRegistry, VellaModel, node_type
from vella.runtime import Runtime

from vella.graph import GraphProjection, GraphView, MotifHop, MotifPattern

from _fixtures import drive

# Pinned ids: str() order is P1 < P2 < C1 < C2 < X1.
P1 = UUID("11111111-1111-1111-1111-111111111111")  # person (anchor)
P2 = UUID("22222222-2222-2222-2222-222222222222")  # person
C1 = UUID("c1111111-1111-1111-1111-111111111111")  # company
C2 = UUID("c2222222-2222-2222-2222-222222222222")  # company
X1 = UUID("e1111111-1111-1111-1111-111111111111")  # WRONG type (gadget) — must be pruned
_TENANT = "t"


def _typed_registry() -> tuple[type, type, type]:
    """Isolated registry with three node kinds: person, company, gadget."""
    reg = CoreRegistry()

    @node_type("person", registry=reg)
    class PersonData(VellaModel):
        label: str = "p"

    @node_type("company", registry=reg)
    class CompanyData(VellaModel):
        label: str = "c"

    @node_type("gadget", registry=reg)
    class GadgetData(VellaModel):
        label: str = "g"

    return PersonData, CompanyData, GadgetData


def _node(data_cls: type, kind: str, node_id: UUID) -> "Node[Any, Any]":
    """A minimal node of ``kind`` with the given id."""
    return Node[data_cls, Any](  # type: ignore[valid-type]
        id=node_id,
        type=kind,
        name="n",
        created_by=uuid4(),
        data=data_cls(),
        tenant_id=_TENANT,
    )


async def _build(mode: str) -> GraphView:
    """Fold a typed graph for motif matching.

    Topology (all edges KNOWS):
        P1 -> P2  (person -> person)
        P1 -> X1  (person -> gadget   — wrong type for a company hop)
        P1 -> C1  (person -> company)
        P2 -> C2  (person -> company; for the two-hop person->person->company motif)
    """
    person, company, gadget = _typed_registry()
    rt = Runtime()
    await rt.create(_node(person, "person", P1))
    await rt.create(_node(person, "person", P2))
    await rt.create(_node(company, "company", C1))
    await rt.create(_node(company, "company", C2))
    await rt.create(_node(gadget, "gadget", X1))
    await rt.link(_TENANT, P1, P2, EdgeTypes.KNOWS)
    await rt.link(_TENANT, P1, X1, EdgeTypes.KNOWS)
    await rt.link(_TENANT, P1, C1, EdgeTypes.KNOWS)
    await rt.link(_TENANT, P2, C2, EdgeTypes.KNOWS)
    return await GraphProjection().fold(rt, _TENANT, mode=mode)  # type: ignore[arg-type]


def test_single_hop_type_pruning_excludes_wrong_type() -> None:
    """A one-hop company motif keeps only company endpoints; the gadget is pruned."""
    drive(_type_pruning_case())


async def _type_pruning_case() -> None:
    view = await _build("full")
    # P1 has three KNOWS out-edges: to P2 (person), X1 (gadget), C1 (company).
    # The hop demands to_node_type="company" -> only C1 survives; P2 and X1 pruned.
    pattern = MotifPattern(
        hops=(MotifHop(edge_type=EdgeTypes.KNOWS, direction="out", to_node_type="company"),)
    )
    matches = await view.match(pattern, anchor=P1)
    assert [m.nodes for m in matches] == [(P1, C1)], (
        "only the company endpoint C1 should match; P2 (person) and X1 (gadget) "
        "must be pruned by to_node_type (mut-m4-motif-no-type-prune)."
    )


def test_single_hop_untyped_returns_all_sorted() -> None:
    """An untyped hop returns every endpoint, canonically sorted by node-id tuple."""
    drive(_untyped_sorted_case())


async def _untyped_sorted_case() -> None:
    view = await _build("full")
    pattern = MotifPattern(hops=(MotifHop(edge_type=EdgeTypes.KNOWS, direction="out"),))
    matches = await view.match(pattern, anchor=P1)
    # All three KNOWS endpoints, sorted by node-id tuple: P2 < C1 < X1 by str().
    assert [m.nodes for m in matches] == [(P1, P2), (P1, C1), (P1, X1)]
    # Every match is anchored at P1.
    assert all(m.nodes[0] == P1 for m in matches)


def test_multi_hop_person_person_company() -> None:
    """A two-hop motif person->person->company joins across hops, type-pruned."""
    drive(_multi_hop_case())


async def _multi_hop_case() -> None:
    view = await _build("full")
    pattern = MotifPattern(
        hops=(
            MotifHop(edge_type=EdgeTypes.KNOWS, direction="out", to_node_type="person"),
            MotifHop(edge_type=EdgeTypes.KNOWS, direction="out", to_node_type="company"),
        )
    )
    matches = await view.match(pattern, anchor=P1)
    # P1 -> P2 (person) -> C2 (company) is the only person->person->company chain.
    assert [m.nodes for m in matches] == [(P1, P2, C2)]


def test_motif_mode_equivalent() -> None:
    """Motif matches (topology only) are identical full vs lean."""
    drive(_motif_equivalence_case())


async def _motif_equivalence_case() -> None:
    full = await _build("full")
    lean = await _build("lean")
    pattern = MotifPattern(hops=(MotifHop(edge_type=EdgeTypes.KNOWS, direction="out"),))
    f = [m.nodes for m in await full.match(pattern, anchor=P1)]
    le = [m.nodes for m in await lean.match(pattern, anchor=P1)]
    assert f == le
