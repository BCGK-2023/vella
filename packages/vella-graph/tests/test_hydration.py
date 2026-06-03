"""Hydration semantics (M3): full = fold-pinned, lean = live via LRU.

Asserts the four hydration invariants from the spec / plan:

* ``full`` hydration returns the FOLD-PINNED body (a pure in-memory lookup);
* ``lean`` hydration returns the LIVE body via the LRU (a ``get()`` on miss);
* an entity edited AFTER the fold shows the documented per-mode difference — ``full``
  still returns the fold-time body, ``lean`` returns the live (edited) one;
* ``hydrate=True`` inline returns the SAME ids as ``hydrate=False`` followed by an
  explicit ``hydrate()`` of those ids (and the same bodies).

The bodies are compared via ``model_dump(mode="json")`` (never Python ``==`` — core's
registry PrivateAttr breaks ``==``), mirroring the runtime house rule.
"""

from __future__ import annotations

from uuid import UUID

from vella.core import EdgeTypes
from vella.runtime import Runtime

from vella.graph import GraphProjection

from _fixtures import drive, make_node, thing_registry

A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_TENANT = "t"


def test_full_is_fold_pinned_lean_is_live() -> None:
    """An entity edited after fold: full keeps fold-time body, lean returns live."""
    drive(_pinned_vs_live_case())


async def _pinned_vs_live_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    await rt.create(make_node(thing, tenant_id=_TENANT, node_id=A, label="before"))
    await rt.create(make_node(thing, tenant_id=_TENANT, node_id=B, label="x"))
    await rt.link(_TENANT, A, B, EdgeTypes.KNOWS)

    proj = GraphProjection()
    full = await proj.fold(rt, _TENANT, mode="full")
    lean = await proj.fold(rt, _TENANT, mode="lean")

    # Mutate node A AFTER both folds. create() stamps version 1; edit expects it.
    await rt.edit(_TENANT, A, expected_version=1, data=thing(label="after"))

    full_body = await full.get(A)
    lean_body = await lean.get(A)
    assert full_body is not None and lean_body is not None
    # full = fold-pinned snapshot (the pre-edit label); lean = live (post-edit).
    assert full_body.data.label == "before"
    assert lean_body.data.label == "after"


def test_inline_equals_explicit_hydrate() -> None:
    """hydrate=True inline returns the same ids and bodies as explicit hydrate()."""
    drive(_inline_case())


async def _inline_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    await rt.create(make_node(thing, tenant_id=_TENANT, node_id=A, label="a"))
    await rt.create(make_node(thing, tenant_id=_TENANT, node_id=B, label="b"))
    await rt.link(_TENANT, A, B, EdgeTypes.KNOWS)

    for mode in ("full", "lean"):
        view = await GraphProjection().fold(rt, _TENANT, mode=mode)

        # neighbours: inline hydrate vs hydrate=False then explicit hydrate().
        plain = await view.neighbors(A, direction="out")
        inline = await view.neighbors(A, direction="out", hydrate=True)
        ids = [n.node_id for n in plain]
        assert [n.node_id for n in inline] == ids  # same ids
        explicit = await view.hydrate(ids)
        # bodies match (compare via JSON dump; None stays None).
        for nb, body in zip(inline, explicit):
            assert _dump(nb.body) == _dump(body)

        # shortest_path: inline hydrate vs explicit hydrate() of the same node seq.
        sp_plain = await view.shortest_path(A, B, direction="out")
        sp_inline = await view.shortest_path(A, B, direction="out", hydrate=True)
        assert sp_plain is not None and sp_inline is not None
        assert sp_inline.nodes == sp_plain.nodes  # same ids
        explicit_bodies = await view.hydrate(list(sp_plain.nodes))
        assert sp_inline.bodies is not None
        assert [_dump(b) for b in sp_inline.bodies] == [_dump(b) for b in explicit_bodies]


def test_dangling_hydrates_to_none() -> None:
    """A dangling endpoint id hydrates to None in both modes (its body is absent)."""
    drive(_dangling_case())


async def _dangling_case() -> None:
    rt = Runtime()
    thing = thing_registry()
    await rt.create(make_node(thing, tenant_id=_TENANT, node_id=A, label="a"))
    ghost = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    await rt.link(_TENANT, A, ghost, EdgeTypes.REFERENCES)

    for mode in ("full", "lean"):
        view = await GraphProjection().fold(rt, _TENANT, mode=mode)
        out = await view.neighbors(A, direction="out", hydrate=True)
        assert [n.node_id for n in out] == [ghost]  # the dangling id is returned
        assert out[0].body is None  # but its body is absent in both modes
        assert await view.get(ghost) is None


def _dump(body: object) -> object:
    """Body -> JSON-mode dump (or None), so comparisons never use Python ==."""
    return None if body is None else body.model_dump(mode="json")  # type: ignore[attr-defined]
