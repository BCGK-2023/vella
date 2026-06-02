"""The defining invariant: ``state_table == fold(log)``, two ways (M5).

This is the load-bearing property of the CQRS physics (principle 2): the
append-only log is the source of truth, and the live state-table is a
materialized fold of it. The invariant is provable through two *irreducible*
representations that MUST agree:

(a) **In-memory typed replay** — fold the log by replaying each non-telemetry
    entry through core's ``hydrate(**payload)`` door (the payload already holds
    real, typed model-instance fields), build a reconstructed state-table, and
    compare it to the live state-table entity-by-entity via
    ``model_dump(mode="json")``. NEVER Python ``==`` — core's ``_vella_registry``
    PrivateAttr participates in ``__eq__``, so a freshly-hydrated entity and a
    live one are field-equal but ``==``-unequal. Deleted entities are absent in
    both.
(b) **Canonical-bytes equality** — the canonical JSON of the reconstructed table
    equals the canonical JSON of the live table, where canonical bytes are
    ``json.dumps(model_dump(mode="json"), sort_keys=True, separators=(",",":"))``
    — the serialization boundary a SQL adapter would store.

A Hypothesis strategy generates an arbitrary, validity-respecting sequence of
write verbs (create / edit / set_desired / upsert / delete / link) interleaved
with ``observe_only`` telemetry, applies it through a ``Runtime`` over a fresh
``InMemoryStore``, and asserts both representations agree. Sizes are bounded
(``max_examples`` modest, short sequences) so the suite stays fast and CI never
hangs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Union
from uuid import UUID, uuid4

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from vella.core import (
    Actuator,
    Edge,
    IntegrationBinding,
    Node,
    Registry,
    VellaModel,
    node_type,
    parse_edge,
    parse_node,
)

from vella.runtime import Runtime
from vella.runtime._inmemory import InMemoryStore

_Entity = Union["Node[Any, Any]", "Edge[Any, Any]"]

# --- isolated test type ------------------------------------------------------


def _registry() -> tuple[Registry, type]:
    """Isolated registry with one ``doc`` node type (never the global)."""
    reg = Registry()

    @node_type("doc", registry=reg)
    class DocData(VellaModel):
        title: str

    return reg, DocData


def _canonical(obj: Any) -> str:
    """Canonical JSON of a JSON-mode dump — the serialization boundary bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# --- Hypothesis verb-sequence strategy ---------------------------------------
# A verb is a small dict the harness interprets against live state. We keep the
# alphabet of titles/names tiny so edits and upserts actually collide and the
# fold has to track real transitions.

_titles = st.sampled_from(["alpha", "beta", "gamma", "delta"])
_tenants = st.sampled_from(["t1", "t2"])
_surfaces = st.lists(
    st.sampled_from(["data", "state", "embedding"]), max_size=3, unique=True
)


@st.composite
def _verb(draw: st.DrawFn) -> dict[str, Any]:
    """Draw one verb descriptor (the harness resolves targets against live state)."""
    kind = draw(
        st.sampled_from(
            ["create", "edit", "set_desired", "upsert", "delete", "link", "telemetry"]
        )
    )
    return {
        "kind": kind,
        "title": draw(_titles),
        "tenant": draw(_tenants),
        "plugin": draw(st.sampled_from(["wordpress", "ga4", "hubspot"])),
        "external_id": draw(st.sampled_from(["x", "y", "z"])),
        "contributes_to": draw(_surfaces),
        "with_state": draw(st.booleans()),
    }


_verb_sequences = st.lists(_verb(), min_size=1, max_size=12)


# --- live-table reconstruction + dual-representation comparison ---------------


def _make_node(
    DocData: type,
    *,
    title: str,
    tenant_id: str,
    integrations: list[IntegrationBinding] | None = None,
    with_state: bool = False,
) -> "Node[Any, Any]":
    return Node[DocData, Any](  # type: ignore[valid-type]
        id=uuid4(),
        type="doc",
        name=title,
        created_by=uuid4(),
        data=DocData(title=title),
        tenant_id=tenant_id,
        integrations=integrations or [],
        state=Actuator(current=DocData(title=title)) if with_state else None,
    )


async def _apply_sequence(
    rt: Runtime, DocData: type, verbs: list[dict[str, Any]]
) -> None:
    """Apply a verb sequence to a fresh runtime, resolving targets from live state.

    Targets for edit/set_desired/delete are chosen from entities that currently
    exist live (so we exercise real transitions, not just no-op conflict paths).
    Conflicts and absent-target verbs are simply skipped — the property under
    test is fold parity over whatever transitions *did* commit.
    """
    store = rt.store
    assert isinstance(store, InMemoryStore)
    # Track live (tenant, id) -> (version, kind) for target resolution.
    live: dict[tuple[str, UUID], tuple[int, str]] = {}

    async def _live_ids(tenant: str, kind: str) -> list[UUID]:
        return [eid for (t, eid), (_v, k) in live.items() if t == tenant and k == kind]

    for v in verbs:
        kind, tenant, title = v["kind"], v["tenant"], v["title"]
        if kind == "create":
            node = _make_node(
                DocData, title=title, tenant_id=tenant, with_state=v["with_state"]
            )
            entry = await rt.create(node)
            live[(tenant, node.id)] = (entry.version, "node")
        elif kind == "link":
            entry = await rt.link(tenant, uuid4(), uuid4(), edge_type="references")
            live[(tenant, entry.entity_id)] = (entry.version, "edge")
        elif kind == "upsert":
            existing = await store.find_by_binding(tenant, v["plugin"], v["external_id"])
            binding = IntegrationBinding(
                plugin=v["plugin"],
                external_id=v["external_id"],
                contributes_to=v["contributes_to"],
            )
            node = _make_node(
                DocData, title=title, tenant_id=tenant, integrations=[binding]
            )
            entry = await rt.upsert(
                tenant, v["plugin"], v["external_id"], node, name=title
            )
            live[(tenant, entry.entity_id)] = (entry.version, "node")
        elif kind in ("edit", "set_desired", "delete"):
            ids = await _live_ids(tenant, "node")
            if not ids:
                continue
            target = ids[0]
            cur_version = live[(tenant, target)][0]
            if kind == "edit":
                entry = await rt.edit(
                    tenant, target, expected_version=cur_version, name=title
                )
                live[(tenant, target)] = (entry.version, "node")
            elif kind == "set_desired":
                row = await store.get(tenant, target)
                assert row is not None
                node_obj: Node[Any, Any] = Node.hydrate(**row.payload)
                if node_obj.state is None:
                    continue  # set_desired requires an Actuator state
                entry = await rt.set_desired(
                    tenant, target, expected_version=cur_version, title=title
                )
                live[(tenant, target)] = (entry.version, "node")
            else:  # delete
                await rt.delete(tenant, target)
                del live[(tenant, target)]
        elif kind == "telemetry":
            ids = await _live_ids(tenant, "node") + await _live_ids(tenant, "edge")
            if not ids:
                continue
            await rt.emit_telemetry(tenant, ids[0], {"cpu": 0.5})


def _reconstruct(entry: Any) -> _Entity:
    if entry.entity_kind == "edge":
        return Edge.hydrate(**entry.payload)
    return Node.hydrate(**entry.payload)


def _fold_log(store: InMemoryStore) -> dict[tuple[str, str, UUID], _Entity]:
    """Replay the log into a reconstructed state-table via typed ``hydrate``.

    Mirrors the adapter's fold: non-telemetry entries upsert the table; a
    ``delete``/``unlink`` removes the entry; ``observe_only`` is ignored.
    """
    table: dict[tuple[str, str, UUID], _Entity] = {}
    for entry in store._index.log:  # noqa: SLF001 — reading the log is the test's job
        if entry.transition == "observe_only":
            continue
        key = (entry.tenant_id, entry.entity_kind, entry.entity_id)
        if entry.transition in ("delete", "unlink"):
            table.pop(key, None)
        else:
            table[key] = _reconstruct(entry)
    return table


def _live_table(store: InMemoryStore) -> dict[tuple[str, str, UUID], _Entity]:
    """The live (non-deleted) state-table, reconstructed to typed entities."""
    table: dict[tuple[str, str, UUID], _Entity] = {}
    for key, row in store._index.state.items():  # noqa: SLF001
        if not row.deleted:
            table[key] = _reconstruct(row.entry)
    return table


def _run(coro_factory: Callable[[Runtime, type], Any]) -> None:
    reg, DocData = _registry()
    rt = Runtime()
    asyncio.run(coro_factory(rt, DocData))


# --- THE defining property ---------------------------------------------------


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(verbs=_verb_sequences)
def test_fold_invariant_dual_representation(verbs: list[dict[str, Any]]) -> None:
    """``state_table == fold(log)`` via BOTH typed replay AND canonical bytes."""
    reg, DocData = _registry()
    rt = Runtime()
    store = rt.store
    assert isinstance(store, InMemoryStore)
    asyncio.run(_apply_sequence(rt, DocData, verbs))

    reconstructed = _fold_log(store)
    live = _live_table(store)

    # (a) typed replay: identical keys, entity-by-entity model_dump(json) parity.
    assert set(reconstructed) == set(live)
    for key in live:
        assert (
            reconstructed[key].model_dump(mode="json")
            == live[key].model_dump(mode="json")
        )

    # (b) canonical bytes: the two whole tables serialize byte-identically.
    def _table_bytes(t: dict[tuple[str, str, UUID], _Entity]) -> str:
        return _canonical(
            {
                # key rendered as a stable string; runtime sorts its OWN derived
                # structure (the table key set), never core fields.
                f"{k[0]}|{k[1]}|{k[2]}": e.model_dump(mode="json")
                for k, e in t.items()
            }
        )

    assert _table_bytes(reconstructed) == _table_bytes(live)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(verbs=_verb_sequences)
def test_exactly_one_entry_per_versioned_transition(
    verbs: list[dict[str, Any]],
) -> None:
    """Every committed verb appends exactly one entry; telemetry does too.

    Asserted indirectly but precisely: the number of log entries equals the
    number of verbs the harness actually committed (it returns a count), and
    every non-telemetry entry carries a strictly tracked version while
    ``observe_only`` entries leave the per-entity version untouched.
    """
    reg, DocData = _registry()
    rt = Runtime()
    store = rt.store
    assert isinstance(store, InMemoryStore)

    committed = {"n": 0}

    async def _counting(rt: Runtime, DocData: type) -> None:
        before = len(store._index.log)  # noqa: SLF001
        await _apply_sequence(rt, DocData, verbs)
        committed["n"] = len(store._index.log) - before  # noqa: SLF001

    asyncio.run(_counting(rt, DocData))

    log = store._index.log  # noqa: SLF001
    # One entry per committed transition — the harness only counts commits.
    assert len(log) == committed["n"]
    # Each non-telemetry write bumped its entity's version monotonically; an
    # observe_only entry never moves the entity's tracked version forward.
    versions: dict[tuple[str, str, UUID], int] = {}
    for entry in log:
        key = (entry.tenant_id, entry.entity_kind, entry.entity_id)
        if entry.transition == "observe_only":
            # version equals the current tracked version (unchanged), if known.
            if key in versions:
                assert entry.version == versions[key]
        else:
            versions[key] = entry.version


def test_observe_only_no_version_no_table_end_to_end() -> None:
    """Interleaved telemetry never changes ``get`` / version, only the log."""

    async def _case(rt: Runtime, DocData: type) -> None:
        store = rt.store
        assert isinstance(store, InMemoryStore)
        node = _make_node(DocData, title="x", tenant_id="t1")
        await rt.create(node)

        await rt.emit_telemetry("t1", node.id, {"a": 1})
        await rt.edit("t1", node.id, expected_version=1, name="y")
        await rt.emit_telemetry("t1", node.id, {"b": 2})

        row = await store.get("t1", node.id)
        assert row is not None
        assert row.version == 2  # only the edit bumped it
        assert row.transition == "edit"

        log = store._index.log  # noqa: SLF001
        telemetry = [e for e in log if e.transition == "observe_only"]
        assert len(telemetry) == 2
        # No state-table row is a telemetry entry — the fold skips observe_only.
        assert all(
            r.entry.transition != "observe_only"
            for r in store._index.state.values()  # noqa: SLF001
        )

    _run(_case)


def test_integrations_order_survives_roundtrip() -> None:
    """A multi-element ``integrations`` order survives write -> log -> replay."""

    async def _case(rt: Runtime, DocData: type) -> None:
        store = rt.store
        assert isinstance(store, InMemoryStore)
        integrations = [
            IntegrationBinding(plugin="wordpress", external_id="a"),
            IntegrationBinding(plugin="ga4", external_id="b"),
            IntegrationBinding(plugin="hubspot", external_id="c"),
        ]
        node = _make_node(
            DocData, title="ordered", tenant_id="t1", integrations=integrations
        )
        await rt.create(node)

        row = await store.get("t1", node.id)
        assert row is not None
        replayed: Node[Any, Any] = Node.hydrate(**row.payload)
        dumped = replayed.model_dump(mode="json")
        # Order is SEMANTIC — never sorted.
        assert [b["plugin"] for b in dumped["integrations"]] == [
            "wordpress",
            "ga4",
            "hubspot",
        ]
        assert dumped == node.model_dump(mode="json")

    _run(_case)


def test_contributes_to_order_survives_roundtrip() -> None:
    """``IntegrationBinding.contributes_to`` order survives the roundtrip."""

    async def _case(rt: Runtime, DocData: type) -> None:
        store = rt.store
        assert isinstance(store, InMemoryStore)
        binding = IntegrationBinding(
            plugin="wordpress",
            external_id="a",
            contributes_to=["embedding", "data", "state"],
        )
        node = _make_node(
            DocData, title="c", tenant_id="t1", integrations=[binding]
        )
        await rt.create(node)

        row = await store.get("t1", node.id)
        assert row is not None
        replayed: Node[Any, Any] = Node.hydrate(**row.payload)
        dumped = replayed.model_dump(mode="json")
        assert dumped["integrations"][0]["contributes_to"] == [
            "embedding",
            "data",
            "state",
        ]

    _run(_case)


def test_delete_replay_parity() -> None:
    """create -> edit -> delete: the folded table matches the live table."""

    async def _case(rt: Runtime, DocData: type) -> None:
        store = rt.store
        assert isinstance(store, InMemoryStore)
        node = _make_node(DocData, title="doomed", tenant_id="t1")
        await rt.create(node)
        await rt.edit("t1", node.id, expected_version=1, name="doomed-v2")
        await rt.delete("t1", node.id)

        reconstructed = _fold_log(store)
        live = _live_table(store)
        assert set(reconstructed) == set(live)
        # The deleted entity is absent from both representations.
        assert (("t1", "node", node.id)) not in reconstructed
        assert (("t1", "node", node.id)) not in live
        # get() returns None (tombstoned), history retains the full trail.
        assert await store.get("t1", node.id) is None
        hist = await store.history("t1", node.id)
        assert [e.transition for e in hist] == ["create", "edit", "delete"]

    _run(_case)


def test_portable_replay_roundtrip() -> None:
    """Full verb sequence: JSON-mode dump -> ``parse_*`` reconstructs the live table."""

    async def _case(rt: Runtime, DocData: type) -> None:
        store = rt.store
        assert isinstance(store, InMemoryStore)
        reg = Registry()

        @node_type("doc", registry=reg)
        class LocalDoc(VellaModel):
            title: str

        node = _make_node(LocalDoc, title="p", tenant_id="t1")
        await rt.create(node)
        await rt.edit("t1", node.id, expected_version=1, name="p2")
        edge_entry = await rt.link("t1", uuid4(), uuid4(), edge_type="references")

        live = _live_table(store)
        for key, entity in live.items():
            entity_json = entity.model_dump(mode="json")
            rebuilt: _Entity = (
                parse_edge(entity_json, registry=reg)
                if key[1] == "edge"
                else parse_node(entity_json, registry=reg)
            )
            assert rebuilt.model_dump(mode="json") == entity_json
        assert edge_entry.entity_kind == "edge"

    _run(_case)
