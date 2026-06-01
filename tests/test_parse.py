"""Tests for the tolerant, registry-driven round-trip: parse_node / parse_edge."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from pydantic import SerializeAsAny

from vella.core import (
    Actuator,
    FlexibleData,
    Node,
    Registry,
    ToolDeclaration,
    UnregisteredTypeError,
    VellaModel,
    node_type,
    parse_edge,
    parse_node,
)


class DocState(VellaModel):
    pinned: bool = False


def _registry_with_doc(version: int = 1, migrations: Any = None) -> tuple[Registry, type]:
    reg = Registry()

    @node_type("doc", state=DocState, version=version, migrations=migrations, registry=reg)
    class DocData(VellaModel):
        title: str

    return reg, DocData


def test_roundtrip_preserves_typed_data_and_state() -> None:
    reg, DocData = _registry_with_doc()
    node = Node[DocData, DocState](  # type: ignore[valid-type]
        type="doc",
        name="readme",
        created_by=uuid4(),
        data=DocData(title="Hello"),
        state=Actuator(current=DocState(pinned=False), desired=DocState(pinned=True)),
    )
    raw = node.model_dump(mode="json")

    back = parse_node(raw, registry=reg)
    assert type(back.data).__name__ == "DocData"
    assert back.data.title == "Hello"
    assert isinstance(back.state, Actuator)
    assert back.state.desired is not None and back.state.desired.pinned is True
    assert back.state.current.pinned is False


def test_unknown_type_lenient_becomes_flexible() -> None:
    reg = Registry()
    raw = {"type": "mystery", "name": "?", "created_by": str(uuid4()), "data": {"k": 1}}
    back = parse_node(raw, registry=reg)
    assert isinstance(back.data, FlexibleData)
    assert back.data.model_dump()["k"] == 1  # arbitrary field preserved


def test_unknown_type_strict_raises() -> None:
    reg = Registry()
    raw = {"type": "mystery", "name": "?", "created_by": str(uuid4()), "data": {}}
    with pytest.raises(UnregisteredTypeError) as exc:
        parse_node(raw, registry=reg, strict=True)
    assert exc.value.type_name == "mystery"


def test_migration_chain_upgrades_old_data() -> None:
    # v1 data used "name"; v2 renamed it to "title". Register a v1->v2 migration.
    reg = Registry()

    @node_type(
        "doc",
        version=2,
        migrations={1: lambda d: {"title": d.pop("name")}},
        registry=reg,
    )
    class DocV2(VellaModel):
        title: str

    raw = {
        "type": "doc",
        "schema_version": 1,
        "name": "old",
        "created_by": str(uuid4()),
        "data": {"name": "legacy title"},
    }
    back = parse_node(raw, registry=reg)
    assert type(back.data).__name__ == "DocV2"
    assert back.data.title == "legacy title"
    assert back.schema_version == 2


def test_validation_failure_is_quarantined_not_raised() -> None:
    reg, _DocData = _registry_with_doc()
    # 'title' is required but missing -> lenient parse quarantines.
    raw = {"type": "doc", "name": "broken", "created_by": str(uuid4()), "data": {}}
    back = parse_node(raw, registry=reg)
    assert isinstance(back.data, FlexibleData)
    dumped = back.data.model_dump()
    assert "vella_repair" in dumped
    assert "reason" in dumped["vella_repair"]


def test_validation_failure_strict_raises() -> None:
    reg, _DocData = _registry_with_doc()
    raw = {"type": "doc", "name": "broken", "created_by": str(uuid4()), "data": {}}
    with pytest.raises(Exception):
        parse_node(raw, registry=reg, strict=True)


def test_unknown_envelope_field_is_tolerated_not_thrown() -> None:
    # A newer writer adds an envelope-level field; the tolerant reader ignores it.
    reg, _DocData = _registry_with_doc()
    raw = {
        "type": "doc",
        "name": "ok",
        "created_by": str(uuid4()),
        "data": {"title": "Hi"},
        "future_envelope_field": "from a newer version",
    }
    back = parse_node(raw, registry=reg)  # must not raise
    assert back.data.title == "Hi"


def test_isolated_registry_tool_overrides_validate_against_that_registry() -> None:
    reg = Registry()

    @node_type("widget", tools=[ToolDeclaration(name="press", description="d")], registry=reg)
    class WidgetData(VellaModel):
        label: str

    raw = {
        "type": "widget",
        "name": "w",
        "created_by": str(uuid4()),
        "data": {"label": "x"},
        "tool_overrides": [{"tool_name": "press"}],  # valid in `reg`, absent from default
    }
    back = parse_node(raw, registry=reg)  # must not raise / must not quarantine
    assert back.tool_overrides[0].tool_name == "press"
    assert type(back.data).__name__ == "WidgetData"


def test_quarantine_preserves_clobbered_user_data_and_does_not_nest() -> None:
    reg, _DocData = _registry_with_doc()
    # 'title' missing -> quarantine; real data also happens to use the marker key.
    raw = {
        "type": "doc",
        "name": "broken",
        "created_by": str(uuid4()),
        "data": {"vella_repair": "REAL USER VALUE"},
    }
    q = parse_node(raw, registry=reg)
    marker = q.data.model_dump()["vella_repair"]
    assert marker["shadowed"] == "REAL USER VALUE"  # not silently clobbered
    assert "reason" in marker
    # Re-parsing the quarantined node must not nest repair markers unboundedly.
    q2 = parse_node(q.model_dump(mode="json"), registry=reg)
    marker2 = q2.data.model_dump()["vella_repair"]
    assert not isinstance(marker2.get("shadowed"), dict) or "reason" not in marker2["shadowed"]


def test_nested_polymorphic_field_round_trips_with_serialize_as_any() -> None:
    reg = Registry()

    class Shape(VellaModel):
        kind: str = "shape"

    class Circle(Shape):
        kind: str = "circle"
        radius: float = 1.0

    @node_type("drawing", registry=reg)
    class DrawingData(VellaModel):
        shape: SerializeAsAny[Shape]  # the documented pattern for nested polymorphism

    node = Node[DrawingData](
        type="drawing", name="d", created_by=uuid4(), data=DrawingData(shape=Circle(radius=2.5))
    )
    back = parse_node(node.model_dump(mode="json"), registry=reg)
    assert back.data.model_dump()["shape"]["radius"] == 2.5  # subclass field survives the wire


def test_nested_polymorphic_WITHOUT_serialize_as_any_loses_data() -> None:
    # Documents the C2 footgun: a base-typed sub-field without SerializeAsAny
    # erodes the subclass's fields on dump. (Discriminated unions are exempt.)
    reg = Registry()

    class Base(VellaModel):
        kind: str = "base"

    class Sub(Base):
        kind: str = "sub"
        extra: str = ""

    @node_type("plain_poly", registry=reg)
    class PlainData(VellaModel):
        item: Base  # NOT SerializeAsAny -> erosion

    node = Node[PlainData](
        type="plain_poly", name="n", created_by=uuid4(), data=PlainData(item=Sub(extra="LOST"))
    )
    assert "extra" not in node.model_dump(mode="json")["data"]["item"]


@pytest.mark.parametrize(
    "bad",
    [
        {"extra_tools": [{"name": "x"}]},          # missing required 'description'
        {"integrations": [{"plugin": "p"}]},        # missing required 'external_id'
        {"state": {"kind": "nonsense"}},            # bad discriminator
        {"embedding": {"model": "m"}},              # missing required vector/dimensions
        {"created_by": "not-a-uuid"},               # malformed scalar -> last-resort path
        {"flags": {"system_protected": "not-a-bool"}},
    ],
)
def test_quarantine_never_throws_on_any_malformed_subsurface(bad: dict[str, Any]) -> None:
    reg, _DocData = _registry_with_doc()
    raw = {"type": "doc", "name": "n", "created_by": str(uuid4()), "data": {"title": "ok"}, **bad}
    node = parse_node(raw, registry=reg)  # invariant: never raises
    assert isinstance(node.data, FlexibleData)


_MALFORMED_TOP_LEVEL = [
    {"data": [1, 2, 3]},          # data is a list, not an object
    {"data": "a string"},         # data is a string
    {"data": 5},                  # data is an int
    {"schema_version": "v-two"},  # non-int schema_version
    {"schema_version": []},       # nonsense schema_version
    {"type": []},                 # unhashable type value
    {"type": {}},                 # unhashable type value
    {"type": None},               # missing-ish type
]


@pytest.mark.parametrize("bad_top", _MALFORMED_TOP_LEVEL)
def test_quarantine_never_throws_on_malformed_top_level_fields(bad_top: dict[str, Any]) -> None:
    reg, _DocData = _registry_with_doc()
    raw: dict[str, Any] = {"type": "doc", "name": "n", "created_by": str(uuid4()), "data": {"title": "ok"}}
    raw.update(bad_top)
    node = parse_node(raw, registry=reg)  # invariant: never raises
    assert isinstance(node.data, FlexibleData)


@pytest.mark.parametrize("bad_top", _MALFORMED_TOP_LEVEL)
def test_parse_edge_quarantine_never_throws_on_malformed_top_level(bad_top: dict[str, Any]) -> None:
    reg, _DocData = _registry_with_doc()
    raw: dict[str, Any] = {
        "type": "doc",
        "from_node_id": str(uuid4()),
        "to_node_id": str(uuid4()),
        "created_by": str(uuid4()),
        "data": {"title": "ok"},
    }
    raw.update(bad_top)
    edge = parse_edge(raw, registry=reg)  # invariant: never raises
    assert isinstance(edge.data, FlexibleData)


def test_non_mapping_data_is_preserved_under_marker() -> None:
    reg, _DocData = _registry_with_doc()
    raw = {"type": "doc", "name": "n", "created_by": str(uuid4()), "data": "raw string payload"}
    node = parse_node(raw, registry=reg)
    assert node.data.model_dump()["vella_repair"]["shadowed_data"] == "raw string payload"


def test_real_data_with_reason_key_is_not_mistaken_for_a_marker() -> None:
    reg, _DocData = _registry_with_doc()
    real = {"reason": "a genuine user note", "level": 5}  # has 'reason' but is NOT our marker
    raw = {"type": "doc", "name": "n", "created_by": str(uuid4()), "data": {"vella_repair": real}}
    node = parse_node(raw, registry=reg)  # missing 'title' -> quarantine
    assert node.data.model_dump()["vella_repair"]["shadowed"] == real  # preserved, not discarded


def test_shadowed_survives_malformed_type_quarantine() -> None:
    # Round-5: a malformed identity field (type) must NOT cause the preserved
    # user value to be lost via the last-resort path.
    reg, _DocData = _registry_with_doc()
    raw = {"type": [], "name": "n", "created_by": str(uuid4()), "data": {"vella_repair": "REAL"}}
    node = parse_node(raw, registry=reg)
    assert node.data.model_dump()["vella_repair"]["shadowed"] == "REAL"


def test_shadowed_data_survives_malformed_type_and_nonmapping_data() -> None:
    reg, _DocData = _registry_with_doc()
    raw = {"type": {}, "name": "n", "created_by": str(uuid4()), "data": "raw payload"}
    node = parse_node(raw, registry=reg)
    assert node.data.model_dump()["vella_repair"]["shadowed_data"] == "raw payload"


def test_marker_preserved_on_last_resort_path() -> None:
    # created_by is unrecoverable -> last-resort node; the marker (and shadowed
    # value) must still be carried through.
    reg, _DocData = _registry_with_doc()
    raw = {
        "type": "doc",
        "name": "n",
        "created_by": {"garbage": "not a ref"},
        "data": {"vella_repair": "REAL"},
    }
    node = parse_node(raw, registry=reg)
    assert node.data.model_dump()["vella_repair"]["shadowed"] == "REAL"


def test_evolve_and_model_copy_work_on_isolated_registry_node() -> None:
    reg = Registry()

    @node_type("gadget", tools=[ToolDeclaration(name="press", description="d")], registry=reg)
    class GadgetData(VellaModel):
        label: str

    raw = {
        "type": "gadget",
        "name": "g",
        "created_by": str(uuid4()),
        "data": {"label": "x"},
        "tool_overrides": [{"tool_name": "press"}],
    }
    node = parse_node(raw, registry=reg)
    evolved = node.evolve(name="renamed")  # must not raise: reuses the carried registry
    assert evolved.name == "renamed" and evolved.tool_overrides[0].tool_name == "press"
    copied = node.model_copy(update={"name": "copied"})
    assert copied.name == "copied"


def test_shadowed_user_value_survives_repeated_reparse() -> None:
    reg, _DocData = _registry_with_doc()
    raw = {"type": "doc", "name": "n", "created_by": str(uuid4()), "data": {"vella_repair": "REAL"}}
    q = parse_node(raw, registry=reg)
    for _ in range(3):  # re-parse repeatedly; the preserved value must not vanish
        q = parse_node(q.model_dump(mode="json"), registry=reg)
        assert q.data.model_dump()["vella_repair"]["shadowed"] == "REAL"


def test_quarantine_reason_does_not_leak_input_values() -> None:
    reg, _DocData = _registry_with_doc()
    raw = {"type": "doc", "name": "n", "created_by": str(uuid4()), "data": {"ssn": "123-45-6789"}}
    q = parse_node(raw, registry=reg)
    reason = q.data.model_dump()["vella_repair"]["reason"]
    assert "123-45-6789" not in reason  # error types/locations only, never the value
